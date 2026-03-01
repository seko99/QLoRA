import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SampleResult:
    idx: int
    user: str
    answer: str
    found_count: int
    total_count: int
    hit_all: bool
    missing: list[str]
    matched_via: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base_model vs merged_model on eval set")
    parser.add_argument("--base-model", default="base_model", help="Path to base model (HF format)")
    parser.add_argument("--tuned-model", default="merged_model", help="Path to tuned/merged model (HF format)")
    parser.add_argument("--eval-path", default="data/pizza_eval_set.jsonl", help="Eval JSONL path")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of eval samples (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=180, help="Max generated tokens per answer")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to HF loaders",
    )
    parser.add_argument("--report-out", default="data/eval_report.json", help="Where to write JSON report")
    parser.add_argument("--min-tuned-hit-all", type=float, default=0.70, help="Minimum tuned hit_all rate")
    parser.add_argument("--min-delta-hit-all", type=float, default=0.20, help="Minimum tuned-base hit_all delta")
    parser.add_argument(
        "--print-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print sample-level failures for tuned model",
    )
    return parser.parse_args()


def pick_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_eval(path: str, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "user" not in rec:
                raise ValueError("Eval row must contain key 'user'")
            if "expected_json" not in rec and "expected_contains" not in rec:
                raise ValueError("Eval row must contain 'expected_json' or 'expected_contains'")
            rows.append(rec)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError("Eval dataset is empty")
    return rows


def normalize(text: str) -> str:
    return str(text).strip().casefold()


def normalize_sla(text: str) -> str:
    m = re.search(r"\d+", str(text))
    return m.group(0) if m else normalize(text)


def build_inputs(tokenizer, prompt: str, system_prompt: str | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if tokenizer.chat_template:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int, system_prompt: str | None) -> str:
    device = model.device
    input_ids, attention_mask = build_inputs(tokenizer, prompt, system_prompt, device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True).strip()


def extract_first_json_obj(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = []
    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)
    candidates += re.findall(r"\{[\s\S]*?\}", text)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def extract_kv_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    patterns = {
        "code": r"code\s*[:=]\s*([^\n,;]+)",
        "sla_min": r"sla(?:_min)?\s*[:=]\s*([^\n,;]+)",
        "escalation": r"escalation\s*[:=]\s*([^\n,;]+)",
        "log_key": r"log_key\s*[:=]\s*([^\n,;]+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            fields[k] = m.group(1).strip().strip("`\"'")
    return fields


def compare_expected_json(answer: str, expected: dict[str, Any]) -> tuple[int, list[str], str]:
    parsed = extract_first_json_obj(answer)
    matched_via = "json" if parsed is not None else "kv"
    if parsed is None:
        parsed = extract_kv_fields(answer)

    missing: list[str] = []
    for k, exp in expected.items():
        got = parsed.get(k, "")
        if k == "sla_min":
            ok = normalize_sla(got) == normalize_sla(exp)
        else:
            ok = normalize(got) == normalize(exp)
        if not ok:
            missing.append(str(exp))
    found_count = len(expected) - len(missing)
    return found_count, missing, matched_via


def compare_expected_contains(answer: str, expected_contains: list[str]) -> tuple[int, list[str], str]:
    answer_norm = normalize(answer)
    missing = [x for x in expected_contains if normalize(x) not in answer_norm]
    return len(expected_contains) - len(missing), missing, "contains"


def score_sample(answer: str, rec: dict[str, Any], idx: int) -> SampleResult:
    if "expected_json" in rec:
        found_count, missing, matched_via = compare_expected_json(answer, rec["expected_json"])
        total = len(rec["expected_json"])
    else:
        expected_contains = rec.get("expected_contains", [])
        found_count, missing, matched_via = compare_expected_contains(answer, expected_contains)
        total = len(expected_contains)
    return SampleResult(
        idx=idx,
        user=rec["user"],
        answer=answer,
        found_count=found_count,
        total_count=total,
        hit_all=(len(missing) == 0),
        missing=missing,
        matched_via=matched_via,
    )


def evaluate_model(
    model_path: str,
    eval_rows: list[dict[str, Any]],
    max_new_tokens: int,
    device_map: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> list[SampleResult]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    results: list[SampleResult] = []
    for i, rec in enumerate(eval_rows):
        answer = generate_answer(
            model,
            tokenizer,
            rec["user"],
            max_new_tokens=max_new_tokens,
            system_prompt=rec.get("system"),
        )
        results.append(score_sample(answer=answer, rec=rec, idx=i))
        if (i + 1) % 10 == 0 or (i + 1) == len(eval_rows):
            print(f"{model_path}: processed {i + 1}/{len(eval_rows)}")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def summarize(results: list[SampleResult]) -> dict[str, float]:
    n = len(results)
    hit_all = sum(1 for r in results if r.hit_all) / n
    partial = sum(r.found_count / max(r.total_count, 1) for r in results) / n
    return {"n": float(n), "hit_all_rate": hit_all, "partial_match_rate": partial}


def as_serializable(results: list[SampleResult]) -> list[dict[str, Any]]:
    return [
        {
            "idx": r.idx,
            "user": r.user,
            "answer": r.answer,
            "found_count": r.found_count,
            "total_count": r.total_count,
            "hit_all": r.hit_all,
            "missing": r.missing,
            "matched_via": r.matched_via,
        }
        for r in results
    ]


def main() -> None:
    args = parse_args()
    eval_rows = load_eval(args.eval_path, args.max_samples)
    dtype = pick_dtype(args.dtype)
    print(f"eval samples: {len(eval_rows)} | dtype: {dtype} | device_map: {args.device_map}")

    base_results = evaluate_model(
        model_path=args.base_model,
        eval_rows=eval_rows,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    tuned_results = evaluate_model(
        model_path=args.tuned_model,
        eval_rows=eval_rows,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )

    base_summary = summarize(base_results)
    tuned_summary = summarize(tuned_results)
    delta_hit_all = tuned_summary["hit_all_rate"] - base_summary["hit_all_rate"]
    delta_partial = tuned_summary["partial_match_rate"] - base_summary["partial_match_rate"]
    success = tuned_summary["hit_all_rate"] >= args.min_tuned_hit_all and delta_hit_all >= args.min_delta_hit_all

    print("\n=== Evaluation Summary ===")
    print(f"base  hit_all_rate:    {base_summary['hit_all_rate']:.3f}")
    print(f"tuned hit_all_rate:    {tuned_summary['hit_all_rate']:.3f}")
    print(f"delta hit_all_rate:    {delta_hit_all:+.3f}")
    print(f"base  partial_match:   {base_summary['partial_match_rate']:.3f}")
    print(f"tuned partial_match:   {tuned_summary['partial_match_rate']:.3f}")
    print(f"delta partial_match:   {delta_partial:+.3f}")
    print(f"verdict: {'SUCCESS' if success else 'NOT SUCCESS'}")

    if args.print_failures:
        print("\n=== Tuned model failures (missing expected markers) ===")
        failed = [r for r in tuned_results if not r.hit_all]
        if not failed:
            print("none")
        else:
            for r in failed:
                print(f"- #{r.idx}: missing={r.missing} | via={r.matched_via} | user={r.user}")

    report = {
        "config": {
            "base_model": args.base_model,
            "tuned_model": args.tuned_model,
            "eval_path": args.eval_path,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "device_map": args.device_map,
            "dtype": str(dtype),
            "min_tuned_hit_all": args.min_tuned_hit_all,
            "min_delta_hit_all": args.min_delta_hit_all,
        },
        "summary": {
            "base": base_summary,
            "tuned": tuned_summary,
            "delta_hit_all_rate": delta_hit_all,
            "delta_partial_match_rate": delta_partial,
            "success": success,
        },
        "base_results": as_serializable(base_results),
        "tuned_results": as_serializable(tuned_results),
    }
    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport saved to: {out_path}")


if __name__ == "__main__":
    main()
