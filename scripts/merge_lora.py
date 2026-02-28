import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", default="base_model", help="Path to base HF model")
    parser.add_argument("--adapter", default="adapter/qlora_ecom_200", help="Path to LoRA adapter")
    parser.add_argument("--out-dir", default="merged_model", help="Output directory for merged model")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype for loading base model",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to HF loaders",
    )
    parser.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save model with safetensors",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    peft = PeftModel.from_pretrained(base, args.adapter)
    merged = peft.merge_and_unload()

    merged.save_pretrained(args.out_dir, safe_serialization=args.safe_serialization)
    tok.save_pretrained(args.out_dir)
    print("Merged model saved to", args.out_dir)


if __name__ == "__main__":
    main()
