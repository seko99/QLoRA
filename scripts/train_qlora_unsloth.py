import argparse
from pathlib import Path

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QLoRA adapter with Unsloth + TRL SFTTrainer")
    parser.add_argument("--model-dir", default="base_model", help="Base model directory")
    parser.add_argument("--data-path", default="data/dataset_200.jsonl", help="Training JSONL path")
    parser.add_argument("--out-dir", default="adapter/qlora_ecom_200", help="Output adapter directory")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable 4-bit model loading",
    )

    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target-modules",
        default=DEFAULT_TARGET_MODULES,
        help="Comma-separated list of LoRA target modules",
    )

    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--num-epochs", type=float, default=2.0, help="Number of train epochs")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint save steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--dataset-num-proc", type=int, default=1, help="Dataset tokenization workers")
    parser.add_argument("--report-to", default="none", help="Reporting backend")
    parser.add_argument("--optim", default="paged_adamw_8bit", help="Optimizer name")
    parser.add_argument("--lr-scheduler-type", default="cosine", help="LR scheduler type")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # Keep explicit Qwen-compatible special tokens to avoid placeholder EOS issues in TRL/Transformers.
    tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer eos_token_id is None for token '<|im_end|>'")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is None for token '<|endoftext|>'")

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing=True,
    )

    ds = load_dataset("json", data_files=args.data_path, split="train")

    def to_text(ex):
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds = ds.map(to_text, remove_columns=["messages"])

    # Use SFTConfig class from SFTTrainer namespace to avoid isinstance mismatch in patched environments.
    trainer_sft_config = SFTTrainer.__init__.__globals__.get("SFTConfig", SFTConfig)
    sft_args = trainer_sft_config(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=False,
        bf16=True,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        report_to=args.report_to,
        dataset_text_field="text",
        dataset_num_proc=args.dataset_num_proc,
        max_length=args.max_seq_length,
        pad_token=tokenizer.pad_token,
    )

    # Work around eos placeholder serialization in this TRL/Transformers stack.
    sft_args.eos_token = None

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=sft_args,
    )

    trainer.train()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("Saved adapter to:", out_dir)


if __name__ == "__main__":
    main()
