#!/usr/bin/env python
"""Drive a real unsloth MLX training run on a local MLX 4-bit model.

This is the t3-2/t3-3 reproduction harness: it exercises unsloth's public
Apple-Silicon path (FastLanguageModel -> get_peft_model -> UnslothTrainer)
against one of the three target models and reports the first real failure.
"""
import argparse, os, sys, time, traceback

MODEL = os.environ.get("MODEL", os.path.expanduser("~/MLXLoRAStudio/models/Qwen3.5-9B-DefiantFable-4bit"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lora", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--out", default=os.path.expanduser("~/mlx-lab-runs/unsloth-mlx/out"))
    args = ap.parse_args()

    import unsloth
    from unsloth import FastLanguageModel
    print(f"[run] unsloth {unsloth.__version__} DEVICE_TYPE={unsloth.DEVICE_TYPE}", flush=True)

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/no_robots", split="train[:64]")

    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        dtype=None,
    )
    print(f"[run] loaded in {time.time()-t0:.1f}s", flush=True)

    # NOTE: unsloth's get_peft_model takes LoRA kwargs, NOT a peft.LoraConfig
    # object -- verified identical on the CUDA path (llama.py binds the 2nd
    # positional to `r`), so this is not an Apple-Silicon-specific gap.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora,
        lora_alpha=args.lora * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing="mlx",
    )
    print("[run] peft model ready", flush=True)

    from unsloth import UnslothTrainer
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    cfg = MLXTrainingConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=args.steps,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=args.steps,
        output_dir=args.out,
        report_to=[],
        optim="adamw",
        seed=42,
        max_seq_length=args.max_seq,
    )

    def fmt(example):
        msgs = example.get("messages") or []
        if msgs:
            return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}
        return {"text": example.get("prompt", "") + example.get("completion", "")}

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        args=cfg,
        train_dataset=ds,
        dataset_text_field="text",
    )
    res = trainer.train()
    print(f"[run] TRAIN OK -> {type(res).__name__} {res}", flush=True)
    try:
        model.save_pretrained(os.path.join(args.out, "adapter"))
        print("[run] adapter saved", flush=True)
    except Exception:
        traceback.print_exc()
    print("[run] DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
