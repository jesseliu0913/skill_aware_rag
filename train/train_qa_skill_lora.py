#!/usr/bin/env python
"""LoRA SFT for normalized QA skill data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


DEFAULT_MODEL = (
    "/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)


class JsonlChatDataset(Dataset):
    def __init__(self, path: Path, tokenizer: Any, max_length: int, limit: int = 0) -> None:
        self.rows = []
        self.skipped_truncated_answer = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                if limit and len(self.rows) >= limit:
                    break
                record = json.loads(line)
                user_messages = [record["messages"][0]]
                prompt_text = tokenizer.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                text = tokenizer.apply_chat_template(record["messages"], tokenize=False)
                tokenized = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                )
                prompt_ids = tokenizer(
                    prompt_text,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"]
                input_ids = tokenized["input_ids"]
                labels = list(input_ids)
                labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
                if all(label == -100 for label in labels):
                    self.skipped_truncated_answer += 1
                    continue
                self.rows.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": tokenized["attention_mask"],
                        "labels": labels,
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.rows[idx]


class CausalCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train", default="outputs/qa_skill_data/train.jsonl")
    parser.add_argument("--dev", default="outputs/qa_skill_data/dev.jsonl", help="Optional dev JSONL. Use '' to disable evaluation during training.")
    parser.add_argument("--output-dir", default="outputs/qa_skill_lora/qwen2_5_7b")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Optional Trainer checkpoint path, e.g. output-dir/checkpoint-375.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = JsonlChatDataset(Path(args.train), tokenizer, args.max_length, args.train_limit)
    dev_dataset = None
    if args.dev:
        dev_path = Path(args.dev)
        if not dev_path.exists():
            raise FileNotFoundError(f"Dev file not found: {dev_path}. Use --dev '' to disable training-time eval.")
        dev_dataset = JsonlChatDataset(dev_path, tokenizer, args.max_length, args.dev_limit)
    print(
        f"Loaded {len(train_dataset)} train examples "
        f"(skipped {train_dataset.skipped_truncated_answer} truncated-answer examples)"
    )
    if dev_dataset is not None:
        print(
            f"Loaded {len(dev_dataset)} dev examples "
            f"(skipped {dev_dataset.skipped_truncated_answer} truncated-answer examples)"
        )
    else:
        print("No dev set provided; disabling training-time evaluation")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps if dev_dataset is not None else None,
        eval_strategy="steps" if dev_dataset is not None else "no",
        save_strategy="steps",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
