#!/usr/bin/env python
"""Run local HF causal LM inference over normalized QA skill JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


DEFAULT_MODEL = (
    "/playpen-shared/jesse/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default="", help="Optional PEFT/LoRA adapter directory.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return parser.parse_args()


def dtype_from_arg(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def iter_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit and len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = iter_records(Path(args.input), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype_from_arg(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=True,
        local_files_only=True,
    )
    if args.adapter:
        if PeftModel is None:
            raise ImportError("peft is required to load --adapter")
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.eval()

    do_sample = args.temperature > 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, record in enumerate(records):
            prompt = record["prompt"]
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated_ids[:, inputs.input_ids.shape[1] :]
            prediction = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            out_record = {
                "id": record["id"],
                "dataset": record["source"],
                "task_type": record["task_type"],
                "question": record["question"],
                "input": record["input_molecule_or_context"],
                "input_type": record["input_type"],
                "decoded_smiles": record.get("decoded_smiles"),
                "reference_output": record["gold_answer"],
                "prediction": prediction,
                "prompt": prompt,
                "retrieved_kg_evidence": record.get("retrieved_kg_evidence", []),
                "retrieved_molecule_evidence": record.get("retrieved_molecule_evidence", []),
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"Wrote {idx + 1}/{len(records)}: {record['source']} {record['id']}", flush=True)


if __name__ == "__main__":
    main()
