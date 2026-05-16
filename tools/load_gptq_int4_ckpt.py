#!/usr/bin/env python3
"""
Load DriveVLM base checkpoint and apply GPTQ int4 decoder delta package.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.gptq_int4_decoder import apply_gptq_delta_to_model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load GPTQ int4 package and run a smoke check")
    p.add_argument("--package_dir", type=str, required=True, help="Folder produced by gptq_int4_decoder.py")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16", choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--text_only",
        action="store_true",
        help="Skip AutoProcessor and only load AutoTokenizer for text-only validation.",
    )
    return p


def _to_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def main() -> None:
    args = _build_parser().parse_args()
    manifest_path = os.path.join(args.package_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for key in ("base_model_path", "gptq_delta"):
        if key not in manifest:
            raise KeyError(f"manifest missing required field: {key}")
    base_model_path = manifest["base_model_path"]
    delta_path = os.path.join(args.package_dir, manifest["gptq_delta"])
    if not os.path.exists(delta_path):
        raise FileNotFoundError(f"gptq delta not found: {delta_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=_to_dtype(args.dtype),
        _attn_implementation="flash_attention_2",
    )
    model.to(args.device)
    apply_gptq_delta_to_model(model, delta_path, map_location="cpu")
    model.eval()

    print("[gptq-load] model loaded and delta applied.")
    print(f"[gptq-load] class={type(model).__name__}, layers={len(model.model.layers)}")
    if args.text_only:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        print(f"[gptq-load] tokenizer_size={len(tokenizer)}")
    else:
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        print(f"[gptq-load] tokenizer_size={len(processor.tokenizer)}")


if __name__ == "__main__":
    main()

