#!/usr/bin/env python3
"""
Load DriveVLM base checkpoint and apply AWQ int4 decoder delta package.
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

from tools.awq_int4_decoder import apply_awq_delta_to_model


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load AWQ int4 package and run a smoke check")
    p.add_argument(
        "--package_dir",
        type=str,
        required=True,
        help="Folder produced by tools/awq_int4_decoder.py --export_dir",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--text_only",
        action="store_true",
        help=(
            "Explicitly skip AutoProcessor and load AutoTokenizer only. "
            "Use this for text-only pipelines without audio/image processing."
        ),
    )
    return p


def _to_dtype(dtype_name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def main() -> None:
    args = _build_parser().parse_args()
    manifest_path = os.path.join(args.package_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    base_model_path = manifest["base_model_path"]
    delta_path = os.path.join(args.package_dir, manifest["awq_delta"])
    dtype = _to_dtype(args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        _attn_implementation="flash_attention_2",
    )
    apply_awq_delta_to_model(model, delta_path, map_location="cpu")
    model.to(args.device)
    model.eval()

    print("[awq-load] model loaded and delta applied.")
    print(f"[awq-load] class={type(model).__name__}, layers={len(model.model.layers)}")
    if args.text_only:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        print("[awq-load] text-only mode enabled: skipped AutoProcessor by request.")
        print(f"[awq-load] tokenizer_size={len(tokenizer)}")
    else:
        try:
            processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
            print(f"[awq-load] tokenizer_size={len(processor.tokenizer)}")
        except Exception as exc:
            raise RuntimeError(
                "AutoProcessor initialization failed in strict mode. "
                "This usually means audio/image processor config is incomplete for this checkpoint. "
                "If your workflow is text-only (no audio/image), rerun with --text_only."
            ) from exc


if __name__ == "__main__":
    main()
