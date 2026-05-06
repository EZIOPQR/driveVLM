"""Batched inference for DriveLM (Phi-4 path).

Mirrors tools/inference.py but exposes --batch_size and --max_new_tokens, and
relies on the (now batch-capable) drivelm_nus_phi4_collate_fn_val to feed
left-padded inputs to model.generate().

Output schema is identical to tools/inference.py to keep tools/evaluation.py
working without changes:
    [{"id": str, "question": str, "answer": str}, ...]
"""
import argparse
import json
from functools import partial

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from drivevlms.build import build_collate_fn


@torch.no_grad()
def main(args):
    base_model = "/root/autodl-tmp/models/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()
    generation_config = GenerationConfig.from_pretrained(base_model)

    collate_fn = build_collate_fn(args.collate_fn)
    val_collate_fn = partial(
        collate_fn,
        processor=processor,
        dtype=torch.bfloat16,
        use_optical_flow=args.use_optical_flow,
        flow_root=args.flow_root or "",
        flow_scale_u=args.flow_scale_u,
        flow_scale_v=args.flow_scale_v,
    )

    dataset = load_from_disk(args.data)
    total = len(dataset)
    if args.limit is not None and args.limit > 0:
        n = min(args.limit, total)
        dataset = Subset(dataset, list(range(n)))
        print(f"[inference_batch] --limit={args.limit} -> running on first {n}/{total} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=val_collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(f"[inference_batch] batch_size={args.batch_size}, "
          f"max_new_tokens={args.max_new_tokens}, "
          f"num_batches={len(dataloader)}")

    data_dict = []
    for batch in tqdm(dataloader):
        inputs, questions, ids = batch
        inputs = inputs.to(args.device)
        input_len = inputs["input_ids"].shape[-1]
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            generation_config=generation_config,
        )
        # Same input_len for every sample in batch (left padding aligns them).
        output = output[:, input_len:]
        answers = processor.batch_decode(output, skip_special_tokens=True)

        assert len(answers) == len(ids) == len(questions), (
            f"len mismatch: answers={len(answers)} ids={len(ids)} questions={len(questions)}"
        )
        for sid, q, a in zip(ids, questions, answers):
            data_dict.append({"id": sid, "question": q, "answer": a})

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="DriveLM Batched Inference (Phi-4)")
    parser.add_argument("--data", type=str, default="data/DriveLM_nuScenes/split/val")
    parser.add_argument(
        "--collate_fn", type=str, default="drivelm_nus_phi4_collate_fn_val",
        help="Must point at a val collate fn that supports batch>1. "
             "drivelm_nus_phi4_collate_fn_val is now batch-capable.",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/DriveLM_nuScenes/refs/infer_results_batch.json",
    )
    parser.add_argument(
        "--model", type=str,
        default="/root/autodl-tmp/models/Phi-4-multimodal-instruct",
        help="Path to model or fine-tuned checkpoint",
    )
    parser.add_argument("--device", default="cuda", help="Device to run inference")
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="DataLoader batch size. Phi-4 bf16 + 6*448 imgs: 4 fits ~24GB, 8 needs more.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Generation cap. 256 is plenty for DriveLM answers; lower if you want more speed.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run on the first N samples (for smoke test). Default: full set.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--use_optical_flow",
        action="store_true",
        help="5-channel SigLIP input; run compute_flow_from_sweeps.py and set --flow-root.",
    )
    parser.add_argument(
        "--flow_root",
        type=str,
        default="",
        help="Root with CAM_*/<jpg_stem>.npz (default: config flow_root when training).",
    )
    parser.add_argument(
        "--flow_scale_u",
        type=float,
        default=8.778,
        help="Normalization divisor for flow u channel.",
    )
    parser.add_argument(
        "--flow_scale_v",
        type=float,
        default=2.888,
        help="Normalization divisor for flow v channel.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
