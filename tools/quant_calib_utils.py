#!/usr/bin/env python3
"""
Utilities for quantization calibration sampling and tag mapping.
"""

import json
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def _normalize_tag(raw_tag) -> List[int]:
    if isinstance(raw_tag, int):
        return [raw_tag]
    if isinstance(raw_tag, list):
        out = []
        for x in raw_tag:
            if isinstance(x, int):
                out.append(x)
        return out
    return []


def _primary_tag(tags: List[int], coord_tag: int) -> int:
    if coord_tag in tags:
        return coord_tag
    for t in (0, 1, 2, 3):
        if t in tags:
            return t
    raise ValueError(f"Unable to infer primary tag from tags={tags}")


def build_id_to_primary_tag(tag_ref_json: str, coord_tag: int = 3) -> Dict[str, int]:
    with open(tag_ref_json, "r", encoding="utf-8") as f:
        refs = json.load(f)

    id_to_tag: Dict[str, int] = {}
    for frame in refs:
        scene_id = frame["scene_id"]
        frame_id = frame["frame_id"]
        qa = frame["QA"]
        qa_list = qa["perception"] + qa["prediction"] + qa["planning"] + qa["behavior"]
        for idx, qa_item in enumerate(qa_list):
            qid = f"{scene_id}_{frame_id}_{idx}"
            tags = _normalize_tag(qa_item.get("tag"))
            if not tags:
                continue
            id_to_tag[qid] = _primary_tag(tags, coord_tag=coord_tag)
    return id_to_tag


def _allocate_targets(
    tags: Iterable[int],
    total: int,
    coord_tag: int,
    coord_ratio: float,
) -> Dict[int, int]:
    uniq = sorted(set(tags))
    if not uniq:
        raise ValueError("No tag pools available for calibration sampling.")
    if total <= 0:
        raise ValueError("total must be > 0")
    if coord_ratio <= 0.0 or coord_ratio >= 1.0:
        raise ValueError("coord_ratio must be in (0, 1).")

    targets = {t: 0 for t in uniq}
    if coord_tag in uniq and len(uniq) > 1:
        coord_n = int(round(total * coord_ratio))
        coord_n = min(max(1, coord_n), total - 1)
        others = [t for t in uniq if t != coord_tag]
        rem = total - coord_n
        each = rem // len(others)
        extra = rem % len(others)
        for i, t in enumerate(others):
            targets[t] = each + (1 if i < extra else 0)
        targets[coord_tag] = coord_n
    else:
        each = total // len(uniq)
        extra = total % len(uniq)
        for i, t in enumerate(uniq):
            targets[t] = each + (1 if i < extra else 0)
    return targets


def select_balanced_calibration_indices(
    dataset,
    tag_ref_json: str,
    calib_samples: int,
    coord_tag: int = 3,
    coord_ratio: float = 0.4,
    seed: int = 42,
) -> Tuple[List[int], Dict[int, int]]:
    id_to_tag = build_id_to_primary_tag(tag_ref_json=tag_ref_json, coord_tag=coord_tag)
    pools: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        qid = item.get("id", "")
        if qid in id_to_tag:
            pools[id_to_tag[qid]].append(idx)

    if not pools:
        raise RuntimeError(
            f"No calibration candidates matched tags from {tag_ref_json}. "
            "Check dataset IDs and tag reference file."
        )

    rng = random.Random(seed)
    for v in pools.values():
        rng.shuffle(v)

    targets = _allocate_targets(
        tags=pools.keys(),
        total=calib_samples,
        coord_tag=coord_tag,
        coord_ratio=coord_ratio,
    )
    selected: List[int] = []
    used = set()
    selected_counts = {t: 0 for t in targets}
    deficits = 0

    for t, tgt in targets.items():
        take = pools[t][:tgt]
        selected.extend(take)
        used.update(take)
        selected_counts[t] = len(take)
        deficits += max(0, tgt - len(take))

    if deficits > 0:
        fallback: List[int] = []
        for t in sorted(pools.keys()):
            for idx in pools[t]:
                if idx not in used:
                    fallback.append(idx)
        selected.extend(fallback[:deficits])

    if len(selected) < calib_samples:
        raise RuntimeError(
            f"Unable to collect enough calibration samples: requested={calib_samples}, got={len(selected)}."
        )

    return selected[:calib_samples], selected_counts

