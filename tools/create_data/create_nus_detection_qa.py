"""Generate object-localization QA from nuScenes 3D annotations.

For each ``sample`` in nuScenes, project 3D bounding boxes into each of the 6
surround-view cameras using nuscenes-devkit, take the projected box-center, scale
it to the training-image space (default 448x448), and emit Q/A pairs in the same
HuggingFace ``Dataset`` schema as DriveLM (``id``, ``image_paths``, ``conversations``).

Answers use the standard DriveLM coordinate-tag format ``<cN,CAM_X,xxx,yyy>`` with
floating-point pixel values; the ``<loc_k>`` quantization happens later in the
collate function. So the produced dataset stays human-readable and is also
compatible with the original (non-loc-token) training path.

Example:
    python tools/create_data/create_nus_detection_qa.py \\
        --nuscenes-root data/DriveLM_nuScenes/nuscenes \\
        --version v1.0-trainval \\
        --out data/nus_detection_qa/split \\
        --img-size 448 \\
        --max-distance 60 \\
        --samples-per-frame 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import view_points, BoxVisibility
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "nuscenes-devkit is required: `pip install nuscenes-devkit`"
    ) from exc

from datasets import Dataset


CAMERAS: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


# Default category buckets. Each maps a friendly name to a list of nuScenes
# ``category.name`` prefixes; matching is by ``startswith`` so subclasses are
# included automatically. Override with --categories-json for sweeps.
DEFAULT_CATEGORY_GROUPS: Dict[str, List[str]] = {
    "car":        ["vehicle.car"],
    "truck":      ["vehicle.truck", "vehicle.construction"],
    "bus":        ["vehicle.bus"],
    "pedestrian": ["human.pedestrian"],
    "bicycle":    ["vehicle.bicycle"],
    "motorcycle": ["vehicle.motorcycle"],
}


CATEGORY_NAMES: Dict[str, Tuple[str, str]] = {
    "car":        ("cars", "car"),
    "truck":      ("trucks", "truck"),
    "bus":        ("buses", "bus"),
    "pedestrian": ("pedestrians", "pedestrian"),
    "bicycle":    ("bicycles", "bicycle"),
    "motorcycle": ("motorcycles", "motorcycle"),
}


DEFAULT_TEMPLATES = [
    "Where are the {plural} in this scene?",
    "List all visible {plural}.",
    "Locate every {singular} you can see.",
    "Identify the positions of all {plural}.",
]


DEFAULT_NO_OBJ_TEMPLATES = [
    "Are there any {plural} visible?",
]


def _classify(category_name: str, groups: Dict[str, List[str]]):
    for friendly, prefixes in groups.items():
        for p in prefixes:
            if category_name.startswith(p):
                return friendly
    return None


def _box_center_2d(box, K: np.ndarray, orig_size=(1600, 900)) -> Tuple[float, float] | None:
    """Project 8 box corners with the camera intrinsic, return the 2D bbox center.

    Skips boxes whose center is behind the camera (any corner with z<=0 in the
    camera frame is fine because we still take the average; but if the *box
    center* is behind the camera, the projection is meaningless — drop it).
    """
    if box.center[2] <= 0:
        return None
    corners_2d = view_points(box.corners(), K, normalize=True)[:2]  # (2, 8)
    x_min, y_min = corners_2d.min(axis=1)
    x_max, y_max = corners_2d.max(axis=1)
    cx_full = (x_min + x_max) / 2.0
    cy_full = (y_min + y_max) / 2.0
    return float(cx_full), float(cy_full)


def _scale_to_target(cx: float, cy: float, orig_size, img_size: int):
    return cx * img_size / orig_size[0], cy * img_size / orig_size[1]


def _resolve_image_paths(nusc: NuScenes, sample, dataroot: str) -> List[str] | None:
    paths = []
    for cam in CAMERAS:
        sd_token = sample["data"][cam]
        sd = nusc.get("sample_data", sd_token)
        full = os.path.join(dataroot, sd["filename"])
        if not os.path.isfile(full):
            return None
        paths.append(full)
    return paths


def _collect_per_cam_boxes(
    nusc: NuScenes,
    sample,
    *,
    img_size: int,
    max_distance: float,
    groups: Dict[str, List[str]],
) -> Dict[str, List[Tuple[str, float, float]]]:
    per_cam: Dict[str, List[Tuple[str, float, float]]] = {c: [] for c in CAMERAS}
    for cam in CAMERAS:
        sd_token = sample["data"][cam]
        try:
            _, boxes_in_cam, K = nusc.get_sample_data(
                sd_token, box_vis_level=BoxVisibility.ANY
            )
        except Exception:
            continue
        for box in boxes_in_cam:
            cat = _classify(box.name, groups)
            if cat is None:
                continue
            if float(np.linalg.norm(box.center)) > max_distance:
                continue
            center_2d = _box_center_2d(box, K)
            if center_2d is None:
                continue
            cx_full, cy_full = center_2d
            cx, cy = _scale_to_target(cx_full, cy_full, (1600, 900), img_size)
            if not (0.0 <= cx < img_size and 0.0 <= cy < img_size):
                continue
            per_cam[cam].append((cat, cx, cy))
    return per_cam


def _build_qa(
    per_cam: Dict[str, List[Tuple[str, float, float]]],
    templates: List[str],
    no_obj_templates: List[str],
    rng: random.Random,
) -> Tuple[str, str] | None:
    cats_present = {c for boxes in per_cam.values() for c, _, _ in boxes}
    if cats_present and rng.random() < 0.85:
        cat = rng.choice(sorted(cats_present))
        plural, singular = CATEGORY_NAMES.get(cat, (cat + "s", cat))
        q = rng.choice(templates).format(plural=plural, singular=singular)
        tags = []
        cid = 1
        for cam in CAMERAS:
            for c, x, y in per_cam[cam]:
                if c != cat:
                    continue
                tags.append(f"<c{cid},{cam},{x:.2f},{y:.2f}>")
                cid += 1
        a = ", ".join(tags) + "." if tags else "None."
        return q, a

    # Negative / "Are there any X" question
    cat = rng.choice(list(CATEGORY_NAMES.keys()))
    plural, singular = CATEGORY_NAMES[cat]
    q = rng.choice(no_obj_templates).format(plural=plural, singular=singular)
    tags, cid = [], 1
    for cam in CAMERAS:
        for c, x, y in per_cam[cam]:
            if c != cat:
                continue
            tags.append(f"<c{cid},{cam},{x:.2f},{y:.2f}>")
            cid += 1
    if tags:
        a = "Yes. " + ", ".join(tags) + "."
    else:
        a = "No."
    return q, a


def _split_scenes(nusc: NuScenes, val_ratio: float, seed: int):
    """Deterministic scene-level split (no train/val leak across scenes)."""
    scenes = sorted(nusc.scene, key=lambda s: s["name"])
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_val = max(1, int(len(scenes) * val_ratio))
    val_tokens = {s["token"] for s in scenes[:n_val]}
    return val_tokens


def generate(args):
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=False)

    if args.categories_json:
        with open(args.categories_json) as f:
            groups = json.load(f)
    else:
        groups = DEFAULT_CATEGORY_GROUPS

    templates = DEFAULT_TEMPLATES if not args.templates_json else json.load(open(args.templates_json))
    no_obj_templates = DEFAULT_NO_OBJ_TEMPLATES

    val_tokens = _split_scenes(nusc, args.val_ratio, args.seed)
    rng = random.Random(args.seed)

    train_rows, val_rows = [], []

    n_skipped_missing_imgs = 0
    for s_idx, sample in enumerate(nusc.sample):
        scene_token = sample["scene_token"]
        is_val = scene_token in val_tokens
        image_paths = _resolve_image_paths(nusc, sample, args.nuscenes_root)
        if image_paths is None:
            n_skipped_missing_imgs += 1
            continue

        per_cam = _collect_per_cam_boxes(
            nusc, sample,
            img_size=args.img_size,
            max_distance=args.max_distance,
            groups=groups,
        )

        for q_idx in range(args.samples_per_frame):
            qa = _build_qa(per_cam, templates, no_obj_templates, rng)
            if qa is None:
                continue
            q, a = qa
            row = {
                "id": f"{sample['token']}_det_{q_idx}",
                "image_paths": image_paths,
                "conversations": [
                    {"from": "human", "value": q},
                    {"from": "gpt",   "value": a},
                ],
            }
            (val_rows if is_val else train_rows).append(row)

        if (s_idx + 1) % 200 == 0:
            print(f"  processed {s_idx+1}/{len(nusc.sample)} samples "
                  f"(train={len(train_rows)}, val={len(val_rows)})")

    print(f"skipped {n_skipped_missing_imgs} samples (missing image files)")
    print(f"final: train={len(train_rows)}, val={len(val_rows)}")

    os.makedirs(args.out, exist_ok=True)
    Dataset.from_list(train_rows).save_to_disk(os.path.join(args.out, "train"))
    Dataset.from_list(val_rows).save_to_disk(os.path.join(args.out, "val"))
    print(f"saved → {args.out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes-root", required=True,
                        help="path to nuScenes root containing v1.0-* and samples/")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--out", required=True,
                        help="output directory (will hold train/ and val/ HF datasets)")
    parser.add_argument("--img-size", type=int, default=448)
    parser.add_argument("--max-distance", type=float, default=60.0,
                        help="drop boxes whose 3D center distance > this (meters)")
    parser.add_argument("--samples-per-frame", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories-json", default=None,
                        help="optional JSON mapping {friendly_name: [category prefixes]}")
    parser.add_argument("--templates-json", default=None,
                        help="optional JSON list of templates with {plural}/{singular} placeholders")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
