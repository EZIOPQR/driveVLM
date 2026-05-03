"""Build a "next-frames" triplet dataset on top of DriveLM-nuScenes.

For each keyframe referenced by DriveLM-nuScenes's ``v1_1_train_nus.json``,
this script walks the nuScenes ``sample.next`` link twice to obtain the
next keyframe (``t+1``) and the frame after that (``t+2``). It then emits:

* ``next_frames_triplets.json`` — one entry per (current, t+1, t+2) triplet,
  with 6-camera image paths for each of the three keyframes and the
  associated sample / scene tokens. Samples where ``t+1`` or ``t+2`` does
  not exist (i.e. the keyframe is at the end of the scene) are dropped, as
  requested.
* ``next_frames_missing_images.txt`` — one path per line, listing every image
  file that the triplet dataset references but that is *not* present under
  ``data/DriveLM_nuScenes/nuscenes/samples/``. Use this list to fetch the
  missing files from the nuScenes blobs.

Prerequisites
-------------
The script needs the nuScenes trainval **metadata** (json tables) unpacked
into ``data/DriveLM_nuScenes/nuscenes/v1.0-trainval/``. The metadata tarball
is ``v1.0-trainval_meta.tgz`` on https://www.nuscenes.org/download (~400 MB).
It does *not* need the image/lidar blobs to run — those are only required
later, at training/inference time, and the script will tell you exactly
which files you still need.

Expected layout after unpacking metadata::

    data/DriveLM_nuScenes/nuscenes/
      samples/CAM_FRONT/...            # already present (DriveLM subset)
      samples/CAM_FRONT_LEFT/...
      ...
      v1.0-trainval/
        sample.json
        sample_data.json
        scene.json
        log.json
        ...

Usage
-----
.. code-block:: bash

    python tools/create_data/build_next_frames.py \
        data/DriveLM_nuScenes/QA_dataset_nus/v1_1_train_nus.json \
        --nuscenes-root data/DriveLM_nuScenes/nuscenes \
        --out-dir data/DriveLM_nuScenes/refs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

CAMERA_CHANNELS: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def load_json_table(path: str) -> list:
    """Load a nuScenes ``*.json`` table (a JSON array of records)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sample_index(sample_records: list) -> Dict[str, dict]:
    """Map ``sample_token -> {scene_token, next, prev, timestamp}``."""
    return {rec["token"]: rec for rec in sample_records}


def build_keyframe_image_index(
    sample_data_records: list,
) -> Dict[Tuple[str, str], str]:
    """Map ``(sample_token, channel) -> relative filename`` for keyframes only.

    Only rows with ``is_key_frame == True`` are kept, and only rows whose
    channel is one of the six surround cameras. This matches the layout of
    ``samples/<CHANNEL>/...``.
    """
    channel_set = set(CAMERA_CHANNELS)
    sensor_channels: Dict[str, str] = {}
    index: Dict[Tuple[str, str], str] = {}

    for rec in sample_data_records:
        if not rec.get("is_key_frame", False):
            continue
        filename = rec.get("filename", "")
        if not filename.startswith("samples/"):
            continue
        parts = filename.split("/")
        if len(parts) < 3:
            continue
        channel = parts[1]
        if channel not in channel_set:
            continue
        index[(rec["sample_token"], channel)] = filename
        sensor_channels[rec["sample_token"]] = channel
    return index


def resolve_image_paths(
    sample_token: str,
    keyframe_image_index: Dict[Tuple[str, str], str],
    nuscenes_root_rel: str,
) -> Dict[str, str]:
    """Return a ``{channel: relative_path}`` dict for all 6 cameras.

    The returned paths are relative to the DriveLM repo root so that they
    plug into the existing DriveLM ``image_paths`` convention (e.g.
    ``../nuscenes/samples/CAM_FRONT/xxx.jpg`` when consumed from
    ``data/DriveLM_nuScenes/refs/``).
    """
    paths: Dict[str, str] = {}
    for cam in CAMERA_CHANNELS:
        fn = keyframe_image_index.get((sample_token, cam))
        if fn is None:
            paths[cam] = ""
        else:
            paths[cam] = f"{nuscenes_root_rel}/{fn}"
    return paths


def next_two_samples(
    sample_token: str, sample_index: Dict[str, dict]
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(t+1_token, t+2_token)`` or ``(None, None)`` at scene end."""
    rec = sample_index.get(sample_token)
    if rec is None:
        return None, None
    t1 = rec.get("next") or None
    if not t1:
        return None, None
    rec1 = sample_index.get(t1)
    if rec1 is None:
        return None, None
    t2 = rec1.get("next") or None
    if not t2:
        return t1, None
    return t1, t2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "src",
        type=str,
        help="Path to DriveLM-nuScenes QA JSON (e.g. v1_1_train_nus.json).",
    )
    parser.add_argument(
        "--nuscenes-root",
        type=str,
        default="data/DriveLM_nuScenes/nuscenes",
        help=(
            "Directory containing 'samples/' and 'v1.0-trainval/' metadata. "
            "Default: data/DriveLM_nuScenes/nuscenes"
        ),
    )
    parser.add_argument(
        "--metadata-version",
        type=str,
        default="v1.0-trainval",
        help="Name of the metadata sub-directory. Default: v1.0-trainval",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/DriveLM_nuScenes/refs",
        help="Output directory. Default: data/DriveLM_nuScenes/refs",
    )
    parser.add_argument(
        "--nuscenes-rel-prefix",
        type=str,
        default="../nuscenes",
        help=(
            "Prefix used in the emitted image paths. Matches the DriveLM "
            "convention where paths live under refs/*.json and reference "
            "'../nuscenes/samples/...'. Default: ../nuscenes"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    meta_dir = os.path.join(args.nuscenes_root, args.metadata_version)
    sample_json = os.path.join(meta_dir, "sample.json")
    sample_data_json = os.path.join(meta_dir, "sample_data.json")

    if not (os.path.isfile(sample_json) and os.path.isfile(sample_data_json)):
        print(
            "[ERROR] nuScenes metadata not found at: {}\n"
            "        Expected files: sample.json, sample_data.json\n\n"
            "To obtain them:\n"
            "  1. Go to https://www.nuscenes.org/download (requires a free account).\n"
            "  2. Download 'Full dataset (v1.0) -> Metadata' → "
            "v1.0-trainval_meta.tgz (~400 MB).\n"
            "  3. Extract into: {}\n".format(meta_dir, args.nuscenes_root),
            file=sys.stderr,
        )
        return 2

    print(f"[info] Loading nuScenes metadata from: {meta_dir}")
    sample_records = load_json_table(sample_json)
    sample_data_records = load_json_table(sample_data_json)
    print(
        f"[info]   sample.json: {len(sample_records)} rows, "
        f"sample_data.json: {len(sample_data_records)} rows"
    )

    sample_index = build_sample_index(sample_records)
    keyframe_image_index = build_keyframe_image_index(sample_data_records)
    print(
        f"[info]   keyframe image index built: "
        f"{len(keyframe_image_index)} (sample_token, channel) entries"
    )

    print(f"[info] Loading DriveLM QA from: {args.src}")
    with open(args.src, "r", encoding="utf-8") as f:
        drivelm = json.load(f)
    num_scenes = len(drivelm)
    num_keyframes = sum(len(v.get("key_frames", {})) for v in drivelm.values())
    print(f"[info]   scenes: {num_scenes}, keyframes referenced: {num_keyframes}")

    triplets: List[dict] = []
    dropped_no_next = 0
    dropped_no_next_next = 0
    dropped_token_unknown = 0

    for scene_token, scene in drivelm.items():
        for sample_token in scene.get("key_frames", {}):
            if sample_token not in sample_index:
                dropped_token_unknown += 1
                continue
            t1, t2 = next_two_samples(sample_token, sample_index)
            if t1 is None:
                dropped_no_next += 1
                continue
            if t2 is None:
                dropped_no_next_next += 1
                continue

            triplets.append(
                {
                    "scene_token": scene_token,
                    "t0": {
                        "sample_token": sample_token,
                        "image_paths": resolve_image_paths(
                            sample_token,
                            keyframe_image_index,
                            args.nuscenes_rel_prefix,
                        ),
                    },
                    "t1": {
                        "sample_token": t1,
                        "image_paths": resolve_image_paths(
                            t1, keyframe_image_index, args.nuscenes_rel_prefix
                        ),
                    },
                    "t2": {
                        "sample_token": t2,
                        "image_paths": resolve_image_paths(
                            t2, keyframe_image_index, args.nuscenes_rel_prefix
                        ),
                    },
                }
            )

    print(
        "[info] Triplet summary: kept={}, dropped_no_next={}, "
        "dropped_no_next_next={}, dropped_token_unknown={}".format(
            len(triplets),
            dropped_no_next,
            dropped_no_next_next,
            dropped_token_unknown,
        )
    )

    os.makedirs(args.out_dir, exist_ok=True)
    triplets_path = os.path.join(args.out_dir, "next_frames_triplets.json")
    with open(triplets_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    print(f"[info] Wrote triplet dataset: {triplets_path}")

    required_files: Dict[str, str] = {}
    for entry in triplets:
        for slot in ("t0", "t1", "t2"):
            for _, rel_path in entry[slot]["image_paths"].items():
                if not rel_path:
                    continue
                fn = rel_path[len(args.nuscenes_rel_prefix) + 1 :]
                required_files[fn] = os.path.join(args.nuscenes_root, fn)

    missing: List[str] = []
    missing_by_channel: Dict[str, int] = defaultdict(int)
    for fn, abs_path in required_files.items():
        if not os.path.isfile(abs_path):
            missing.append(fn)
            parts = fn.split("/")
            if len(parts) >= 2:
                missing_by_channel[parts[1]] += 1

    missing_path = os.path.join(args.out_dir, "next_frames_missing_images.txt")
    with open(missing_path, "w", encoding="utf-8") as f:
        for fn in sorted(missing):
            f.write(fn + "\n")

    total_required = len(required_files)
    present = total_required - len(missing)
    print(
        "[info] Image presence: required={}, already_local={}, missing={}".format(
            total_required, present, len(missing)
        )
    )
    if missing_by_channel:
        print("[info] Missing images per camera channel:")
        for ch in CAMERA_CHANNELS:
            if ch in missing_by_channel:
                print(f"         {ch:15s} {missing_by_channel[ch]}")
    print(f"[info] Wrote missing-image list: {missing_path}")

    if missing:
        print(
            "\n[next steps]\n"
            "  The triplet dataset references {n_missing} image files that are not\n"
            "  on disk yet. They live inside one of the v1.0-trainval*_blobs\n"
            "  tarballs on the nuScenes download page. You have two options:\n\n"
            "    (A) Download the full trainval blobs (~220 GB total) and copy\n"
            "        the missing files into '{samples_dir}'.\n"
            "    (B) Use a mirror that supports per-file fetches (e.g. the\n"
            "        nuScenes HuggingFace mirror) and pull only the paths\n"
            "        listed in '{missing_path}'.\n\n"
            "  After fetching, re-run this script to verify everything is in\n"
            "  place. The triplet JSON itself does not need to be regenerated.".format(
                n_missing=len(missing),
                samples_dir=os.path.join(args.nuscenes_root, "samples"),
                missing_path=missing_path,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
