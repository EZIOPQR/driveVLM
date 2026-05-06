#!/usr/bin/env python3
"""Precompute causal-window optical flow (nuScenes sweeps) for DriveLM keyframes.

For each keyframe image in a DriveLM QA JSON, walks the nuScenes sample_data
prev-chain for the same camera, collects rows with timestamp in (t0-1s, t0],
computes pairwise dense flow (Farneback), accumulates u/v with weights = Δt,
saves flow/CAM/*.npz next to a configurable flow root (mirror samples/ layout).

After Farneback + Δt-weighted average, **u** and **v** are **independently** clipped to
±3σ using fixed global σ (see below); clip limits scale with ``--out-size`` vs the
**448** reference grid the σ were estimated on.

Requires:
  - v1.0-trainval metadata under NUSC_ROOT/v1.0-trainval/
  - sweeps/ and samples/ under NUSC_ROOT (camera blobs unpacked)

Usage:
  python tools/create_data/compute_flow_from_sweeps.py \\
    data/DriveLM_nuScenes/QA_dataset_nus/v1_1_train_nus.json \\
    --nuscenes-root /path/to/nuscenes \\
    --flow-root data/DriveLM_nuScenes/flow \\
    --out-size 14 \\
    --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

CAMS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

US_PER_S = 1_000_000

# Dataset-wide σ from ``stats_flow_range.py`` (full nuScenes flow dump). Saved u/v are clipped
# to ± FLOW_CLIP_SIGMA * σ per channel before float16 (values beyond are cut, not zeroed).
FLOW_STATS_U_STD = 8.77799059738783
FLOW_STATS_V_STD = 2.887801423657257
FLOW_CLIP_SIGMA = 3.0
# σ above were computed on flow saved at H=W=448; scale clip bounds when ``--out-size`` differs.
FLOW_STATS_REF_OUT = 448
_PAR_SD_BY_TOKEN: Optional[Dict[str, dict]] = None
_PAR_KF_INDEX: Optional[Dict[Tuple[str, str], dict]] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "drivelm_json",
        type=str,
        help="DriveLM v1_1_train_nus.json (or any JSON with scene -> key_frames -> image_paths).",
    )
    p.add_argument(
        "--nuscenes-root",
        type=str,
        required=True,
        help="nuScenes root with v1.0-trainval/, samples/, sweeps/.",
    )
    p.add_argument(
        "--flow-root",
        type=str,
        required=True,
        help="Output root; writes flow/CAM/*.npz mirroring samples/CAM/*.jpg basenames.",
    )
    p.add_argument(
        "--out-size",
        type=int,
        default=14,
        help="Saved flow H=W (default 14). Farneback field is resized to this grid.",
    )
    p.add_argument(
        "--flow-compute-size",
        type=int,
        default=224,
        help="Farneback on this square size; flow upscaled to out-size.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if target .npz already exists.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Process pool size (1 = no parallelism). Use about physical CPU cores.",
    )
    return p.parse_args()


def load_tables(nusc_root: str, version: str = "v1.0-trainval") -> Tuple[List[dict], List[dict]]:
    meta = Path(nusc_root) / version
    sp = meta / "sample.json"
    sd = meta / "sample_data.json"
    if not sp.is_file() or not sd.is_file():
        print(f"[ERROR] Missing metadata under {meta}", file=sys.stderr)
        sys.exit(2)
    print("[info] Loading sample.json ...")
    with open(sp, "r", encoding="utf-8") as f:
        samples_tbl = json.load(f)
    print("[info] Loading sample_data.json (large) ...")
    with open(sd, "r", encoding="utf-8") as f:
        sample_data_tbl = json.load(f)
    return samples_tbl, sample_data_tbl


def build_indexes(sample_data_tbl: List[dict]) -> Tuple[Dict[str, dict], Dict[Tuple[str, str], dict]]:
    sd_by_token: Dict[str, dict] = {r["token"]: r for r in sample_data_tbl}
    kf_index: Dict[Tuple[str, str], dict] = {}
    for r in sample_data_tbl:
        if not r.get("is_key_frame", False):
            continue
        fn = r.get("filename", "")
        if not fn.startswith("samples/"):
            continue
        parts = fn.split("/")
        if len(parts) < 3:
            continue
        ch = parts[1]
        if ch not in CAMS:
            continue
        kf_index[(r["sample_token"], ch)] = r
    return sd_by_token, kf_index


def collect_unique_tasks(drivelm: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    """Return list of (sample_token, cam, jpg_basename, samples_rel_path)."""
    seen: set = set()
    out: List[Tuple[str, str, str, str]] = []
    for scene_tok, scene in drivelm.items():
        for sample_tok, kf in scene.get("key_frames", {}).items():
            paths = kf.get("image_paths", {})
            for cam in CAMS:
                rel = paths.get(cam)
                if not rel:
                    continue
                rel_norm = rel.replace("\\", "/")
                parts = rel_norm.split("/")
                if "samples" not in parts:
                    continue
                i = parts.index("samples")
                sub = "/".join(parts[i:])
                base = Path(sub).stem
                key = (sample_tok, cam, base)
                if key in seen:
                    continue
                seen.add(key)
                out.append((sample_tok, cam, base, sub))
    return out


def causal_chain(
    kf_row: dict,
    sd_by_token: Dict[str, dict],
    t0: int,
) -> List[dict]:
    """Chronological sample_data rows in (t0-1s, t0], same sensor stream."""
    chain_newest_first: List[dict] = []
    cur: Optional[dict] = kf_row
    while cur is not None:
        t = int(cur["timestamp"])
        if t <= t0 - US_PER_S:
            break
        if t <= t0:
            chain_newest_first.append(cur)
        prev = cur.get("prev") or ""
        if not prev:
            break
        cur = sd_by_token.get(prev)
        if cur is None:
            break
    chain_newest_first.reverse()
    return chain_newest_first


def read_rgb_jpeg(nusc_root: Path, filename: str) -> np.ndarray:
    path = nusc_root / filename.replace("\\", "/")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2.imread failed: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_pair_flow(
    rgb0: np.ndarray,
    rgb1: np.ndarray,
    compute_size: int,
    out_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    g0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY)
    if g0.shape[0] != compute_size:
        g0 = cv2.resize(g0, (compute_size, compute_size), interpolation=cv2.INTER_AREA)
        g1 = cv2.resize(g1, (compute_size, compute_size), interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(
        g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    scale = out_size / float(compute_size)
    u = cv2.resize(flow[:, :, 0], (out_size, out_size), interpolation=cv2.INTER_LINEAR) * scale
    v = cv2.resize(flow[:, :, 1], (out_size, out_size), interpolation=cv2.INTER_LINEAR) * scale
    return u.astype(np.float32), v.astype(np.float32)


def clip_bounds_for_out_size(out_size: int) -> Tuple[float, float]:
    """± bounds for u and v from global σ, scaled to this grid vs FLOW_STATS_REF_OUT."""
    s = float(out_size) / float(FLOW_STATS_REF_OUT)
    bu = float(FLOW_CLIP_SIGMA * FLOW_STATS_U_STD * s)
    bv = float(FLOW_CLIP_SIGMA * FLOW_STATS_V_STD * s)
    return bu, bv


def clip_flow_uv_3sigma(
    u: np.ndarray, v: np.ndarray, out_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    bu, bv = clip_bounds_for_out_size(out_size)
    return np.clip(u, -bu, bu), np.clip(v, -bv, bv)


def weighted_flow_average(
    chain: List[dict],
    nusc_root: Path,
    compute_size: int,
    out_size: int,
) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    if len(chain) < 2:
        return (
            np.zeros((out_size, out_size), dtype=np.float32),
            np.zeros((out_size, out_size), dtype=np.float32),
            False,
            0.0,
        )
    u_acc = np.zeros((out_size, out_size), dtype=np.float64)
    v_acc = np.zeros((out_size, out_size), dtype=np.float64)
    w_sum = 0.0
    for i in range(len(chain) - 1):
        r0, r1 = chain[i], chain[i + 1]
        dt = (int(r1["timestamp"]) - int(r0["timestamp"])) / US_PER_S
        if dt <= 0:
            continue
        rgb0 = read_rgb_jpeg(nusc_root, r0["filename"])
        rgb1 = read_rgb_jpeg(nusc_root, r1["filename"])
        u, v = compute_pair_flow(rgb0, rgb1, compute_size, out_size)
        w = float(dt)
        u_acc += w * u
        v_acc += w * v
        w_sum += w
    if w_sum <= 0:
        return (
            np.zeros((out_size, out_size), dtype=np.float32),
            np.zeros((out_size, out_size), dtype=np.float32),
            False,
            0.0,
        )
    u_out = (u_acc / w_sum).astype(np.float32)
    v_out = (v_acc / w_sum).astype(np.float32)
    return u_out, v_out, True, w_sum


def _init_parallel_tables(sd_by_token: Dict[str, dict], kf_index: Dict[Tuple[str, str], dict]) -> None:
    global _PAR_SD_BY_TOKEN, _PAR_KF_INDEX
    _PAR_SD_BY_TOKEN = sd_by_token
    _PAR_KF_INDEX = kf_index


def _worker_limit_threads() -> None:
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _process_one_task(
    payload: Tuple[str, str, str, str, str, str, int, int],
) -> str:
    """One keyframe job. Returns 'ok' | 'fail' | 'exc:…'."""
    global _PAR_SD_BY_TOKEN, _PAR_KF_INDEX
    _worker_limit_threads()
    sample_tok, cam, base, _sub, nusc_s, flow_root_s, out_size, flow_compute_size = payload
    sd_by_token = _PAR_SD_BY_TOKEN
    kf_index = _PAR_KF_INDEX
    if sd_by_token is None or kf_index is None:
        return "exc:parallel tables not initialized"

    nusc = Path(nusc_s)
    flow_root = Path(flow_root_s)
    out_path = flow_root / cam / f"{base}.npz"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        kf_row = kf_index.get((sample_tok, cam))
        if kf_row is None:
            np.savez_compressed(
                str(out_path),
                u=np.zeros((out_size, out_size), np.float16),
                v=np.zeros((out_size, out_size), np.float16),
                valid=False,
                reason=b"missing_keyframe_index",
            )
            return "fail"
        t0 = int(kf_row["timestamp"])
        chain = causal_chain(kf_row, sd_by_token, t0)
        u, v, valid, w_sum = weighted_flow_average(chain, nusc, flow_compute_size, out_size)
        u, v = clip_flow_uv_3sigma(u, v, out_size)
        bu, bv = clip_bounds_for_out_size(out_size)
        np.savez_compressed(
            str(out_path),
            u=u.astype(np.float16),
            v=v.astype(np.float16),
            valid=valid,
            w_sum=np.float32(w_sum),
            n_pairs=len(chain) - 1,
            clip_bound_u=np.float32(bu),
            clip_bound_v=np.float32(bv),
        )
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"exc:{type(exc).__name__}: {exc}"


def _collect_pending_tasks(
    tasks: List[Tuple[str, str, str, str]],
    flow_root: Path,
    skip_existing: bool,
) -> Tuple[List[Tuple[str, str, str, str]], int]:
    pending: List[Tuple[str, str, str, str]] = []
    n_skip = 0
    for sample_tok, cam, base, sub in tasks:
        out_path = flow_root / cam / f"{base}.npz"
        if skip_existing and out_path.is_file():
            n_skip += 1
            continue
        pending.append((sample_tok, cam, base, sub))
    return pending, n_skip


def main() -> None:
    args = parse_args()
    nusc = Path(args.nuscenes_root)
    flow_root = Path(args.flow_root)
    flow_root.mkdir(parents=True, exist_ok=True)

    _, sample_data_tbl = load_tables(str(nusc))
    sd_by_token, kf_index = build_indexes(sample_data_tbl)
    print(f"[info] keyframes indexed: {len(kf_index)}, sample_data rows: {len(sample_data_tbl)}")

    with open(args.drivelm_json, "r", encoding="utf-8") as f:
        drivelm = json.load(f)
    tasks = collect_unique_tasks(drivelm)
    print(f"[info] unique (sample,camera) keyframes: {len(tasks)}")

    pending, n_skip = _collect_pending_tasks(tasks, flow_root, args.skip_existing)
    workers = max(1, args.workers)
    _init_parallel_tables(sd_by_token, kf_index)

    n_ok = n_fail = n_exc = 0

    def _run_payloads(payloads: List[Tuple[str, str, str, str, str, str, int, int]]) -> None:
        nonlocal n_ok, n_fail, n_exc
        for payload in tqdm(payloads, desc="flow"):
            res = _process_one_task(payload)
            if res == "ok":
                n_ok += 1
            elif res == "fail":
                n_fail += 1
            else:
                n_exc += 1
                print(f"[error] {res}", file=sys.stderr)

    payloads = [
        (
            sample_tok,
            cam,
            base,
            sub,
            str(nusc),
            str(flow_root),
            args.out_size,
            args.flow_compute_size,
        )
        for sample_tok, cam, base, sub in pending
    ]

    if workers <= 1 or not payloads:
        _run_payloads(payloads)
    else:
        windows = sys.platform.startswith("win")
        mp_ctx = multiprocessing.get_context("spawn" if windows else "fork")
        use_initializer = windows
        print(f"[info] parallel workers={workers} start_method={mp_ctx.get_start_method()}")
        ex_kw: Dict[str, Any] = {"max_workers": workers, "mp_context": mp_ctx}
        if use_initializer:
            ex_kw["initializer"] = _init_parallel_tables
            ex_kw["initargs"] = (sd_by_token, kf_index)
        with ProcessPoolExecutor(**ex_kw) as ex:
            futures = [ex.submit(_process_one_task, p) for p in payloads]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="flow"):
                res = fut.result()
                if res == "ok":
                    n_ok += 1
                elif res == "fail":
                    n_fail += 1
                else:
                    n_exc += 1
                    print(f"[error] {res}", file=sys.stderr)

    print(
        f"[done] wrote={n_ok}, skip_existing={n_skip}, missing_index={n_fail}, "
        f"errors={n_exc}, flow_root={flow_root}"
    )


if __name__ == "__main__":
    main()
