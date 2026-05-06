#!/usr/bin/env python3
"""Aggregate min/max/mean/std and sampled percentiles over all flow .npz (u, v).

Mirrors layout from ``compute_flow_from_sweeps.py``: FLOW_ROOT/CAM/*.npz .

Usage:
  python tools/create_data/stats_flow_range.py /path/to/flow \\
    --only-valid \\
    --hist-png /path/to/flow_uv_hist.png \\
    --percentile-reservoir 2000000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
    matplotlib = None

CAMS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("flow_root", type=str, help="Root with CAM/*.npz")
    p.add_argument(
        "--only-valid",
        action="store_true",
        help="Only include files where npz['valid'] is True (if missing, include).",
    )
    p.add_argument(
        "--percentile-reservoir",
        type=int,
        default=0,
        help="If >0, keep this many uniform subsamples for approximate percentiles (slower, more memory).",
    )
    p.add_argument(
        "--per-file-sample-cap",
        type=int,
        default=80_000,
        help="Max pixels sampled per file when building percentile reservoir.",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Write summary JSON to this path.",
    )
    p.add_argument(
        "--hist-png",
        type=str,
        default=None,
        help="If set, second pass over npz files and save u/v histograms (linear bins) to this PNG.",
    )
    p.add_argument(
        "--hist-bins",
        type=int,
        default=120,
        help="Number of bins for each of u and v (default 120).",
    )
    return p.parse_args()


def list_npz_files(flow_root: Path) -> List[Path]:
    out: List[Path] = []
    for cam in CAMS:
        d = flow_root / cam
        if not d.is_dir():
            continue
        out.extend(sorted(d.glob("*.npz")))
    return out


def _merge_reservoir(
    rng: np.random.Generator,
    buf: np.ndarray | None,
    add: np.ndarray,
    cap: int,
) -> np.ndarray | None:
    if add.size == 0:
        return buf
    if buf is None or buf.size == 0:
        cur = add.astype(np.float64, copy=False)
    else:
        cur = np.concatenate([buf, add.astype(np.float64, copy=False)])
    if cur.size <= cap:
        return cur
    idx = rng.choice(cur.size, size=cap, replace=False)
    return cur[idx]


def _edges_lin(lo: float, hi: float, nbins: int) -> np.ndarray:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.linspace(-1.0, 1.0, nbins + 1)
    if hi <= lo:
        pad = max(abs(lo), 1.0) * 1e-6
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, nbins + 1)


def _load_uv_no_stats(
    path: Path,
    only_valid: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        z = np.load(path)
        u = np.asarray(z["u"], dtype=np.float64)
        v = np.asarray(z["v"], dtype=np.float64)
        if only_valid and "valid" in z.files and not bool(z["valid"]):
            return None
    except Exception:  # noqa: BLE001
        return None
    if u.shape != v.shape:
        return None
    return u.ravel(), v.ravel()


def accumulate_uv_histograms(
    paths: List[Path],
    only_valid: bool,
    min_u: float,
    max_u: float,
    min_v: float,
    max_v: float,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges_u = _edges_lin(min_u, max_u, nbins)
    edges_v = _edges_lin(min_v, max_v, nbins)
    hu = np.zeros(nbins, dtype=np.uint64)
    hv = np.zeros(nbins, dtype=np.uint64)
    for path in tqdm(paths, desc="hist"):
        pair = _load_uv_no_stats(path, only_valid)
        if pair is None:
            continue
        uf, vf = pair
        hu += np.histogram(uf, bins=edges_u)[0].astype(np.uint64)
        hv += np.histogram(vf, bins=edges_v)[0].astype(np.uint64)
    return hu, edges_u, hv, edges_v


def save_uv_histogram_png(
    path: Path,
    hu: np.ndarray,
    edges_u: np.ndarray,
    hv: np.ndarray,
    edges_v: np.ndarray,
    n_pixels: int,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for --hist-png (pip install matplotlib)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    centers_u = 0.5 * (edges_u[:-1] + edges_u[1:])
    centers_v = 0.5 * (edges_v[:-1] + edges_v[1:])
    wu = float(edges_u[1] - edges_u[0])
    wv = float(edges_v[1] - edges_v[0])
    ax0, ax1 = axes
    ax0.bar(centers_u, hu.astype(np.float64), width=wu * 0.98, align="center", color="steelblue", edgecolor="none")
    ax0.set_xlabel("u (pixels)")
    ax0.set_ylabel("count")
    ax0.set_title("u histogram (all pixels)")
    ax1.bar(centers_v, hv.astype(np.float64), width=wv * 0.98, align="center", color="darkorange", edgecolor="none")
    ax1.set_xlabel("v (pixels)")
    ax1.set_ylabel("count")
    ax1.set_title("v histogram (all pixels)")
    fig.suptitle(f"flow u/v over n≈{n_pixels:.3g} pixels", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    flow_root = Path(args.flow_root)
    if not flow_root.is_dir():
        print(f"[error] not a directory: {flow_root}", file=sys.stderr)
        sys.exit(2)

    paths = list_npz_files(flow_root)
    if not paths:
        print(f"[error] no *.npz under {flow_root}/<CAM>/", file=sys.stderr)
        sys.exit(2)

    rng = np.random.default_rng(0)
    res_cap = max(0, args.percentile_reservoir)
    per_cap = max(1, args.per_file_sample_cap)

    n_files = n_skipped = 0
    n_pix = 0
    min_u = min_v = min_mag = np.inf
    max_u = max_v = max_mag = -np.inf
    sum_u = sum_v = 0.0
    sum_u2 = sum_v2 = 0.0
    sum_mag = 0.0
    sum_mag2 = 0.0
    ru = rv = rm = None

    for path in tqdm(paths, desc="scan"):
        try:
            z = np.load(path)
            u = np.asarray(z["u"], dtype=np.float64)
            v = np.asarray(z["v"], dtype=np.float64)
            if args.only_valid and "valid" in z.files and not bool(z["valid"]):
                n_skipped += 1
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skip {path}: {exc}", file=sys.stderr)
            n_skipped += 1
            continue

        if u.shape != v.shape:
            print(f"[warn] skip {path}: u/v shape mismatch", file=sys.stderr)
            n_skipped += 1
            continue

        n_files += 1
        uf = u.ravel()
        vf = v.ravel()
        pn = uf.size
        n_pix += pn

        tmin_u = float(uf.min())
        tmax_u = float(uf.max())
        tmin_v = float(vf.min())
        tmax_v = float(vf.max())
        if tmin_u < min_u:
            min_u = tmin_u
        if tmax_u > max_u:
            max_u = tmax_u
        if tmin_v < min_v:
            min_v = tmin_v
        if tmax_v > max_v:
            max_v = tmax_v

        su = float(uf.sum())
        sv = float(vf.sum())
        sum_u += su
        sum_v += sv
        sum_u2 += float(np.square(uf).sum())
        sum_v2 += float(np.square(vf).sum())
        mag = np.hypot(uf, vf)
        tmin_m = float(mag.min())
        tmax_m = float(mag.max())
        if tmin_m < min_mag:
            min_mag = tmin_m
        if tmax_m > max_mag:
            max_mag = tmax_m
        sum_mag += float(mag.sum())
        sum_mag2 += float(np.square(mag).sum())

        if res_cap > 0:
            if pn <= per_cap:
                idx = np.arange(pn)
            else:
                idx = rng.choice(pn, size=per_cap, replace=False)
            ru = _merge_reservoir(rng, ru, uf[idx], res_cap)
            rv = _merge_reservoir(rng, rv, vf[idx], res_cap)
            rm = _merge_reservoir(rng, rm, mag[idx], res_cap)

    if n_pix == 0:
        print("[error] no pixels accumulated (all skipped?)", file=sys.stderr)
        sys.exit(1)

    mean_u = sum_u / n_pix
    mean_v = sum_v / n_pix
    std_u = (sum_u2 / n_pix - mean_u**2) ** 0.5
    std_v = (sum_v2 / n_pix - mean_v**2) ** 0.5
    mean_mag = sum_mag / n_pix
    std_mag = (sum_mag2 / n_pix - mean_mag**2) ** 0.5

    pct: Dict[str, Any] = {}
    if res_cap > 0 and ru is not None and ru.size > 0:
        qs = [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]
        pct["u"] = {f"p{q:g}": float(np.percentile(ru, q)) for q in qs}
        pct["v"] = {f"p{q:g}": float(np.percentile(rv, q)) for q in qs}
        pct["mag"] = {f"p{q:g}": float(np.percentile(rm, q)) for q in qs}

    summary = {
        "flow_root": str(flow_root.resolve()),
        "n_files_used": n_files,
        "n_files_skipped": n_skipped,
        "n_pixels": n_pix,
        "u": {"min": min_u, "max": max_u, "mean": mean_u, "std": std_u},
        "v": {"min": min_v, "max": max_v, "mean": mean_v, "std": std_v},
        "magnitude": {"min": min_mag, "max": max_mag, "mean": mean_mag, "std": std_mag},
        "percentiles_note": (
            f"approximate from reservoir n={ru.size}" if ru is not None else None
        ),
        "percentiles": pct or None,
    }

    print(json.dumps(summary, indent=2))

    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[info] wrote {outp}", file=sys.stderr)

    if args.hist_png:
        nb = max(8, args.hist_bins)
        hu, eu, hv, ev = accumulate_uv_histograms(
            paths,
            args.only_valid,
            float(min_u),
            float(max_u),
            float(min_v),
            float(max_v),
            nb,
        )
        outp = Path(args.hist_png)
        save_uv_histogram_png(outp, hu, eu, hv, ev, float(n_pix))
        print(f"[info] histogram PNG: {outp.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
