#!/usr/bin/env python3
"""Visualize keyframes, all sweep images used for flow, and precomputed flow (.npz).

Flow panel: neutral gray background + subsampled quiver arrows (OpenCV-style: +x right, +y down),
not a color wheel.

Matches ``compute_flow_from_sweeps.py``: for each DriveLM keyframe, the causal chain
in (t0-1s, t0] (same camera ``prev`` chain) is the set of images averaged pairwise
into the stored u/v field.

Usage:
  python tools/create_data/visualize_flow_from_sweeps.py \\
    data/DriveLM_nuScenes/QA_dataset_nus/v1_1_train_nus.json \\
    --nuscenes-root /path/to/nuscenes \\
    --flow-root data/DriveLM_nuScenes/flow \\
    --num-frames 4 \\
    --save /tmp/flow_preview.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compute_flow_from_sweeps import (  # noqa: E402
    CAMS,
    build_indexes,
    causal_chain,
    collect_unique_tasks,
    load_tables,
    read_rgb_jpeg,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "drivelm_json",
        type=str,
        help="DriveLM JSON with scene -> key_frames -> image_paths.",
    )
    p.add_argument("--nuscenes-root", type=str, required=True)
    p.add_argument("--flow-root", type=str, required=True)
    p.add_argument("--num-frames", type=int, default=4, help="Max keyframes (tasks) to plot.")
    p.add_argument(
        "--cam",
        type=str,
        default=None,
        choices=CAMS,
        help="Only use this camera (default: any).",
    )
    p.add_argument(
        "--chain-thumb-h",
        type=int,
        default=200,
        help="Max height in pixels for sweep thumbnails in the top strip.",
    )
    p.add_argument(
        "--arrow-step",
        type=int,
        default=0,
        help="Subsample: one arrow every N pixels (0 = auto ~grid/28).",
    )
    p.add_argument(
        "--arrow-scale",
        type=float,
        default=0.0,
        help="Matplotlib quiver scale; larger => shorter arrows. "
        "0 = auto (recommended). If manual, try 1–5 for ~pixel-scale motion, not 10+.",
    )
    p.add_argument(
        "--dpi",
        type=float,
        default=300.0,
        help="PNG/PDF resolution when using --save (default 300). Use 400–600 for very large posters.",
    )
    p.add_argument(
        "--figscale",
        type=float,
        default=1.35,
        help="Multiply default figure size in inches (width,height); larger => more pixels at same DPI.",
    )
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, save figure to this path instead of showing.",
    )
    return p.parse_args()


def resize_max_h(rgb: np.ndarray, max_h: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h <= max_h:
        return rgb
    nh = max_h
    nw = max(1, int(round(w * (max_h / float(h)))))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def draw_flow_quiver(
    ax: Axes,
    u: np.ndarray,
    v: np.ndarray,
    *,
    step: int,
    arrow_scale: float,
    bg: float = 0.92,
) -> None:
    """Draw subsampled flow as small arrows on a neutral background (pixel coords, +y down).

    Uses ``minlength=0`` so matplotlib does not replace short vectors with microscopic dots
    (the default ``minlength=1`` together with a large ``scale`` made arrows vanish on export).
    """
    h, w = u.shape[:2]
    bg_img = np.full((h, w, 3), bg, dtype=np.float32)
    ax.imshow(bg_img)
    if step <= 0:
        step = max(1, min(h, w) // 28)
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    if ys.size == 0 or xs.size == 0:
        return
    X, Y = np.meshgrid(xs, ys)
    Uq = u[np.ix_(ys, xs)].astype(np.float64)
    Vq = v[np.ix_(ys, xs)].astype(np.float64)
    # scale=None lets matplotlib pick a sensible scale; large fixed scale shrinks arrows to invisibility.
    scale_kw = {} if arrow_scale <= 0 else {"scale": arrow_scale}
    ax.quiver(
        X,
        Y,
        Uq,
        Vq,
        color="0.06",
        angles="xy",
        scale_units="xy",
        width=0.003,
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.0,
        minlength=0,
        pivot="middle",
        **scale_kw,
    )
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


def chain_titles(rows: List[dict], t0: int) -> List[str]:
    """Chronological rows (oldest .. keyframe); titles show Δt from keyframe in ms."""
    out: List[str] = []
    for j, r in enumerate(rows):
        dt_us = int(r["timestamp"]) - t0
        dt_ms = dt_us / 1000.0
        fn = Path(str(r.get("filename", ""))).name
        out.append(f"#{j}\nΔt={dt_ms:.1f} ms\n{fn}")
    return out


def gather_tasks(
    drivelm: Dict[str, Any],
    nusc_root: Path,
    flow_root: Path,
    sd_by_token: Dict[str, dict],
    kf_index: Dict[Tuple[str, str], dict],
    cam_filter: str | None,
    limit: int,
    thumb_h: int,
) -> List[
    Tuple[
        str,
        str,
        List[dict],
        List[np.ndarray],
        List[str],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        bool,
    ]
]:
    """Build list of plottable entries for each keyframe with existing flow."""
    tasks = collect_unique_tasks(drivelm)
    out: List[
        Tuple[
            str,
            str,
            List[dict],
            List[np.ndarray],
            List[str],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            bool,
        ]
    ] = []
    for sample_tok, cam, base, sub in tasks:
        if cam_filter is not None and cam != cam_filter:
            continue
        fp = flow_root / cam / f"{base}.npz"
        if not fp.is_file():
            continue
        try:
            z = np.load(fp)
            u = np.asarray(z["u"], dtype=np.float32)
            v = np.asarray(z["v"], dtype=np.float32)
            valid = bool(z["valid"]) if "valid" in z.files else True
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {fp}: {exc}", file=sys.stderr)
            continue

        kf_row = kf_index.get((sample_tok, cam))
        if kf_row is None:
            chain_rows: List[dict] = []
        else:
            t0 = int(kf_row["timestamp"])
            chain_rows = causal_chain(kf_row, sd_by_token, t0)

        fh, fw = u.shape[:2]
        chain_thumbs: List[np.ndarray] = []
        for r in chain_rows:
            rgb = read_rgb_jpeg(nusc_root, r["filename"])
            chain_thumbs.append(resize_max_h(rgb, thumb_h))

        rgb_kf = read_rgb_jpeg(nusc_root, sub)
        rgb_kf_r = cv2.resize(rgb_kf, (fw, fh), interpolation=cv2.INTER_AREA)

        if kf_row is None:
            titles = []
        else:
            t0 = int(kf_row["timestamp"])
            titles = chain_titles(chain_rows, t0)

        out.append((cam, base, chain_rows, chain_thumbs, titles, rgb_kf_r, u, v, valid))
        if len(out) >= limit:
            break

    return out


def main() -> None:
    args = parse_args()
    nusc_root = Path(args.nuscenes_root)
    flow_root = Path(args.flow_root)

    with open(args.drivelm_json, "r", encoding="utf-8") as f:
        drivelm = json.load(f)

    _, sample_data_tbl = load_tables(str(nusc_root))
    sd_by_token, kf_index = build_indexes(sample_data_tbl)

    rows = gather_tasks(
        drivelm,
        nusc_root,
        flow_root,
        sd_by_token,
        kf_index,
        args.cam,
        args.num_frames,
        args.chain_thumb_h,
    )
    if not rows:
        print("No frames found (check paths and that .npz exist).", file=sys.stderr)
        sys.exit(1)

    n_tasks = len(rows)
    fw, fh = 14.0 * args.figscale, (4.2 * n_tasks) * args.figscale
    fig = plt.figure(figsize=(fw, fh))
    outer = GridSpec(n_tasks, 1, figure=fig, hspace=0.35)

    for i, (cam, base, chain_rows, chain_thumbs, titles, rgb_kf_r, u, v, valid) in enumerate(rows):
        inner = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer[i],
            height_ratios=[1.0, 1.15],
            hspace=0.2,
        )
        n_ch = max(1, len(chain_thumbs))
        top = GridSpecFromSubplotSpec(1, n_ch, subplot_spec=inner[0], wspace=0.15)

        if not chain_thumbs:
            ax = fig.add_subplot(top[0, 0])
            ax.text(
                0.5,
                0.5,
                "No causal chain (missing keyframe index\nor empty window).",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            for j in range(len(chain_thumbs)):
                ax = fig.add_subplot(top[0, j])
                ax.imshow(chain_thumbs[j])
                ax.set_title(titles[j] if j < len(titles) else f"#{j}", fontsize=7)
                ax.axis("off")

        bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=inner[1], wspace=0.08)
        ax_kf = fig.add_subplot(bot[0, 0])
        ax_kf.imshow(rgb_kf_r)
        n_pairs = max(0, len(chain_rows) - 1)
        win_s = 1.0
        ax_kf.set_title(
            f"{cam} / {base}.jpg  valid={valid}  "
            f"(chain={len(chain_rows)} imgs, {n_pairs} pairs, window=({-win_s:.0f}s, 0]s vs t0)"
        )
        ax_kf.axis("off")

        ax_fl = fig.add_subplot(bot[0, 1])
        draw_flow_quiver(ax_fl, u, v, step=args.arrow_step, arrow_scale=args.arrow_scale)
        ax_fl.set_title("weighted mean optical flow (arrows, +x right, +y down)")
        ax_fl.axis("off")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
