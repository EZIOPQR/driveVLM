"""Evaluation for the nuScenes object-localization QA dataset.

Companion to ``tools/create_nus_detection_qa.py`` + ``tools/inference_batch.py``.
GT is the HF Dataset (``conversations[1].value``); predictions come from
inference_batch's JSON (``[{id, question, answer}, ...]``).

Reports:
- overall: total samples, predicted-empty rate, average box L1 error
- per-category F1 (greedy nearest-neighbour match within --threshold pixels)
- per-camera F1
- yes/no (existence) question accuracy
- "None." negative-answer precision/recall
- pred-vs-gt box-count distribution

Usage:
    python tools/evaluation_nus_detection.py \\
        --src data/DriveLM_nuScenes/refs/infer_loc_epoch3_detect.json \\
        --gt  /root/autodl-tmp/nus_detection_qa/split_local/val \\
        --threshold 16 \\
        [--per-question]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from datasets import load_from_disk


# Reuse the same regexes as visualize_eval / collate_fn — coord values are floats
# in 448-pixel space (post-loc-token decode).
_TAG_RE = re.compile(r"<c\d+,(CAM_[A-Z_]+),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)>")

CAM_NAMES = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

_PLURAL_TO_SINGULAR = {
    "cars": "car", "trucks": "truck", "buses": "bus",
    "construction vehicles": "construction_vehicle",
    "trailers": "trailer", "motorcycles": "motorcycle", "bicycles": "bicycle",
    "emergency vehicles": "emergency_vehicle",
    "pedestrians": "pedestrian", "animals": "animal",
    "barriers": "barrier", "traffic cones": "traffic_cone",
    "pieces of debris": "debris",
    "pushable/pullable objects": "pushable_pullable",
    "bicycle racks": "bicycle_rack",
}
_SINGULARS = set(_PLURAL_TO_SINGULAR.values())


def parse_tags(text: str) -> List[Tuple[str, float, float]]:
    """Extract (cam, x, y) tuples from a string of <cN,CAM_X,x,y> tags."""
    return [(m.group(1), float(m.group(2)), float(m.group(3)))
            for m in _TAG_RE.finditer(text)]


def extract_categories(question: str) -> List[str]:
    """Heuristic: scan the question for any plural/singular category words."""
    q = question.lower()
    found = []
    for plural, singular in _PLURAL_TO_SINGULAR.items():
        if plural in q:
            found.append(singular)
    if not found:
        for cat in _SINGULARS:
            if re.search(rf"\b{cat}\b", q):
                found.append(cat)
    # Preserve insertion order, dedupe.
    seen = set()
    out = []
    for c in found:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def is_existence_question(q: str) -> bool:
    """True for templates starting with Are/Is/Can/Do."""
    return bool(re.match(r"^(Are|Is|Can|Do)\b", q.strip(), re.IGNORECASE))


def is_negative_answer(text: str) -> bool:
    """Detect 'No.' / 'None.' / 'None visible.' style answers."""
    t = text.strip().lower()
    return (
        t.startswith("no")
        or t.startswith("none")
        or "do not see" in t
        or "cannot see" in t
        or "not visible" in t
    )


def match_points(
    pred_pts: List[Tuple[str, float, float]],
    gt_pts: List[Tuple[str, float, float]],
    threshold: float,
):
    """Greedy nearest-neighbour match in L1 distance, restricted to same camera.

    Returns dict with tp, fp, fn, matched_dists (list of L1 distances of TPs).
    """
    remaining = list(range(len(gt_pts)))
    matched_dists = []
    fp = 0
    for p_cam, px, py in pred_pts:
        best_gi = None
        best_d = float("inf")
        for gi in remaining:
            g_cam, gx, gy = gt_pts[gi]
            if g_cam != p_cam:
                continue
            d = abs(px - gx) + abs(py - gy)
            if d < best_d:
                best_gi, best_d = gi, d
        if best_gi is not None and best_d < threshold:
            matched_dists.append(best_d)
            remaining.remove(best_gi)
        else:
            fp += 1
    tp = len(matched_dists)
    fn = len(remaining)
    return {"tp": tp, "fp": fp, "fn": fn, "matched_dists": matched_dists}


def f1_from(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate(args):
    print(f"[eval] loading GT {args.gt}")
    ds = load_from_disk(args.gt)
    gt_by_id = {r["id"]: r for r in ds}

    print(f"[eval] loading pred {args.src}")
    with open(args.src) as f:
        preds = json.load(f)
    print(f"[eval] gt={len(gt_by_id)}  pred={len(preds)}")

    # Aggregate counters
    overall = {"tp": 0, "fp": 0, "fn": 0, "matched_dists": []}
    per_cat: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "matched_dists": []}
    )
    per_cam: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "matched_dists": []}
    )

    # Existence-question accuracy & negative-answer precision/recall
    exist_total = 0
    exist_correct = 0
    neg_tp = neg_fp = neg_fn = 0      # treat "No./None." as positive class

    n_pred_empty = 0
    n_gt_empty = 0
    n_total = 0
    n_missing_gt = 0
    pred_box_counts = []
    gt_box_counts = []

    per_question_records = []

    for r in preds:
        sid = r["id"]
        pred_text = r.get("answer", "")
        gt_row = gt_by_id.get(sid)
        if gt_row is None:
            n_missing_gt += 1
            continue

        question = gt_row["conversations"][0]["value"]
        gt_text = gt_row["conversations"][1]["value"]

        gt_pts = parse_tags(gt_text)
        pred_pts = parse_tags(pred_text)

        gt_box_counts.append(len(gt_pts))
        pred_box_counts.append(len(pred_pts))
        if not gt_pts:
            n_gt_empty += 1
        if not pred_pts:
            n_pred_empty += 1

        # Existence-question accuracy: was Yes/No correctly produced?
        if is_existence_question(question):
            exist_total += 1
            gt_pos = bool(gt_pts) or not is_negative_answer(gt_text)
            pred_pos = bool(pred_pts) or not is_negative_answer(pred_text)
            if gt_pos == pred_pos:
                exist_correct += 1

        # Negative-answer P/R: classify whether each side claims "no objects".
        gt_neg = is_negative_answer(gt_text) and not gt_pts
        pred_neg = is_negative_answer(pred_text) and not pred_pts
        if pred_neg and gt_neg:
            neg_tp += 1
        elif pred_neg and not gt_neg:
            neg_fp += 1
        elif (not pred_neg) and gt_neg:
            neg_fn += 1

        match = match_points(pred_pts, gt_pts, args.threshold)
        for k in ("tp", "fp", "fn"):
            overall[k] += match[k]
        overall["matched_dists"].extend(match["matched_dists"])

        # Per-camera split (re-match within each camera for accurate per-cam F1)
        for cam in CAM_NAMES:
            pp = [t for t in pred_pts if t[0] == cam]
            gg = [t for t in gt_pts if t[0] == cam]
            if not pp and not gg:
                continue
            mm = match_points(pp, gg, args.threshold)
            for k in ("tp", "fp", "fn"):
                per_cam[cam][k] += mm[k]
            per_cam[cam]["matched_dists"].extend(mm["matched_dists"])

        # Per-category split: only meaningful for single-cat positive questions.
        cats = extract_categories(question)
        if len(cats) == 1 and not is_existence_question(question):
            cat = cats[0]
            for k in ("tp", "fp", "fn"):
                per_cat[cat][k] += match[k]
            per_cat[cat]["matched_dists"].extend(match["matched_dists"])

        n_total += 1
        if args.per_question:
            p, r2, f1 = f1_from(match["tp"], match["fp"], match["fn"])
            per_question_records.append({
                "id": sid, "question": question,
                "pred": pred_text, "gt": gt_text,
                "tp": match["tp"], "fp": match["fp"], "fn": match["fn"],
                "f1": round(f1, 4),
            })

    # ----------------- summary -----------------
    print()
    print(f"=== Evaluation summary (threshold = {args.threshold} px L1) ===")
    print(f"matched samples: {n_total}  (gt missing for {n_missing_gt})")
    print(f"pred empty:      {n_pred_empty}/{n_total}")
    print(f"gt empty:        {n_gt_empty}/{n_total}")
    if pred_box_counts:
        print(f"avg #boxes pred: {statistics.mean(pred_box_counts):.2f}, "
              f"gt: {statistics.mean(gt_box_counts):.2f}")
    p, r2, f1 = f1_from(overall["tp"], overall["fp"], overall["fn"])
    md = overall["matched_dists"]
    avg_l1 = statistics.mean(md) if md else float("nan")
    print(f"overall: P={p:.3f}  R={r2:.3f}  F1={f1:.3f}  "
          f"avg_L1(TP)={avg_l1:.2f}px  (TP={overall['tp']} FP={overall['fp']} FN={overall['fn']})")

    if exist_total:
        print(f"existence Q acc: {exist_correct}/{exist_total} "
              f"= {exist_correct/exist_total:.3f}")
    if neg_tp + neg_fp + neg_fn:
        np_, nr_, nf_ = f1_from(neg_tp, neg_fp, neg_fn)
        print(f"negative-answer P/R: P={np_:.3f}  R={nr_:.3f}  F1={nf_:.3f} "
              f"(tp={neg_tp} fp={neg_fp} fn={neg_fn})")

    print("\n--- per category (single-category positive questions only) ---")
    print(f"{'category':<22}  {'P':>6} {'R':>6} {'F1':>6}  "
          f"{'TP':>6} {'FP':>6} {'FN':>6}  {'avg_L1':>7}")
    for cat in sorted(per_cat.keys()):
        c = per_cat[cat]
        p, r2, f1 = f1_from(c["tp"], c["fp"], c["fn"])
        md = c["matched_dists"]
        avg = statistics.mean(md) if md else float("nan")
        print(f"{cat:<22}  {p:>6.3f} {r2:>6.3f} {f1:>6.3f}  "
              f"{c['tp']:>6} {c['fp']:>6} {c['fn']:>6}  {avg:>7.2f}")

    print("\n--- per camera ---")
    print(f"{'camera':<22}  {'P':>6} {'R':>6} {'F1':>6}  "
          f"{'TP':>6} {'FP':>6} {'FN':>6}  {'avg_L1':>7}")
    for cam in CAM_NAMES:
        c = per_cam.get(cam)
        if c is None or (c["tp"] + c["fp"] + c["fn"]) == 0:
            continue
        p, r2, f1 = f1_from(c["tp"], c["fp"], c["fn"])
        md = c["matched_dists"]
        avg = statistics.mean(md) if md else float("nan")
        print(f"{cam:<22}  {p:>6.3f} {r2:>6.3f} {f1:>6.3f}  "
              f"{c['tp']:>6} {c['fp']:>6} {c['fn']:>6}  {avg:>7.2f}")

    if args.per_question:
        out = args.src.rsplit(".", 1)[0] + ".per_question.json"
        with open(out, "w") as f:
            json.dump(per_question_records, f, indent=2, ensure_ascii=False)
        print(f"\n[eval] per-question records -> {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="inference JSON: [{id, question, answer}, ...]")
    ap.add_argument("--gt", required=True,
                    help="HF Dataset directory (e.g. .../nus_detection_qa/split_local/val)")
    ap.add_argument("--threshold", type=float, default=16.0,
                    help="L1 pixel threshold for matching (in 448-pixel space)")
    ap.add_argument("--per-question", action="store_true",
                    help="dump per-question scores to <src>.per_question.json")
    return ap.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
