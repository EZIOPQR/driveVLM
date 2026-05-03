"""Build a static HTML report for one inference json (DriveLM, Phi-4 path).

Per-sample scoring:
- tag=0 (MCQ / yes-no)         -> exact match {0.0, 1.0}
- tag=2 (free-form perception) -> single-sentence ROUGE-L F1
- tag=3 (object coords)        -> match-F1 (same logic as tools/evaluation.py)
- tag=1 (planning)             -> dropped (no auto score in evaluation.py either)

Output (per --src):
    {out}/{src_basename}.html

Image references are written relative to a `DriveLM_nuScenes/` root
(i.e. `nuscenes/samples/CAM_*/*.jpg`). To view: copy the .html into your
local `DriveLM_nuScenes/` folder (the one containing `nuscenes/samples/...`)
and open in a browser. No thumbnails are generated; original full-res images
are referenced directly and lazily loaded. When GT/pred contain
`<c*,CAM_*,x,y>` tags, green (GT) and orange (prediction) markers with 224-space
coordinates are drawn on the matching camera thumbnails after expanding a card.

Usage:
    python tools/visualize_eval.py \\
        --src data/DriveLM_nuScenes/refs/infer_epoch3.json \\
        --out viz_eval/
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

_OBJ_TAG_RE = re.compile(r"<c\d+,CAM_[A-Z_]+,\d+\.?\d*,\d+\.?\d*>")
# Full object-id tags with camera + pixel coords (same 224-space as val_cot.json).
_OBJ_TAG_PARSE_RE = re.compile(
    r"<c\d+,(CAM_[A-Z_]+),(\d+\.?\d*),(\d+\.?\d*)>",
    re.IGNORECASE,
)
# evaluation.py keeps only floats with explicit decimal, mirror it for parity.
_FLOAT_RE = re.compile(r"\d+\.\d+")

# Coordinates in *_cot.json are expressed in this square grid (see create_drivelm_nus.rescale_coords).
OVERLAY_COORD_SPACE = 224


def rouge_l_f1(pred: str, gt: str) -> float:
    """LCS-based single-sentence ROUGE-L F1."""
    p_toks = pred.lower().split()
    g_toks = gt.lower().split()
    if not p_toks or not g_toks:
        return 0.0
    m, n = len(p_toks), len(g_toks)
    # 1-row DP for memory; we only need lcs length.
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        pi = p_toks[i - 1]
        for j in range(1, n + 1):
            if pi == g_toks[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


def obj_tags_by_cam(text: str) -> dict[str, list[list[float]]]:
    """Parse `<c*,CAM_*,x,y>` tokens into {CAM_*: [[x,y], ...]}."""
    out: dict[str, list[list[float]]] = {}
    for m in _OBJ_TAG_PARSE_RE.finditer(text):
        cam = m.group(1).upper()
        x, y = float(m.group(2)), float(m.group(3))
        out.setdefault(cam, []).append([x, y])
    return out


def merge_cam_points(gt_text: str, pred_text: str) -> dict[str, dict[str, list[list[float]]]]:
    """GT object tags from gt_text; pred tags from pred_text — same camera grid."""
    merged: dict[str, dict[str, list[list[float]]]] = {}
    for cam, pts in obj_tags_by_cam(gt_text).items():
        merged.setdefault(cam, {"gt": [], "pred": []})
        merged[cam]["gt"] = pts
    for cam, pts in obj_tags_by_cam(pred_text).items():
        merged.setdefault(cam, {"gt": [], "pred": []})
        merged[cam]["pred"] = pts
    return {k: v for k, v in merged.items() if v["gt"] or v["pred"]}


def coord_pairs(text: str) -> list[tuple[float, float]]:
    nums = _FLOAT_RE.findall(text)
    if len(nums) % 2 != 0:
        nums = nums[:-1]
    return [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums), 2)]


def coord_match(pred: str, gt: str, threshold: float = 16.0):
    """Greedy nearest-neighbour match (L1 distance), mirroring evaluation.match_result.

    Returns:
        f1: float in [0, 1]
        matched: list of {gt_idx, pred_idx, dist}
        missed: list of unmatched gt indices
        extra:  list of unmatched pred indices
        gt_pairs / pred_pairs
    """
    gt_pairs = coord_pairs(gt)
    pred_pairs = coord_pairs(pred)
    remaining = list(range(len(gt_pairs)))
    matched, extra = [], []
    for pi, p in enumerate(pred_pairs):
        best_gi, best_d = None, float("inf")
        for gi in remaining:
            g = gt_pairs[gi]
            d = abs(p[0] - g[0]) + abs(p[1] - g[1])
            if d < best_d:
                best_gi, best_d = gi, d
        if best_gi is not None and best_d < threshold:
            matched.append({"gt_idx": best_gi, "pred_idx": pi, "dist": round(best_d, 2)})
            remaining.remove(best_gi)
        else:
            extra.append(pi)
    missed = remaining
    tp, fp, fn = len(matched), len(extra), len(missed)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, matched, missed, extra, gt_pairs, pred_pairs


def score_one(tag: int, pred: str, gt: str):
    """Returns (score in [0,1], extra dict for tag=3 coord render)."""
    if tag == 0:
        return (1.0 if pred.strip() == gt.strip() else 0.0), None
    if tag == 2:
        return rouge_l_f1(pred, gt), None
    if tag == 3:
        f1, matched, missed, extra, gt_pairs, pred_pairs = coord_match(pred, gt)
        return f1, {
            "matched": matched,
            "missed": missed,
            "extra": extra,
            "gt_pairs": [list(p) for p in gt_pairs],
            "pred_pairs": [list(p) for p in pred_pairs],
        }
    raise ValueError(f"unsupported tag {tag}")


def score_bucket(tag: int, score: float) -> str:
    """correct / partial / wrong (drives the colour badge)."""
    if tag == 0:
        return "correct" if score == 1.0 else "wrong"
    if score >= 0.8:
        return "correct"
    if score >= 0.3:
        return "partial"
    return "wrong"


# ---------------------------------------------------------------------------
# diff helpers (server-side pre-compute, simpler client)
# ---------------------------------------------------------------------------

# crude tokenizer that keeps `<c1,CAM_FRONT,1.2,3.4>` as one token.
_TOKEN_RE = re.compile(r"<c\d+,CAM_[A-Z_]+,\d+\.?\d*,\d+\.?\d*>|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def diff_marks(gt: str, pred: str):
    """Returns (gt_tokens, gt_marks, pred_tokens, pred_marks).

    Mark values: 'ok' | 'miss' (gt only) | 'extra' (pred only).
    Multiset-aware: a duplicate that appears once in the other side is matched once.
    """
    gt_toks = tokenize(gt)
    pred_toks = tokenize(pred)
    gt_low = [t.lower() for t in gt_toks]
    pred_low = [t.lower() for t in pred_toks]

    pred_remaining = Counter(pred_low)
    gt_marks = []
    for t in gt_low:
        if pred_remaining.get(t, 0) > 0:
            pred_remaining[t] -= 1
            gt_marks.append("ok")
        else:
            gt_marks.append("miss")

    gt_remaining = Counter(gt_low)
    pred_marks = []
    for t in pred_low:
        if gt_remaining.get(t, 0) > 0:
            gt_remaining[t] -= 1
            pred_marks.append("ok")
        else:
            pred_marks.append("extra")
    return gt_toks, gt_marks, pred_toks, pred_marks


# ---------------------------------------------------------------------------
# data loading & alignment with val_cot.json
# ---------------------------------------------------------------------------

CAM_NAMES = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

# All image_paths in val_cot.json have this prefix; strip it so the HTML
# img src becomes `nuscenes/samples/CAM_*/...jpg`, relative to the local
# DriveLM_nuScenes/ root the user opens the HTML from.
_IMG_STRIP_PREFIX = "data/DriveLM_nuScenes/"


def to_img_src(image_path: str) -> str:
    if image_path.startswith(_IMG_STRIP_PREFIX):
        return image_path[len(_IMG_STRIP_PREFIX):]
    return image_path


def load_aligned(src_path: str, gt_path: str):
    """Return list of dicts: id, scene_id, frame_id, qa_idx, tag, question, gt, pred, image_paths."""
    with open(src_path) as f:
        pred_list = json.load(f)
    pred_by_id = {r["id"]: r for r in pred_list}

    with open(gt_path) as f:
        gt_frames = json.load(f)

    out = []
    for frame in gt_frames:
        scene_id = frame["scene_id"]
        frame_id = frame["frame_id"]
        image_paths = frame["image_paths"]
        qas = (frame["QA"]["perception"]
               + frame["QA"]["prediction"]
               + frame["QA"]["planning"]
               + frame["QA"]["behavior"])
        for i, qa in enumerate(qas):
            tag_list = qa["tag"]
            # Drop tag=1 (planning, no auto score).
            if not (0 in tag_list or 2 in tag_list or 3 in tag_list):
                continue
            sid = f"{scene_id}_{frame_id}_{i}"
            if sid not in pred_by_id:
                continue
            tag = 0 if 0 in tag_list else (2 if 2 in tag_list else 3)
            out.append({
                "id": sid,
                "scene_id": scene_id,
                "frame_id": frame_id,
                "qa_idx": i,
                "tag": tag,
                "question": qa["Q"],
                "gt": qa["A"],
                "pred": pred_by_id[sid]["answer"],
                "image_paths": image_paths,
            })
    return out


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_SHELL = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval Viz - __TITLE__</title>
<style>
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0; background: #f5f5f7; color: #222; }
@media (prefers-color-scheme: dark) {
  body { background: #1a1a1c; color: #ddd; }
  .card { background: #26262a !important; }
  .toolbar { background: #2a2a2e !important; border-bottom-color: #3a3a3e !important; }
  .badge.tag { background: #34343a !important; color: #ddd !important; }
  .obj-tag { background: #1d3a5f !important; color: #9cd0ff !important; }
  .miss { background: rgba(120, 120, 120, .3) !important; color: #aaa !important; }
  .extra { background: rgba(180, 140, 0, .35) !important; }
  .coord-table th { background: #2a2a2e !important; }
  .coord-table td, .coord-table th { border-color: #3a3a3e !important; }
  input, button, select { background: #2a2a2e !important; color: #ddd !important;
                          border: 1px solid #3a3a3e !important; }
}

.toolbar {
  position: sticky; top: 0; z-index: 10;
  background: #fff; border-bottom: 1px solid #ddd;
  padding: 10px 14px; display: flex; flex-wrap: wrap; gap: 8px 14px;
  align-items: center; font-size: 13px;
}
.toolbar .summary { font-weight: 600; }
.toolbar .summary .metric { margin-left: 10px; font-weight: 400; opacity: .8; }
.toolbar label { display: inline-flex; align-items: center; gap: 4px; }
.toolbar input[type=text] { padding: 4px 8px; width: 180px; }
.toolbar button { padding: 4px 10px; cursor: pointer; }
.toolbar button.active { background: #2563eb; color: #fff; border-color: #2563eb; }
.toolbar .pager { margin-left: auto; display: flex; align-items: center; gap: 6px; }

.list { padding: 12px; display: flex; flex-direction: column; gap: 10px; }

.card {
  background: #fff; border-radius: 6px; padding: 10px 12px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, .06);
  font-size: 13px; line-height: 1.45;
}
.card .head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; cursor: pointer; }
.card .head .id { font-family: ui-monospace, monospace; opacity: .55; font-size: 11px;
                  word-break: break-all; }
.card .head .preview { margin-top: 4px; flex-basis: 100%;
                       overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.card .head .preview .label { font-weight: 600; opacity: .7; margin-right: 4px; }
.card .body { display: none; margin-top: 10px; }
.card.open .body { display: block; }
.card .field { margin: 6px 0; }
.card .field .lbl { font-weight: 600; font-size: 11px; opacity: .65;
                    text-transform: uppercase; letter-spacing: .04em; margin-bottom: 2px; }
.card .field .val { white-space: pre-wrap; word-break: break-word; }

.badge { padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.badge.tag { background: #ececf0; color: #444; }
.badge.score { color: #fff; }
.badge.score.correct { background: #16a34a; }
.badge.score.partial { background: #d97706; }
.badge.score.wrong   { background: #dc2626; }

.obj-tag { background: #dbeafe; color: #1e40af; padding: 1px 5px;
           border-radius: 3px; font-family: ui-monospace, monospace; font-size: 11px; }
.miss   { background: rgba(150, 150, 150, .25); text-decoration: line-through; opacity: .55; }
.extra  { background: rgba(252, 211, 77, .55); }

.thumbs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 8px; }
.thumbs .cam { display: flex; flex-direction: column; gap: 2px; }
.thumbs .cam .name { font-size: 10px; opacity: .65; font-family: ui-monospace, monospace; }
.thumbs .cam-figure { position: relative; width: 100%; border-radius: 4px; overflow: hidden; }
.thumbs .cam-figure img { width: 100%; height: auto; display: block; vertical-align: top;
                          background: #ddd; cursor: zoom-in; }
.thumbs .cam-overlay { position: absolute; left: 0; top: 0; pointer-events: none; }

.overlay-legend {
  display: flex; flex-wrap: wrap; gap: 10px; margin-top: 6px; font-size: 11px; opacity: .85;
}
.overlay-legend span { display: inline-flex; align-items: center; gap: 5px; }
.overlay-legend i {
  display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  border: 1px solid rgba(0,0,0,.25);
}
.legend-gt { background: #22c55e; }
.legend-pr { background: #f97316; }

.coord-table { margin-top: 10px; border-collapse: collapse; font-size: 12px; }
.coord-table th, .coord-table td { border: 1px solid #ddd; padding: 3px 8px; text-align: left; }
.coord-table th { background: #f0f0f3; }
.coord-table .row-matched td { color: #16a34a; }
.coord-table .row-missed td { color: #888; }
.coord-table .row-extra td { color: #d97706; }

#empty { padding: 30px; text-align: center; opacity: .6; }
</style>
</head>
<body>
<div class="toolbar">
  <div class="summary">
    <span>__SRC_NAME__</span>
    <span class="metric">n=__N_TOTAL__</span>
    <span class="metric">tag0 acc=__ACC__</span>
    <span class="metric">tag2 rougeL=__ROUGE__</span>
    <span class="metric">tag3 f1=__MATCH__</span>
  </div>
  <label>tag:
    <button data-filter-tag="all" class="active">all</button>
    <button data-filter-tag="0">0</button>
    <button data-filter-tag="2">2</button>
    <button data-filter-tag="3">3</button>
  </label>
  <label>score:
    <button data-filter-score="all" class="active">all</button>
    <button data-filter-score="correct">correct</button>
    <button data-filter-score="partial">partial</button>
    <button data-filter-score="wrong">wrong</button>
  </label>
  <label>sort:
    <select id="sort">
      <option value="score_asc">score asc (worst first)</option>
      <option value="score_desc">score desc</option>
      <option value="id">id</option>
    </select>
  </label>
  <label>search: <input id="search" type="text" placeholder="Q / GT / Pred"></label>
  <div class="pager">
    <button id="prev">prev</button>
    <span id="pageinfo">1/1</span>
    <button id="next">next</button>
  </div>
</div>
<div id="list" class="list"></div>
<div id="empty" style="display:none">no records match current filters</div>

<script>
const RECORDS = __RECORDS_JSON__;
const PAGE_SIZE = 50;
const CAM_NAMES = __CAM_NAMES__;
const OVERLAY_COORD_SPACE = __OVERLAY_COORD_SPACE__;

const state = { tag: "all", score: "all", search: "", sort: "score_asc", page: 1 };
let view = [];

function escapeHtml(s){return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function encDataJson(obj){
  return JSON.stringify(obj ?? []).replace(/&/g,"&amp;").replace(/"/g,"&quot;");
}

function drawCamOverlay(img, canvas, gtPts, predPts){
  const w = Math.round(img.clientWidth), h = Math.round(img.clientHeight);
  if(w < 4 || h < 4) return;
  if(canvas.width !== w || canvas.height !== h){ canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const sx = w / OVERLAY_COORD_SPACE, sy = h / OVERLAY_COORD_SPACE;
  const r = Math.max(3, Math.min(w, h) * 0.012);
  function dot(px, py, color, label){
    ctx.fillStyle = color;
    ctx.strokeStyle = "rgba(0,0,0,.45)";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.font = `600 ${Math.max(9, Math.round(r * 2.2))}px ui-sans-serif, system-ui, sans-serif`;
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "rgba(0,0,0,.55)";
    ctx.lineWidth = 2.5;
    const t = String(label);
    const tw = ctx.measureText(t).width;
    let tx = px + r + 3, ty = py - r - 2;
    if(tx + tw > w - 2) tx = Math.max(2, px - r - tw - 4);
    if(ty < 12) ty = py + r + 12;
    ctx.strokeText(t, tx, ty);
    ctx.fillText(t, tx, ty);
  }
  function fmt224(v){ return String(Math.round(v * 100) / 100); }
  (gtPts || []).forEach((p, idx) => {
    if(!Array.isArray(p) || p.length < 2) return;
    const xy = fmt224(p[0]) + "," + fmt224(p[1]);
    const lab = (gtPts.length > 1) ? ("GT" + (idx + 1) + " " + xy) : ("GT " + xy);
    dot(p[0] * sx, p[1] * sy, "#22c55e", lab);
  });
  (predPts || []).forEach((p, idx) => {
    if(!Array.isArray(p) || p.length < 2) return;
    const xy = fmt224(p[0]) + "," + fmt224(p[1]);
    const lab = (predPts.length > 1) ? ("P" + (idx + 1) + " " + xy) : ("P " + xy);
    dot(p[0] * sx, p[1] * sy, "#f97316", lab);
  });
}

const OBJ_TAG_RE = /<c\d+,CAM_[A-Z_]+,\d+\.?\d*,\d+\.?\d*>/g;
function highlightObjTags(s){
  return escapeHtml(s).replace(/&lt;c\d+,CAM_[A-Z_]+,\d+\.?\d*,\d+\.?\d*&gt;/g,
                               m => `<span class="obj-tag">${m}</span>`);
}

function renderTokens(tokens, marks){
  // tokens is array of raw strings (preserve casing); marks: ok|miss|extra
  let out = "";
  for(let i=0;i<tokens.length;i++){
    const t = tokens[i];
    const m = marks[i];
    let html;
    if(/^<c\d+,CAM_[A-Z_]+,/.test(t)) html = `<span class="obj-tag">${escapeHtml(t)}</span>`;
    else html = escapeHtml(t);
    if(m === "miss") html = `<span class="miss">${html}</span>`;
    else if(m === "extra") html = `<span class="extra">${html}</span>`;
    out += html + " ";
  }
  return out;
}

function applyFilters(){
  const q = state.search.trim().toLowerCase();
  view = RECORDS.filter(r => {
    if(state.tag !== "all" && String(r.tag) !== state.tag) return false;
    if(state.score !== "all" && r.score_bucket !== state.score) return false;
    if(q && !(r.question.toLowerCase().includes(q) ||
              r.gt.toLowerCase().includes(q) ||
              r.pred.toLowerCase().includes(q) ||
              r.id.toLowerCase().includes(q))) return false;
    return true;
  });
  if(state.sort === "score_asc") view.sort((a,b) => a.score - b.score);
  else if(state.sort === "score_desc") view.sort((a,b) => b.score - a.score);
  else view.sort((a,b) => a.id.localeCompare(b.id));
  state.page = 1;
  render();
}

function renderCard(r){
  const head = `
    <div class="head">
      <span class="badge tag">tag${r.tag}</span>
      <span class="badge score ${r.score_bucket}">${r.score_bucket} ${r.score.toFixed(2)}</span>
      <span class="id">${escapeHtml(r.id)}</span>
      <div class="preview"><span class="label">Q:</span>${escapeHtml(r.question)}</div>
      <div class="preview"><span class="label">GT:</span>${escapeHtml(r.gt)}</div>
      <div class="preview"><span class="label">Pred:</span>${escapeHtml(r.pred)}</div>
    </div>`;

  const gtBlock = r.gt_marks
    ? renderTokens(r.gt_tokens, r.gt_marks)
    : highlightObjTags(r.gt);
  const predBlock = r.pred_marks
    ? renderTokens(r.pred_tokens, r.pred_marks)
    : highlightObjTags(r.pred);

  let coordTable = "";
  if(r.coord){
    const rows = [];
    for(const m of r.coord.matched){
      rows.push(`<tr class="row-matched"><td>matched</td><td>${m.gt_idx}</td><td>${m.pred_idx}</td><td>L1=${m.dist}</td></tr>`);
    }
    for(const gi of r.coord.missed){
      const g = r.coord.gt_pairs[gi];
      rows.push(`<tr class="row-missed"><td>missed (gt)</td><td>${gi}</td><td>-</td><td>(${g[0]}, ${g[1]})</td></tr>`);
    }
    for(const pi of r.coord.extra){
      const p = r.coord.pred_pairs[pi];
      rows.push(`<tr class="row-extra"><td>extra (pred)</td><td>-</td><td>${pi}</td><td>(${p[0]}, ${p[1]})</td></tr>`);
    }
    coordTable = `
      <div class="field">
        <div class="lbl">coord match (L1 &lt; 16 = matched)</div>
        <table class="coord-table">
          <tr><th>kind</th><th>gt_idx</th><th>pred_idx</th><th>info</th></tr>
          ${rows.join("")}
        </table>
      </div>`;
  }

  const thumbs = r.thumbs.map((src, i) => {
    const cam = CAM_NAMES[i];
    const ov = r.cam_overlay && r.cam_overlay[cam];
    const gtA = ov && ov.gt ? ov.gt : [];
    const prA = ov && ov.pred ? ov.pred : [];
    const hasOv = (gtA && gtA.length) || (prA && prA.length);
    const inner = hasOv
      ? `<div class="cam-figure" data-gt="${encDataJson(gtA)}" data-pred="${encDataJson(prA)}">
           <img loading="lazy" src="${src}" alt="${cam}" onclick="window.open(this.src, '_blank')">
           <canvas class="cam-overlay"></canvas>
         </div>`
      : `<div class="cam-figure">
           <img loading="lazy" src="${src}" alt="${cam}" onclick="window.open(this.src, '_blank')">
         </div>`;
    return `<div class="cam"><span class="name">${cam}</span>${inner}</div>`;
  }).join("");

  const legend = (() => {
    if(!r.cam_overlay) return "";
    let ngt = 0, npr = 0;
    for(const c of CAM_NAMES){
      const o = r.cam_overlay[c];
      if(!o) continue;
      ngt += (o.gt && o.gt.length) || 0;
      npr += (o.pred && o.pred.length) || 0;
    }
    if(!ngt && !npr) return "";
    return `<div class="overlay-legend">
      <span><i class="legend-gt"></i> Ground truth (${ngt})</span>
      <span><i class="legend-pr"></i> Prediction (${npr})</span>
      <span style="opacity:.7">coords → ${OVERLAY_COORD_SPACE}px space</span>
    </div>`;
  })();

  const body = `
    <div class="body">
      <div class="field"><div class="lbl">question</div>
        <div class="val">${highlightObjTags(r.question)}</div></div>
      <div class="field"><div class="lbl">ground truth</div>
        <div class="val">${gtBlock}</div></div>
      <div class="field"><div class="lbl">prediction</div>
        <div class="val">${predBlock}</div></div>
      ${coordTable}
      ${legend}
      <div class="thumbs">${thumbs}</div>
    </div>`;

  const div = document.createElement("div");
  div.className = "card";
  div.dataset.id = r.id;
  div.innerHTML = head + body;
  div.querySelector(".head").addEventListener("click", () => {
    const wasOpen = div.classList.contains("open");
    div.classList.toggle("open");
    if(!wasOpen && div.classList.contains("open")){
      requestAnimationFrame(() => {
        div.querySelectorAll(".cam-figure img").forEach(img => {
          const fig = img.closest(".cam-figure");
          const cv = fig && fig.querySelector("canvas.cam-overlay");
          if(cv){
            const g = JSON.parse(fig.getAttribute("data-gt") || "[]");
            const p = JSON.parse(fig.getAttribute("data-pred") || "[]");
            if((g && g.length) || (p && p.length)){
              if(img.complete) drawCamOverlay(img, cv, g, p);
              else img.addEventListener("load", () => drawCamOverlay(img, cv, g, p), {once:true});
            }
          }
        });
      });
    }
  });
  div.querySelectorAll(".cam-figure img").forEach(img => {
    img.addEventListener("load", () => {
      if(!div.classList.contains("open")) return;
      const fig = img.closest(".cam-figure");
      const cv = fig && fig.querySelector("canvas.cam-overlay");
      if(!cv) return;
      const g = JSON.parse(fig.getAttribute("data-gt") || "[]");
      const p = JSON.parse(fig.getAttribute("data-pred") || "[]");
      if((g && g.length) || (p && p.length)) drawCamOverlay(img, cv, g, p);
    });
  });
  return div;
}

function render(){
  const list = document.getElementById("list");
  const empty = document.getElementById("empty");
  list.innerHTML = "";
  if(!view.length){
    empty.style.display = "block";
    document.getElementById("pageinfo").textContent = "0/0";
    return;
  }
  empty.style.display = "none";
  const totalPages = Math.max(1, Math.ceil(view.length / PAGE_SIZE));
  if(state.page > totalPages) state.page = totalPages;
  const lo = (state.page - 1) * PAGE_SIZE;
  const slice = view.slice(lo, lo + PAGE_SIZE);
  for(const r of slice) list.appendChild(renderCard(r));
  document.getElementById("pageinfo").textContent =
    `${state.page}/${totalPages} (${view.length} matched)`;
  window.scrollTo({top: 0, behavior: "instant"});
}

document.querySelectorAll("[data-filter-tag]").forEach(b => {
  b.addEventListener("click", () => {
    document.querySelectorAll("[data-filter-tag]").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    state.tag = b.dataset.filterTag;
    applyFilters();
  });
});
document.querySelectorAll("[data-filter-score]").forEach(b => {
  b.addEventListener("click", () => {
    document.querySelectorAll("[data-filter-score]").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    state.score = b.dataset.filterScore;
    applyFilters();
  });
});
document.getElementById("sort").addEventListener("change", e => {
  state.sort = e.target.value;
  applyFilters();
});
let searchTimer = null;
document.getElementById("search").addEventListener("input", e => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.search = e.target.value; applyFilters(); }, 200);
});
document.getElementById("prev").addEventListener("click", () => {
  if(state.page > 1) { state.page--; render(); }
});
document.getElementById("next").addEventListener("click", () => {
  const total = Math.max(1, Math.ceil(view.length / PAGE_SIZE));
  if(state.page < total) { state.page++; render(); }
});

applyFilters();
</script>
</body>
</html>
"""


def build_records(aligned):
    """Add score / score_bucket / diff marks / coord match info."""
    out = []
    for r in aligned:
        score, extra = score_one(r["tag"], r["pred"], r["gt"])
        bucket = score_bucket(r["tag"], score)
        rec = {
            "id": r["id"],
            "scene_id": r["scene_id"],
            "frame_id": r["frame_id"],
            "qa_idx": r["qa_idx"],
            "tag": r["tag"],
            "score": round(score, 4),
            "score_bucket": bucket,
            "question": r["question"],
            "gt": r["gt"],
            "pred": r["pred"],
            "thumbs": [to_img_src(p) for p in r["image_paths"][:6]],
        }
        if r["tag"] == 2:
            gt_toks, gt_marks, pred_toks, pred_marks = diff_marks(r["gt"], r["pred"])
            rec["gt_tokens"] = gt_toks
            rec["gt_marks"] = gt_marks
            rec["pred_tokens"] = pred_toks
            rec["pred_marks"] = pred_marks
        if r["tag"] == 3 and extra is not None:
            rec["coord"] = extra
        cam_ov = merge_cam_points(r["gt"], r["pred"])
        if cam_ov:
            rec["cam_overlay"] = cam_ov
        out.append(rec)
    out.sort(key=lambda x: x["score"])  # worst first
    return out


def build_summary(records, src_name: str):
    by_tag = {0: [], 2: [], 3: []}
    for r in records:
        by_tag[r["tag"]].append(r["score"])

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "src_name": src_name,
        "n_total": len(records),
        "tag0": {"n": len(by_tag[0]), "accuracy": avg(by_tag[0])},
        "tag2": {"n": len(by_tag[2]), "rougeL_mean": avg(by_tag[2])},
        "tag3": {"n": len(by_tag[3]), "f1_mean": avg(by_tag[3])},
    }


def fmt_metric(x):
    return f"{x:.3f}" if isinstance(x, float) else "n/a"


def render_html(records, summary) -> str:
    html = HTML_SHELL
    html = html.replace("__TITLE__", summary["src_name"])
    html = html.replace("__SRC_NAME__", summary["src_name"])
    html = html.replace("__N_TOTAL__", str(summary["n_total"]))
    html = html.replace("__ACC__", fmt_metric(summary["tag0"]["accuracy"]))
    html = html.replace("__ROUGE__", fmt_metric(summary["tag2"]["rougeL_mean"]))
    html = html.replace("__MATCH__", fmt_metric(summary["tag3"]["f1_mean"]))
    html = html.replace("__RECORDS_JSON__",
                        json.dumps(records, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__CAM_NAMES__", json.dumps(CAM_NAMES))
    html = html.replace("__OVERLAY_COORD_SPACE__", str(OVERLAY_COORD_SPACE))
    return html


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build a static HTML eval report")
    ap.add_argument("--src", required=True,
                    help="path to inference json (output of tools/inference_batch.py)")
    ap.add_argument("--gt", default="data/DriveLM_nuScenes/refs/val_cot.json",
                    help="ground truth json (val_cot.json)")
    ap.add_argument("--out", default="viz_eval/",
                    help="output dir; will write {out}/{src_basename}.html")
    ap.add_argument("--limit", type=int, default=None,
                    help="for debug: only keep first N aligned records")
    args = ap.parse_args()

    src_name = Path(args.src).stem
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[viz] loading {args.src}")
    aligned = load_aligned(args.src, args.gt)
    print(f"[viz] aligned records (tag in 0/2/3): {len(aligned)}")
    if args.limit:
        aligned = aligned[: args.limit]
        print(f"[viz] --limit -> {len(aligned)}")

    print("[viz] scoring & building records")
    records = build_records(aligned)
    summary = build_summary(records, src_name)
    print(f"[viz] summary: {summary}")

    html = render_html(records, summary)
    report_path = out_dir / f"{src_name}.html"
    report_path.write_text(html, encoding="utf-8")
    size_mb = report_path.stat().st_size / 1024 / 1024
    print(f"[viz] wrote {report_path}  ({size_mb:.2f} MB)")
    print(f"[viz] copy {report_path} into your local DriveLM_nuScenes/ folder and open it")


if __name__ == "__main__":
    main()
