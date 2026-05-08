"""Build a static HTML report for the nuScenes object-localization QA dataset.

Companion to `tools/create_data/create_nus_detection_qa.py`. Loads the HuggingFace
``Dataset`` it produces (``id``, ``image_paths``, ``conversations``) and renders a
browsable, paginated, single-file HTML for sanity-checking the generated samples.

Differences from ``tools/visualize_eval.py``:
- no predictions: only question + ground-truth answer
- coord space is 448x448 (matches the generator's ``--img-size``)
- adds category-based filtering + per-category histogram stats

Output:
    {out}/{split_name}.html

Image references are written relative to a `DriveLM_nuScenes/` root, the same
convention as ``visualize_eval.py``. Open the HTML from a parent directory whose
relative ``nuscenes/samples/CAM_*/...jpg`` paths resolve.

Usage:
    python tools/visualize_nus_detection_qa.py \\
        --src data/nus_detection_qa/split/train \\
        --out viz_eval/ \\
        --limit 2000
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_from_disk


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

_OBJ_TAG_PARSE_RE = re.compile(
    r"<c\d+,(CAM_[A-Z_]+),(-?\d+\.?\d*),(-?\d+\.?\d*)>",
    re.IGNORECASE,
)

CAM_NAMES = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

_IMG_STRIP_PREFIX = "data/DriveLM_nuScenes/"

# Built-in categories tracked for filters / histograms; extended automatically when
# the question mentions an unknown category.
_KNOWN_CATEGORIES = ["car", "truck", "bus", "pedestrian", "bicycle", "motorcycle"]
_PLURAL_TO_SINGULAR = {
    "cars": "car", "trucks": "truck", "buses": "bus",
    "pedestrians": "pedestrian", "bicycles": "bicycle", "motorcycles": "motorcycle",
}


def to_img_src(image_path: str) -> str:
    if image_path.startswith(_IMG_STRIP_PREFIX):
        return image_path[len(_IMG_STRIP_PREFIX):]
    return image_path


def extract_category(question: str) -> str:
    """Heuristic: scan the question for the first known plural/singular noun."""
    q = question.lower()
    for plural, singular in _PLURAL_TO_SINGULAR.items():
        if plural in q:
            return singular
    for cat in _KNOWN_CATEGORIES:
        if re.search(rf"\b{cat}\b", q):
            return cat
    return "other"


def obj_tags_by_cam(text: str) -> dict:
    """Parse <cN,CAM_X,x,y> into {CAM_*: [[x, y], ...]}."""
    out: dict = {}
    for m in _OBJ_TAG_PARSE_RE.finditer(text):
        cam = m.group(1).upper()
        x, y = float(m.group(2)), float(m.group(3))
        out.setdefault(cam, []).append([x, y])
    return out


def total_boxes(answer: str) -> int:
    return sum(len(v) for v in obj_tags_by_cam(answer).values())


# ---------------------------------------------------------------------------
# record building
# ---------------------------------------------------------------------------

def build_records(ds, *, limit=None):
    records = []
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    for ex in ds:
        conv = ex["conversations"]
        question = conv[0]["value"]
        gt = conv[1]["value"]
        cam_pts = obj_tags_by_cam(gt)
        records.append({
            "id": ex["id"],
            "category": extract_category(question),
            "n_boxes": total_boxes(gt),
            "is_negative": gt.strip().lower().startswith("no"),
            "question": question,
            "gt": gt,
            "thumbs": [to_img_src(p) for p in ex["image_paths"][:6]],
            "cam_overlay": {cam: {"gt": pts} for cam, pts in cam_pts.items()} or None,
        })
    return records


def build_summary(records):
    cat_counter = Counter(r["category"] for r in records)
    cam_counter = Counter()
    n_neg = sum(1 for r in records if r["is_negative"])
    n_with_boxes = sum(1 for r in records if r["n_boxes"] > 0)
    total_boxes_all = sum(r["n_boxes"] for r in records)
    for r in records:
        if r["cam_overlay"]:
            for cam, info in r["cam_overlay"].items():
                cam_counter[cam] += len(info.get("gt", []))
    return {
        "n_total": len(records),
        "n_negative": n_neg,
        "n_with_boxes": n_with_boxes,
        "total_boxes": total_boxes_all,
        "avg_boxes": (total_boxes_all / len(records)) if records else 0.0,
        "categories": cat_counter.most_common(),
        "by_camera": [(c, cam_counter.get(c, 0)) for c in CAM_NAMES],
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_SHELL = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>nuScenes Detection QA - __TITLE__</title>
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
  .stats { background: #26262a !important; border-color: #3a3a3e !important; }
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
.toolbar select { padding: 3px 6px; }
.toolbar .pager { margin-left: auto; display: flex; align-items: center; gap: 6px; }

.stats {
  margin: 10px 14px 0; padding: 10px 14px; background: #fff;
  border: 1px solid #e5e5ea; border-radius: 6px; font-size: 12px;
  display: flex; flex-wrap: wrap; gap: 14px;
}
.stats .group { display: flex; flex-direction: column; gap: 2px; }
.stats .group .ttl { font-weight: 600; opacity: .7; font-size: 11px; text-transform: uppercase; }
.stats .row { display: flex; gap: 8px; align-items: center; }
.stats .bar { display: inline-block; height: 8px; background: #2563eb; border-radius: 2px; }

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
.badge.cat { background: #ddd6fe; color: #5b21b6; }
.badge.boxes { background: #dcfce7; color: #166534; }
.badge.boxes.zero { background: #fee2e2; color: #991b1b; }

.obj-tag { background: #dbeafe; color: #1e40af; padding: 1px 5px;
           border-radius: 3px; font-family: ui-monospace, monospace; font-size: 11px; }

.thumbs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 8px; }
.thumbs .cam { display: flex; flex-direction: column; gap: 2px; }
.thumbs .cam .name { font-size: 10px; opacity: .65; font-family: ui-monospace, monospace; }
.thumbs .cam-figure { position: relative; width: 100%; border-radius: 4px; overflow: hidden; }
.thumbs .cam-figure img { width: 100%; height: auto; display: block; vertical-align: top;
                          background: #ddd; cursor: zoom-in; }
.thumbs .cam-overlay { position: absolute; left: 0; top: 0; pointer-events: none; }

#empty { padding: 30px; text-align: center; opacity: .6; }
</style>
</head>
<body>
<div class="toolbar">
  <div class="summary">
    <span>__TITLE__</span>
    <span class="metric">n=__N_TOTAL__</span>
    <span class="metric">+boxes=__N_WITH_BOXES__</span>
    <span class="metric">none=__N_NEGATIVE__</span>
    <span class="metric">avg=__AVG_BOXES__/sample</span>
  </div>
  <label>category:
    <select id="cat-filter">
      <option value="all">all</option>
      __CAT_OPTIONS__
    </select>
  </label>
  <label>has-boxes:
    <button data-filter-has="all" class="active">all</button>
    <button data-filter-has="yes">yes</button>
    <button data-filter-has="no">no (None.)</button>
  </label>
  <label>sort:
    <select id="sort">
      <option value="boxes_desc">boxes desc</option>
      <option value="boxes_asc">boxes asc</option>
      <option value="id">id</option>
    </select>
  </label>
  <label>search: <input id="search" type="text" placeholder="Q / GT / id"></label>
  <div class="pager">
    <button id="prev">prev</button>
    <span id="pageinfo">1/1</span>
    <button id="next">next</button>
  </div>
</div>

<div class="stats">
  <div class="group">
    <div class="ttl">categories</div>
    __CAT_HISTOGRAM__
  </div>
  <div class="group">
    <div class="ttl">boxes per camera</div>
    __CAM_HISTOGRAM__
  </div>
</div>

<div id="list" class="list"></div>
<div id="empty" style="display:none">no records match current filters</div>

<script>
const RECORDS = __RECORDS_JSON__;
const PAGE_SIZE = 50;
const CAM_NAMES = __CAM_NAMES__;
const OVERLAY_COORD_SPACE = __OVERLAY_COORD_SPACE__;

const state = { cat: "all", has: "all", search: "", sort: "boxes_desc", page: 1 };
let view = [];

function escapeHtml(s){return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function encDataJson(obj){
  return JSON.stringify(obj ?? []).replace(/&/g,"&amp;").replace(/"/g,"&quot;");
}

function drawCamOverlay(img, canvas, gtPts){
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
  function fmtCoord(v){ return String(Math.round(v * 100) / 100); }
  (gtPts || []).forEach((p, idx) => {
    if(!Array.isArray(p) || p.length < 2) return;
    const xy = fmtCoord(p[0]) + "," + fmtCoord(p[1]);
    const lab = (gtPts.length > 1) ? ((idx + 1) + " " + xy) : xy;
    dot(p[0] * sx, p[1] * sy, "#22c55e", lab);
  });
}

function highlightObjTags(s){
  return escapeHtml(s).replace(/&lt;c\d+,CAM_[A-Z_]+,-?\d+\.?\d*,-?\d+\.?\d*&gt;/g,
                               m => `<span class="obj-tag">${m}</span>`);
}

function applyFilters(){
  const q = state.search.trim().toLowerCase();
  view = RECORDS.filter(r => {
    if(state.cat !== "all" && r.category !== state.cat) return false;
    if(state.has === "yes" && r.n_boxes === 0) return false;
    if(state.has === "no"  && r.n_boxes  >  0) return false;
    if(q && !(r.question.toLowerCase().includes(q) ||
              r.gt.toLowerCase().includes(q) ||
              r.id.toLowerCase().includes(q))) return false;
    return true;
  });
  if(state.sort === "boxes_desc") view.sort((a,b) => b.n_boxes - a.n_boxes);
  else if(state.sort === "boxes_asc") view.sort((a,b) => a.n_boxes - b.n_boxes);
  else view.sort((a,b) => a.id.localeCompare(b.id));
  state.page = 1;
  render();
}

function renderCard(r){
  const boxesBadgeClass = r.n_boxes > 0 ? "badge boxes" : "badge boxes zero";
  const head = `
    <div class="head">
      <span class="badge cat">${escapeHtml(r.category)}</span>
      <span class="${boxesBadgeClass}">${r.n_boxes} box${r.n_boxes === 1 ? "" : "es"}</span>
      <span class="id">${escapeHtml(r.id)}</span>
      <div class="preview"><span class="label">Q:</span>${escapeHtml(r.question)}</div>
      <div class="preview"><span class="label">A:</span>${escapeHtml(r.gt)}</div>
    </div>`;

  const thumbs = r.thumbs.map((src, i) => {
    const cam = CAM_NAMES[i];
    const ov = r.cam_overlay && r.cam_overlay[cam];
    const gtA = ov && ov.gt ? ov.gt : [];
    const inner = (gtA && gtA.length)
      ? `<div class="cam-figure" data-gt="${encDataJson(gtA)}">
           <img loading="lazy" src="${src}" alt="${cam}" onclick="window.open(this.src, '_blank')">
           <canvas class="cam-overlay"></canvas>
         </div>`
      : `<div class="cam-figure">
           <img loading="lazy" src="${src}" alt="${cam}" onclick="window.open(this.src, '_blank')">
         </div>`;
    return `<div class="cam"><span class="name">${cam}</span>${inner}</div>`;
  }).join("");

  const body = `
    <div class="body">
      <div class="field"><div class="lbl">question</div>
        <div class="val">${highlightObjTags(r.question)}</div></div>
      <div class="field"><div class="lbl">answer (ground truth)</div>
        <div class="val">${highlightObjTags(r.gt)}</div></div>
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
            if(g && g.length){
              if(img.complete) drawCamOverlay(img, cv, g);
              else img.addEventListener("load", () => drawCamOverlay(img, cv, g), {once:true});
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
      if(g && g.length) drawCamOverlay(img, cv, g);
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

document.getElementById("cat-filter").addEventListener("change", e => {
  state.cat = e.target.value;
  applyFilters();
});
document.querySelectorAll("[data-filter-has]").forEach(b => {
  b.addEventListener("click", () => {
    document.querySelectorAll("[data-filter-has]").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    state.has = b.dataset.filterHas;
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


def _render_histogram(pairs, max_label_width=14):
    if not pairs:
        return "<div>(empty)</div>"
    max_v = max(v for _, v in pairs) or 1
    rows = []
    for k, v in pairs:
        pct = v / max_v
        bar_w = max(4, int(round(pct * 120)))
        rows.append(
            f'<div class="row">'
            f'<span style="width:{max_label_width}ch; font-family:ui-monospace,monospace">{k}</span>'
            f'<span class="bar" style="width:{bar_w}px"></span>'
            f'<span style="opacity:.7">{v}</span>'
            f"</div>"
        )
    return "\n".join(rows)


def render_html(records, summary, *, title: str, coord_space: int) -> str:
    cat_options = "".join(
        f'<option value="{cat}">{cat} ({n})</option>'
        for cat, n in summary["categories"]
    )
    cat_hist = _render_histogram(summary["categories"])
    cam_hist = _render_histogram(summary["by_camera"])

    html = HTML_SHELL
    html = html.replace("__TITLE__", title)
    html = html.replace("__N_TOTAL__", str(summary["n_total"]))
    html = html.replace("__N_WITH_BOXES__", str(summary["n_with_boxes"]))
    html = html.replace("__N_NEGATIVE__", str(summary["n_negative"]))
    html = html.replace("__AVG_BOXES__", f"{summary['avg_boxes']:.2f}")
    html = html.replace("__CAT_OPTIONS__", cat_options)
    html = html.replace("__CAT_HISTOGRAM__", cat_hist)
    html = html.replace("__CAM_HISTOGRAM__", cam_hist)
    html = html.replace("__RECORDS_JSON__",
                        json.dumps(records, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__CAM_NAMES__", json.dumps(CAM_NAMES))
    html = html.replace("__OVERLAY_COORD_SPACE__", str(coord_space))
    return html


def main():
    ap = argparse.ArgumentParser(description="Build a static HTML viewer for nuScenes detection QA")
    ap.add_argument("--src", required=True,
                    help="HF Dataset directory produced by create_nus_detection_qa.py "
                         "(e.g. data/nus_detection_qa/split/train)")
    ap.add_argument("--out", default="viz_eval/", help="output dir")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of samples (useful for huge train splits)")
    ap.add_argument("--coord-space", type=int, default=448,
                    help="coordinate space the answers use (default 448, matches generator's --img-size)")
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[viz] loading {args.src}")
    ds = load_from_disk(str(src))
    print(f"[viz] dataset size = {len(ds)}")

    print("[viz] parsing records")
    records = build_records(ds, limit=args.limit)
    summary = build_summary(records)
    print(f"[viz] summary: n_total={summary['n_total']} "
          f"+boxes={summary['n_with_boxes']} none={summary['n_negative']} "
          f"total_boxes={summary['total_boxes']} avg={summary['avg_boxes']:.2f}")
    print(f"[viz] categories: {summary['categories']}")
    print(f"[viz] per-cam: {summary['by_camera']}")

    title = f"{src.parent.name}/{src.name}"
    html = render_html(records, summary, title=title, coord_space=args.coord_space)
    name = src.name if src.name not in ("", ".") else src.parent.name
    out_path = out_dir / f"nus_det_qa_{name}.html"
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[viz] wrote {out_path}  ({size_mb:.2f} MB)")
    print(f"[viz] copy {out_path} into your local DriveLM_nuScenes/ folder and open it")


if __name__ == "__main__":
    main()
