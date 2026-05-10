"""Generate object-localization QA from nuScenes 3D annotations.

For each ``sample`` in nuScenes, project 3D bounding boxes into each of the 6
surround-view cameras using nuscenes-devkit, take the projected box-center, scale
it to the training-image space (default 448x448), and emit Q/A pairs in the same
HuggingFace ``Dataset`` schema as DriveLM (``id``, ``image_paths``, ``conversations``).

Answers use the standard DriveLM coordinate-tag format ``<cN,CAM_X,xxx,yyy>`` with
floating-point pixel values; the ``<loc_k>`` quantization happens later in the
collate function. So the produced dataset stays human-readable and is also
compatible with the original (non-loc-token) training path.

Each generated question may target 1-3 random categories chosen from those
present in the frame (e.g. "Where are the cars and pedestrians?"). The answer
concatenates tags from every requested category with a single global ``cN``
counter, in the order the user asked.

Example:
    python tools/create_data/create_nus_detection_qa.py \\
        --nuscenes-root data/DriveLM_nuScenes/nuscenes \\
        --version v1.0-trainval \\
        --out data/nus_detection_qa/split \\
        --img-size 448 \\
        --max-distance 60 \\
        --samples-per-frame 3 \\
        --max-categories-per-question 3
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
from collections import Counter
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
from tqdm import tqdm


CAMERAS: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


# Comprehensive default covering every nuScenes v1.0 category (~23). Each friendly
# bucket maps to one or more ``category.name`` prefixes (matched via ``startswith``,
# longest-prefix wins so subclasses are routed to the most specific bucket).
DEFAULT_CATEGORY_GROUPS: Dict[str, List[str]] = {
    "car":                  ["vehicle.car"],
    "truck":                ["vehicle.truck"],
    "construction_vehicle": ["vehicle.construction"],
    "bus":                  ["vehicle.bus"],            # rigid + bendy
    "trailer":              ["vehicle.trailer"],
    "motorcycle":           ["vehicle.motorcycle"],
    "bicycle":              ["vehicle.bicycle"],
    "emergency_vehicle":    ["vehicle.emergency"],      # ambulance + police
    "pedestrian":           ["human.pedestrian"],       # all 7 subtypes
    "animal":               ["animal"],
    "barrier":              ["movable_object.barrier"],
    "traffic_cone":         ["movable_object.trafficcone"],
    "debris":               ["movable_object.debris"],
    "pushable_pullable":    ["movable_object.pushable_pullable"],
    "bicycle_rack":         ["static_object.bicycle_rack"],
}


# (plural noun phrase, singular noun phrase) for each bucket.
CATEGORY_NAMES: Dict[str, Tuple[str, str]] = {
    "car":                  ("cars", "car"),
    "truck":                ("trucks", "truck"),
    "construction_vehicle": ("construction vehicles", "construction vehicle"),
    "bus":                  ("buses", "bus"),
    "trailer":              ("trailers", "trailer"),
    "motorcycle":           ("motorcycles", "motorcycle"),
    "bicycle":              ("bicycles", "bicycle"),
    "emergency_vehicle":    ("emergency vehicles", "emergency vehicle"),
    "pedestrian":           ("pedestrians", "pedestrian"),
    "animal":               ("animals", "animal"),
    "barrier":              ("barriers", "barrier"),
    "traffic_cone":         ("traffic cones", "traffic cone"),
    # ``debris`` is a mass noun; no countable singular form.
    "debris":               ("pieces of debris", "piece of debris"),
    "pushable_pullable":    ("pushable/pullable objects", "pushable/pullable object"),
    "bicycle_rack":         ("bicycle racks", "bicycle rack"),
}


# Positive multi-category templates. Use ``{plurals}`` for noun phrases that should
# be plural ("cars and pedestrians") and ``{singulars}`` for "every X" forms.
DEFAULT_TEMPLATES_PLURAL = [
    "Where are the {plurals} in this scene?",
    "Where are the {plurals}?",
    "List all visible {plurals}.",
    "List the {plurals} you can see.",
    "Identify the positions of all {plurals}.",
    "Point out every visible {plurals} in the surround view.",
    "Find all {plurals} in the surrounding cameras.",
    "Mark the locations of all {plurals}.",
    "Show me where the {plurals} are.",
    "Detect all {plurals} visible in any of the six cameras.",
    "Enumerate the {plurals} present in the scene.",
    "Which positions correspond to the {plurals}?",
]

DEFAULT_TEMPLATES_SINGULAR = [
    "Locate every {singulars} you can see.",
    "Indicate the position of each {singulars}.",
    "Point out each {singulars} visible in the cameras.",
    "Pinpoint every {singulars} around the ego vehicle.",
    "Highlight each {singulars} in the surround view.",
]

# Existence questions. Always single-category to keep the answer unambiguous.
DEFAULT_EXISTENCE_TEMPLATES = [
    "Are there any {plural} visible?",
    "Is there a {singular} in this scene?",
    "Can you see any {plural}?",
    "Are any {plural} present?",
    "Do you see a {singular}?",
    "Is there at least one {singular} around the ego vehicle?",
    "Are any {plural} visible in the surround view?",
    "Is a {singular} visible from any camera?",
]

# Affirmative answer prefixes for existence questions when objects exist.
DEFAULT_AFFIRMATIVE_PREFIXES = [
    "Yes. ",
    "Yes, I can see ",
    "Yes: ",
    "Yes, there is ",
    "Yes, the following are visible: ",
    "Yes, ",
]

# Negative answers for existence questions when nothing matches.
DEFAULT_NEGATIVE_ANSWERS = [
    "No.",
    "No, none are visible.",
    "No, I cannot see any.",
    "No, there are none.",
    "None visible.",
    "No, none in any of the six cameras.",
]

# Optional lead-ins for plain "list/locate" answers; used 50% of the time so the
# model sees both bare-tag-list and lightly-narrated forms.
DEFAULT_POSITIVE_PREFIXES = [
    "",
    "",
    "",
    "I can see ",
    "There are ",
    "Visible objects: ",
    "Detected: ",
    "The following are visible: ",
]

DEFAULT_EMPTY_LIST_ANSWERS = [
    "None.",
    "None visible.",
    "I do not see any.",
]


def _classify(category_name: str, groups: Dict[str, List[str]]) -> str | None:
    """Map a nuScenes ``category.name`` to a friendly bucket name.

    Uses *longest-prefix* matching so e.g. ``vehicle.bus.bendy`` routes to ``bus``
    (prefix ``vehicle.bus``) rather than to a shorter ``vehicle`` bucket if one
    were ever added.
    """
    best_friendly, best_len = None, -1
    for friendly, prefixes in groups.items():
        for p in prefixes:
            if category_name.startswith(p) and len(p) > best_len:
                best_friendly, best_len = friendly, len(p)
    return best_friendly


def _box_center_and_area_2d(box, K: np.ndarray) -> Tuple[float, float, float] | None:
    if box.center[2] <= 0:
        return None
    corners_2d = view_points(box.corners(), K, normalize=True)[:2]
    x_min, y_min = corners_2d.min(axis=1)
    x_max, y_max = corners_2d.max(axis=1)
    cx = float((x_min + x_max) / 2.0)
    cy = float((y_min + y_max) / 2.0)
    area = float((x_max - x_min) * (y_max - y_min))
    return cx, cy, area


def _scale_to_target(cx: float, cy: float, orig_size, img_size: int):
    return cx * img_size / orig_size[0], cy * img_size / orig_size[1]


def _resolve_image_paths(nusc: NuScenes, sample, dataroot: str) -> List[str] | None:
    paths = []
    for cam in CAMERAS:
        sd_token = sample["data"][cam]
        sd = nusc.get("sample_data", sd_token)
        paths.append(os.path.join(dataroot, sd["filename"]))
    return paths


def _collect_per_cam_boxes(
    nusc: NuScenes, sample, *, img_size: int, max_distance: float,
    groups: Dict[str, List[str]],
) -> Dict[str, List[Tuple[str, float, float, float]]]:
    per_cam: Dict[str, List[Tuple[str, float, float, float]]] = {c: [] for c in CAMERAS}
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
            result = _box_center_and_area_2d(box, K)
            if result is None:
                continue
            cx_full, cy_full, area_full = result
            cx, cy = _scale_to_target(cx_full, cy_full, (1600, 900), img_size)
            if not (0.0 <= cx < img_size and 0.0 <= cy < img_size):
                continue
            per_cam[cam].append((cat, cx, cy, area_full))
    return per_cam


def _humanize_list(items: List[str], conj: str = "and") -> str:
    """Oxford-comma join: ['a'] -> 'a', ['a','b'] -> 'a and b', ['a','b','c'] -> 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return ", ".join(items[:-1]) + f", {conj} " + items[-1]


def _names_for(cat: str) -> Tuple[str, str]:
    """Look up (plural, singular) noun phrases; fallback to ``cat`` + 's'."""
    return CATEGORY_NAMES.get(cat, (cat + "s", cat))


def _pick_n_categories(
    cats_present: List[str], rng: random.Random, max_cats: int,
) -> List[str]:
    """Choose 1..min(max_cats, len(cats_present)) distinct categories, biased to single-cat."""
    upper = min(max_cats, len(cats_present))
    if upper <= 1:
        return [rng.choice(cats_present)] if cats_present else []
    # Bias toward fewer categories; weights are clipped to ``upper``.
    weights = [60, 40, 20, 10][:upper]
    n = rng.choices(range(1, upper + 1), weights=weights, k=1)[0]
    return rng.sample(cats_present, n)


def _gather_tags_by_cat(
    chosen_cats: List[str],
    per_cam: Dict[str, List[Tuple[str, float, float, float]]],
    *,
    top_k_per_cat: int,
) -> List[Tuple[str, List[str]]]:
    """Group top-k tags per chosen category, preserving ``chosen_cats`` order.

    Returns a list of ``(category, [tags])`` pairs. ``cN`` counter is global and
    increments in final emission order; categories with no visible boxes are
    still included with an empty list.
    """
    result: List[Tuple[str, List[str]]] = []
    cid = 1
    for cat in chosen_cats:
        candidates: List[Tuple[float, int, str, float, float]] = []
        for cam_idx, cam in enumerate(CAMERAS):
            for c, x, y, area in per_cam[cam]:
                if c != cat:
                    continue
                # Negative area so Python's stable sort gives area desc, then cam asc.
                candidates.append((-area, cam_idx, cam, x, y))
        candidates.sort()
        tags = []
        for _, _, cam, x, y in candidates[:top_k_per_cat]:
            tags.append(f"<c{cid},{cam},{x:.2f},{y:.2f}>")
            cid += 1
        result.append((cat, tags))
    return result


def _format_positive_question(
    chosen_cats: List[str],
    rng: random.Random,
    plural_templates: List[str],
    singular_templates: List[str],
) -> str:
    plurals_phrase = _humanize_list([_names_for(c)[0] for c in chosen_cats])
    singulars_phrase = _humanize_list([_names_for(c)[1] for c in chosen_cats])
    pool = []
    for t in plural_templates:
        pool.append(("plural", t))
    for t in singular_templates:
        pool.append(("singular", t))
    kind, tmpl = rng.choice(pool)
    if kind == "plural":
        return tmpl.format(plurals=plurals_phrase)
    return tmpl.format(singulars=singulars_phrase)


def _format_category_section(cat: str, tags: List[str]) -> str:
    """Fixed ``Plural: <tag>, <tag>.`` section, with plural capitalized."""
    plural, _ = _names_for(cat)
    label = plural[:1].upper() + plural[1:]
    if not tags:
        return f"{label}: none."
    return f"{label}: {', '.join(tags)}."


def _format_positive_answer(grouped: List[Tuple[str, List[str]]]) -> str:
    """Deterministic multi-category answer: ``Cars: <..>. Pedestrians: <..>.``"""
    if not any(tags for _, tags in grouped):
        return "None."
    sections = [_format_category_section(cat, tags) for cat, tags in grouped]
    return " ".join(sections)


def _format_existence_qa(
    cat: str,
    per_cam: Dict[str, List[Tuple[str, float, float, float]]],
    existence_templates: List[str],
    rng: random.Random,
    *,
    top_k_per_cat: int,
) -> Tuple[str, str]:
    plural, singular = _names_for(cat)
    q = rng.choice(existence_templates).format(plural=plural, singular=singular)
    grouped = _gather_tags_by_cat([cat], per_cam, top_k_per_cat=top_k_per_cat)
    tags = grouped[0][1]
    if tags:
        return q, f"Yes. {_format_category_section(cat, tags)}"
    return q, "No."


def _build_qa(
    per_cam: Dict[str, List[Tuple[str, float, float, float]]],
    *,
    plural_templates: List[str],
    singular_templates: List[str],
    existence_templates: List[str],
    rng: random.Random,
    max_cats: int,
    p_existence: float,
) -> Tuple[str, str] | None:
    cats_present = sorted({c for boxes in per_cam.values() for c, _, _, _ in boxes})

    # Each question keeps at most ``top_k_per_cat`` largest boxes per category,
    # sampled per-question to give the model variety between 1- and 4-answer forms.
    top_k_per_cat = rng.randint(1, 4)

    # Prefer positive multi-cat questions when the frame has any objects.
    if cats_present and rng.random() > p_existence:
        chosen = _pick_n_categories(cats_present, rng, max_cats)
        if chosen:
            q = _format_positive_question(chosen, rng, plural_templates, singular_templates)
            grouped = _gather_tags_by_cat(chosen, per_cam, top_k_per_cat=top_k_per_cat)
            a = _format_positive_answer(grouped)
            return q, a

    # Existence question: pick any known category. Sometimes hits a category that
    # IS present (Yes + tags), sometimes one that isn't (No), giving the model
    # both signals at the same template.
    cat = rng.choice(list(CATEGORY_NAMES.keys()))
    return _format_existence_qa(
        cat, per_cam, existence_templates, rng, top_k_per_cat=top_k_per_cat,
    )


def _split_scenes(nusc: NuScenes, val_ratio: float, seed: int):
    scenes = sorted(nusc.scene, key=lambda s: s["name"])
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_val = max(1, int(len(scenes) * val_ratio))
    return {s["token"] for s in scenes[:n_val]}


def _load_templates_json(path: str | None, fallback: List[str]) -> List[str]:
    if not path:
        return fallback
    with open(path) as f:
        loaded = json.load(f)
    if not isinstance(loaded, list) or not loaded:
        raise SystemExit(f"{path} must contain a non-empty JSON list")
    return loaded


def _warn_uncovered_categories(nusc: NuScenes, groups: Dict[str, List[str]]):
    seen = sorted({c["name"] for c in nusc.category})
    missing = [c for c in seen if _classify(c, groups) is None]
    if missing:
        print(f"[warn] {len(missing)} nuScenes categories not covered by --categories-json "
              f"and will be skipped: {missing}")


def generate(args):
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=False)

    if args.categories_json:
        with open(args.categories_json) as f:
            groups = json.load(f)
    else:
        groups = DEFAULT_CATEGORY_GROUPS
    _warn_uncovered_categories(nusc, groups)

    plural_templates = _load_templates_json(args.templates_json, DEFAULT_TEMPLATES_PLURAL)
    singular_templates = _load_templates_json(args.templates_singular_json,
                                              DEFAULT_TEMPLATES_SINGULAR)
    existence_templates = _load_templates_json(args.existence_templates_json,
                                               DEFAULT_EXISTENCE_TEMPLATES)

    val_tokens = _split_scenes(nusc, args.val_ratio, args.seed)
    rng = random.Random(args.seed)

    # Resolve total-cap into per-split caps. None = no cap (process every sample).
    if args.max_total_qa is not None:
        max_val_qa = int(round(args.max_total_qa * args.val_ratio))
        max_train_qa = args.max_total_qa - max_val_qa
        print(f"[cap] max_total_qa={args.max_total_qa} -> "
              f"train cap={max_train_qa}, val cap={max_val_qa}")
    else:
        max_train_qa = None
        max_val_qa = None

    # Shuffle the sample iteration order. Without this, ``nusc.sample`` is in
    # scene-name order, so an early stop (cap hit) would oversample a few scenes.
    sample_indices = list(range(len(nusc.sample)))
    rng.shuffle(sample_indices)

    train_rows, val_rows = [], []
    n_skipped_missing_imgs = 0
    n_processed = 0
    cat_q_counter: Counter = Counter()
    n_existence_q = 0
    n_multi_cat_q = 0

    def _train_full() -> bool:
        return max_train_qa is not None and len(train_rows) >= max_train_qa

    def _val_full() -> bool:
        return max_val_qa is not None and len(val_rows) >= max_val_qa

    # Generating tens of thousands of rows makes gen2 GC sweeps grow linearly with
    # the number of tracked objects, which manifests as gradually slowing iteration.
    # We don't build cycles, so it's safe to disable cyclic GC during the hot loop
    # and let refcounting reclaim memory as usual.
    gc.disable()
    try:
        pbar = tqdm(sample_indices, desc="generating QA", unit="sample")
        for s_idx in pbar:
        if _train_full() and _val_full():
            break
        sample = nusc.sample[s_idx]
        scene_token = sample["scene_token"]
        is_val = scene_token in val_tokens

        # Skip samples whose target split is already full.
        if is_val and _val_full():
            continue
        if (not is_val) and _train_full():
            continue

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
            if is_val and _val_full():
                break
            if (not is_val) and _train_full():
                break
            qa = _build_qa(
                per_cam,
                plural_templates=plural_templates,
                singular_templates=singular_templates,
                existence_templates=existence_templates,
                rng=rng,
                max_cats=args.max_categories_per_question,
                p_existence=args.p_existence,
            )
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

            mentioned = [c for c in CATEGORY_NAMES if _names_for(c)[0] in q or _names_for(c)[1] in q]
            for c in mentioned:
                cat_q_counter[c] += 1
            if len(mentioned) >= 2:
                n_multi_cat_q += 1
            if any(t.split()[0] in q.split() for t in ("Are", "Is", "Can", "Do")):
                n_existence_q += 1

        n_processed += 1
        pbar.set_postfix(
            train=f"{len(train_rows)}/{max_train_qa if max_train_qa is not None else '∞'}",
            val=f"{len(val_rows)}/{max_val_qa if max_val_qa is not None else '∞'}",
        )

    print(f"skipped {n_skipped_missing_imgs} samples (missing image files)")
    print(f"final: train={len(train_rows)}, val={len(val_rows)}")
    print(f"existence-style questions ≈ {n_existence_q}, "
          f"multi-category questions ≈ {n_multi_cat_q}")
    print(f"category coverage in questions: "
          f"{cat_q_counter.most_common()}")

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
    parser.add_argument("--samples-per-frame", type=int, default=3,
                        help="how many Q/A pairs to emit per nuScenes sample")
    parser.add_argument("--max-total-qa", type=int, default=None,
                        help="cap on the total number of Q/A pairs generated across "
                             "train + val. When set, the script iterates samples in a "
                             "deterministic random order and stops once both per-split "
                             "caps are reached. Per-split caps are derived from "
                             "--val-ratio (e.g. --max-total-qa 20000 with --val-ratio 0.15 "
                             "→ ~17000 train + ~3000 val).")
    parser.add_argument("--max-categories-per-question", type=int, default=3,
                        help="cap on how many categories a single question can ask about")
    parser.add_argument("--p-existence", type=float, default=0.15,
                        help="probability of issuing an existence (Yes/No) question instead "
                             "of a positive multi-category one")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories-json", default=None,
                        help="optional JSON mapping {friendly_name: [category prefixes]} that "
                             "REPLACES the default mapping")
    parser.add_argument("--templates-json", default=None,
                        help="optional JSON list of plural-form positive templates with "
                             "a {plurals} placeholder")
    parser.add_argument("--templates-singular-json", default=None,
                        help="optional JSON list of singular-form positive templates with "
                             "a {singulars} placeholder")
    parser.add_argument("--existence-templates-json", default=None,
                        help="optional JSON list of existence-question templates with "
                             "{plural} and {singular} placeholders")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
