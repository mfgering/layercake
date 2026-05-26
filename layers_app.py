#!/usr/bin/env python3
"""
layercake — interactive Gradio UI for the layers.py CLI.

Workflow:
  1. Upload source image.
  2. Add named layers (order = depth; first = nearest).
  3. For the active layer, click to add points (include/exclude), drop a
     box with two clicks, move a point (two clicks: pick + drop), or erase.
  4. Save → depth-ordered RGBA PNGs at source dims + points.json + snippet.html.

Notes:
  - SAM 2 supports multiple points + one axis-aligned box per predict() call.
    Multiple contiguous blobs? Make multiple layers rather than cramming one.
  - Undo pops the last added thing (point or box) for the active layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from layers import (
    _coerce_alpha,
    autocast_ctx,
    bool_mask_to_sam2_prior,
    build_css_snippet,
    detect_device,
    infill_background,
    load_sam2_predictor,
    load_sam3_concept_pipeline,
    mask_to_rgba,
    matting_refine,
    segment_by_concept,
)

LAYER_COLORS = [
    (255, 80, 80),
    (80, 200, 120),
    (80, 140, 255),
    (230, 190, 80),
    (200, 100, 230),
    (80, 220, 220),
]

PREVIEW_MAX_DIM = 1280  # longest edge of the click canvas; source is always full-res

STATE: dict = {
    "predictor": None,
    "model_loaded": None,
    "device_resolved": None,
    "sam3_pipe": None,            # (Sam3Model, Sam3Processor) lazy-loaded
    "sam3_device": None,
    "rgb": None,
    # View transform: the preview shows (origin_x, origin_y, view_w, view_h)
    # from the source image, scaled by preview_scale. When zoom is off, origin
    # is (0,0) and view is (W,H). When zoomed, origin is the zoom box's top-left
    # and view is its dims.
    "preview_origin": (0, 0),
    "preview_scale": 1.0,
    "zoom_box": None,             # [x1,y1,x2,y2] in SOURCE coords, or None
    "awaiting_zoom": False,       # True while capturing 2 clicks to define a zoom box
    "pending_zoom_corner": None,  # [x, y] in SOURCE coords, after 1st zoom click
    "cached_masks": {},
    "pending_box_corner": None,   # [x, y] in SOURCE coords, awaiting 2nd box click
    "move_selected": None,        # (layer_name, point_idx) awaiting drop
}


def _load_predictor(model: str, device: str):
    device_resolved = detect_device(device)
    if (STATE["predictor"] is None or STATE["model_loaded"] != model
            or STATE["device_resolved"] != device_resolved):
        STATE["predictor"] = load_sam2_predictor(model, device_resolved)
        STATE["model_loaded"] = model
        STATE["device_resolved"] = device_resolved
    return STATE["predictor"], device_resolved


def _new_layer(name: str) -> dict:
    # SAM 2 returns 3 candidate masks per predict() call at subpart/part/whole
    # scale. We store all of them so the user can cycle through alternates.
    # refined_regions: list of {"box": [x1,y1,x2,y2], "alpha": np.ndarray}
    # mask_prior: bool HxW mask from SAM 3 that seeds SAM 2 click refinement
    #   (Level 2 chaining). None = pure SAM 2 click path.
    return {
        "name": name, "points": [], "labels": [], "box": None,
        "history": [], "refined_regions": [],
        "candidates": [], "candidate_scores": [], "active_candidate": 0,
        "mask_prior": None, "prior_source": None,
    }


def _segment(layer: dict) -> Optional[np.ndarray]:
    """
    Run SAM 2 on an active layer's prompts, returning the active mask.

    Level 2 chaining: if the layer has a `mask_prior` (populated by SAM 3
    concept segmentation), it's passed to SAM 2 via `mask_input` so click
    prompts refine the SAM 3 mask instead of replacing it.
    """
    prior = layer.get("mask_prior")
    has_pts = bool(layer["points"])
    has_box = layer["box"] is not None

    # No new click prompts: just use the prior as-is (SAM 3 output stands).
    if not has_pts and not has_box:
        if prior is not None:
            layer["candidates"] = [prior]
            layer["candidate_scores"] = [1.0]
            layer["active_candidate"] = 0
            return prior
        layer["candidates"] = []
        layer["candidate_scores"] = []
        layer["active_candidate"] = 0
        return None

    predictor = STATE["predictor"]
    pts = np.asarray(layer["points"], dtype=np.float32) if has_pts else None
    lbls = np.asarray(layer["labels"], dtype=np.int32) if has_pts else None
    box = np.asarray(layer["box"], dtype=np.float32) if has_box else None
    mask_input = bool_mask_to_sam2_prior(prior) if prior is not None else None

    with autocast_ctx(STATE["device_resolved"]):
        masks, scores, _ = predictor.predict(
            point_coords=pts, point_labels=lbls, box=box,
            mask_input=mask_input,
            multimask_output=True,
        )
    order = np.argsort(-np.asarray(scores))
    layer["candidates"] = [np.asarray(masks[i]).astype(bool) for i in order]
    layer["candidate_scores"] = [float(scores[i]) for i in order]
    # Reset to best candidate when prompts change.
    layer["active_candidate"] = 0
    return layer["candidates"][0]


def _render(layers_state: list[dict], active_idx: int) -> Optional[np.ndarray]:
    """Compose preview at full source dims then downscale to PREVIEW_MAX_DIM for speed."""
    rgb = STATE["rgb"]
    if rgb is None:
        return None
    H, W, _ = rgb.shape
    canvas = Image.fromarray(rgb).convert("RGBA")
    # Tint all layers, back→front (so nearer layer tints sit on top).
    for idx, layer in list(enumerate(layers_state))[::-1]:
        mask = STATE["cached_masks"].get(layer["name"])
        if mask is None:
            continue
        color = LAYER_COLORS[idx % len(LAYER_COLORS)]
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        overlay[..., :3] = color
        overlay[..., 3] = mask.astype(np.uint8) * 90
        canvas = Image.alpha_composite(canvas, Image.fromarray(overlay, "RGBA"))
    # Overlay active layer's prompts.
    if 0 <= active_idx < len(layers_state):
        draw = ImageDraw.Draw(canvas)
        r = max(6, min(W, H) // 180)
        layer = layers_state[active_idx]
        if layer["box"] is not None:
            x1, y1, x2, y2 = layer["box"]
            draw.rectangle((x1, y1, x2, y2),
                           outline=(255, 220, 60, 255), width=max(2, r // 2))
        for region in layer.get("refined_regions", []):
            x1, y1, x2, y2 = region["box"]
            # Cyan outline: "this region has been edge-refined"
            draw.rectangle((x1, y1, x2, y2),
                           outline=(80, 220, 220, 255), width=max(2, r // 2))
        if STATE["pending_box_corner"] is not None:
            x, y = STATE["pending_box_corner"]
            draw.ellipse((x - r, y - r, x + r, y + r),
                         fill=(255, 220, 60, 255), outline=(255, 255, 255, 255), width=2)
        sel = STATE["move_selected"]
        sel_idx = sel[1] if sel and sel[0] == layer["name"] else -1
        for i, ((x, y), lbl) in enumerate(zip(layer["points"], layer["labels"])):
            fill = (0, 230, 50, 255) if lbl == 1 else (240, 60, 60, 255)
            if i == sel_idx:
                # "Picked up" highlight — bigger ring in yellow.
                rr = int(r * 1.8)
                draw.ellipse((x - rr, y - rr, x + rr, y + rr),
                             outline=(255, 220, 60, 255), width=max(3, r // 2))
            draw.ellipse((x - r, y - r, x + r, y + r),
                         fill=fill, outline=(255, 255, 255, 255), width=2)
    # Draw the pending zoom corner if we're mid-capture (always in source coords).
    if STATE["pending_zoom_corner"] is not None and 0 <= active_idx < len(layers_state):
        # (We draw the corner indicator regardless of active layer so it's visible.)
        pass  # drawn below, outside the active-layer branch

    if STATE["pending_zoom_corner"] is not None:
        draw2 = ImageDraw.Draw(canvas)
        r = max(6, min(W, H) // 180)
        x, y = STATE["pending_zoom_corner"]
        draw2.ellipse((x - r, y - r, x + r, y + r),
                      fill=(120, 200, 255, 255),
                      outline=(255, 255, 255, 255), width=2)

    # Crop to the zoomed view (if any), then scale to preview dims.
    ox, oy, view_w, view_h, scale = _view_transform()
    STATE["preview_origin"] = (ox, oy)
    STATE["preview_scale"] = scale
    if STATE["zoom_box"] is not None:
        canvas = canvas.crop((ox, oy, ox + view_w, oy + view_h))
    if scale < 1.0:
        pw, ph = int(round(view_w * scale)), int(round(view_h * scale))
        canvas = canvas.resize((pw, ph), Image.BILINEAR)
    return np.asarray(canvas.convert("RGB"))


def _compute_preview_scale(W: int, H: int) -> float:
    longest = max(W, H)
    if longest <= PREVIEW_MAX_DIM:
        return 1.0
    return PREVIEW_MAX_DIM / float(longest)


def _view_transform() -> tuple[int, int, int, int, float]:
    """
    Return (origin_x, origin_y, view_w, view_h, scale) where (ox, oy) is the
    top-left of what's currently shown in preview space (in source coords),
    (view_w, view_h) is the source-space size shown, and scale maps source
    pixels to preview pixels.
    """
    rgb = STATE["rgb"]
    if rgb is None:
        return (0, 0, 0, 0, 1.0)
    H, W, _ = rgb.shape
    if STATE["zoom_box"] is None:
        return (0, 0, W, H, _compute_preview_scale(W, H))
    x1, y1, x2, y2 = [int(round(v)) for v in STATE["zoom_box"]]
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(x1 + 1, min(W, x2))
    y2 = max(y1 + 1, min(H, y2))
    view_w, view_h = x2 - x1, y2 - y1
    longest = max(view_w, view_h)
    scale = PREVIEW_MAX_DIM / float(longest) if longest > PREVIEW_MAX_DIM else 1.0
    return (x1, y1, view_w, view_h, scale)


REFINE_BAND_PX = 12
REFINE_PAD_PX = 8  # pad the crop so matting has room for a trimap band


def _refine_region(layer: dict, box: list[float]) -> Optional[dict]:
    """
    Run matting on a crop of this layer's hard mask inside `box`. Returns a
    dict {box, alpha} to append to layer["refined_regions"], or None if the
    layer has no mask yet or the box is degenerate.
    """
    mask = STATE["cached_masks"].get(layer["name"])
    rgb = STATE["rgb"]
    if mask is None or rgb is None:
        return None
    H, W, _ = rgb.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    # Snap + pad
    x1 = max(0, min(W - 1, x1 - REFINE_PAD_PX))
    y1 = max(0, min(H - 1, y1 - REFINE_PAD_PX))
    x2 = max(0, min(W, x2 + REFINE_PAD_PX))
    y2 = max(0, min(H, y2 + REFINE_PAD_PX))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    rgb_crop = rgb[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]
    # If the crop has no foreground at all, nothing to refine.
    if not mask_crop.any():
        return None
    try:
        from layers import matting_refine
        alpha_crop = matting_refine(rgb_crop, mask_crop, band_px=REFINE_BAND_PX, algo="cf")
    except Exception as e:
        print(f"[refine] matting failed on {layer['name']!r} crop: {e}")
        return None
    return {"box": [x1, y1, x2, y2], "alpha": alpha_crop.astype(np.float32)}


def _find_idx(layers_state: list[dict], name: Optional[str]) -> int:
    if not name:
        return -1
    for i, L in enumerate(layers_state):
        if L["name"] == name:
            return i
    return -1


def _nearest_point_idx(layer: dict, x: float, y: float, threshold: float) -> int:
    best, best_d = -1, threshold * threshold
    for i, (px, py) in enumerate(layer["points"]):
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best_d:
            best, best_d = i, d
    return best


def _recompute(layer: dict) -> Optional[np.ndarray]:
    mask = _segment(layer)
    if mask is None:
        STATE["cached_masks"].pop(layer["name"], None)
    else:
        STATE["cached_masks"][layer["name"]] = mask
    return mask


def _layer_to_df(layer: Optional[dict]) -> list[list]:
    if layer is None:
        return []
    return [[int(x), int(y), "+" if lbl == 1 else "−"]
            for (x, y), lbl in zip(layer["points"], layer["labels"])]


def _df_to_layer_fields(df_rows) -> tuple[list[list[float]], list[int]]:
    """Parse dataframe rows into (points, labels). Tolerant of blanks/invalid."""
    if df_rows is None:
        return [], []
    # Gradio sends either a DataFrame-like with .values or a list of lists.
    rows = df_rows.values.tolist() if hasattr(df_rows, "values") else list(df_rows)
    points: list[list[float]] = []
    labels: list[int] = []
    for row in rows:
        if row is None or len(row) < 3:
            continue
        try:
            x = float(row[0]); y = float(row[1])
        except (TypeError, ValueError):
            continue
        lbl_raw = str(row[2]).strip() if row[2] is not None else "+"
        lbl = 0 if lbl_raw in ("-", "−", "0", "exclude", "ex") else 1
        points.append([x, y])
        labels.append(lbl)
    return points, labels


def _prompt_summary(layer: dict) -> str:
    n_in = sum(1 for l in layer["labels"] if l == 1)
    n_ex = sum(1 for l in layer["labels"] if l == 0)
    parts = []
    if n_in: parts.append(f"{n_in}+")
    if n_ex: parts.append(f"{n_ex}−")
    if layer["box"] is not None: parts.append("1 box")
    return ", ".join(parts) if parts else "(empty)"


# --- handlers ---

def on_upload(image, model, device, _layers_state):
    if image is None:
        return gr.update(), [], gr.update(choices=[], value=None), "No image.", []
    rgb = np.asarray(image.convert("RGB"))
    predictor, device_resolved = _load_predictor(model, device)
    with autocast_ctx(device_resolved):
        predictor.set_image(rgb)
    STATE["rgb"] = rgb
    STATE["cached_masks"] = {}
    STATE["pending_box_corner"] = None
    STATE["move_selected"] = None
    STATE["zoom_box"] = None
    STATE["awaiting_zoom"] = False
    STATE["pending_zoom_corner"] = None
    H, W, _ = rgb.shape
    STATE["preview_scale"] = _compute_preview_scale(W, H)
    return (_render([], -1), [], gr.update(choices=[], value=None),
            f"Loaded {W}×{H} on {device_resolved} ({model}). "
            f"Preview @ {int(W*STATE['preview_scale'])}×{int(H*STATE['preview_scale'])}. "
            f"Add a layer to start.", [])


def on_add_layer(new_name, layers_state):
    new_name = (new_name or "").strip()
    if not new_name:
        return layers_state, gr.update(), gr.update(value=""), "Name can't be empty."
    if any(L["name"] == new_name for L in layers_state):
        return layers_state, gr.update(), gr.update(value=""), f"Layer {new_name!r} exists."
    layers_state = layers_state + [_new_layer(new_name)]
    choices = [L["name"] for L in layers_state]
    return (layers_state, gr.update(choices=choices, value=new_name),
            gr.update(value=""), f"Added layer {new_name!r} (depth #{len(layers_state)}).")


def on_remove_layer(active_layer_name, layers_state):
    layers_state = [L for L in layers_state if L["name"] != active_layer_name]
    STATE["cached_masks"].pop(active_layer_name, None)
    STATE["pending_box_corner"] = None
    choices = [L["name"] for L in layers_state]
    new_active = choices[0] if choices else None
    preview = _render(layers_state, 0 if choices else -1)
    new_layer = next((L for L in layers_state if L["name"] == new_active), None)
    return (layers_state, gr.update(choices=choices, value=new_active),
            preview, f"Removed {active_layer_name!r}.", _layer_to_df(new_layer))


def on_clear_points(active_layer_name, layers_state):
    idx = _find_idx(layers_state, active_layer_name)
    if idx < 0:
        return layers_state, _render(layers_state, -1), "No active layer.", []
    layer = layers_state[idx]
    had_prior = layer.get("mask_prior") is not None
    layer.update(points=[], labels=[], box=None, history=[], refined_regions=[])
    STATE["pending_box_corner"] = None
    # Preserve the SAM 3 prior: "clear" means reset refinements, not drop the
    # concept-segmentation starting point. To drop the prior entirely, remove
    # the layer.
    if had_prior:
        layer["candidates"] = [layer["mask_prior"]]
        layer["candidate_scores"] = [1.0]
        layer["active_candidate"] = 0
        STATE["cached_masks"][active_layer_name] = layer["mask_prior"]
        msg = f"Cleared refinements on {active_layer_name!r} — reset to SAM 3 mask."
    else:
        STATE["cached_masks"].pop(active_layer_name, None)
        msg = f"Cleared {active_layer_name!r}."
    return layers_state, _render(layers_state, idx), msg, []


def on_image_click(evt: gr.SelectData, mode, active_layer_name, layers_state):
    if STATE["rgb"] is None:
        return layers_state, None, "Upload an image first.", [], gr.update()
    # Preview click → source-image coord via current view transform.
    ox, oy, _vw, _vh, scale = _view_transform()
    H, W, _ = STATE["rgb"].shape
    x = ox + int(round(evt.index[0] / scale))
    y = oy + int(round(evt.index[1] / scale))
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))

    # If we're mid-capture for a zoom box, intercept before normal mode dispatch.
    if STATE["awaiting_zoom"]:
        idx_a = _find_idx(layers_state, active_layer_name)
        layer_a = layers_state[idx_a] if idx_a >= 0 else None
        if STATE["pending_zoom_corner"] is None:
            STATE["pending_zoom_corner"] = [x, y]
            return (layers_state, _render(layers_state, idx_a),
                    "Zoom: first corner set. Click the opposite corner.",
                    _layer_to_df(layer_a), gr.update())
        px, py = STATE["pending_zoom_corner"]
        STATE["pending_zoom_corner"] = None
        STATE["awaiting_zoom"] = False
        # Minimum zoom size guard (40 px on a side).
        zx1, zy1 = min(px, x), min(py, y)
        zx2, zy2 = max(px, x), max(py, y)
        if zx2 - zx1 < 40 or zy2 - zy1 < 40:
            STATE["zoom_box"] = None
            return (layers_state, _render(layers_state, idx_a),
                    "Zoom cancelled (box too small, need ≥40 px on each side).",
                    _layer_to_df(layer_a), gr.update(value="Zoom to region"))
        STATE["zoom_box"] = [float(zx1), float(zy1), float(zx2), float(zy2)]
        return (layers_state, _render(layers_state, idx_a),
                f"Zoomed to {zx2-zx1}×{zy2-zy1} px. Click mode = {mode} — clicks now "
                f"land with source-pixel precision. Hit Exit zoom to return.",
                _layer_to_df(layer_a), gr.update(value="Exit zoom"))

    idx = _find_idx(layers_state, active_layer_name)
    if idx < 0:
        return layers_state, _render(layers_state, -1), "Add and select a layer first.", []
    layer = layers_state[idx]

    if mode in ("include", "exclude"):
        lbl = 1 if mode == "include" else 0
        layer["points"].append([x, y])
        layer["labels"].append(lbl)
        layer["history"].append(("point", None))
        _recompute(layer)
        msg = f"{layer['name']}: {_prompt_summary(layer)}"
    elif mode == "box":
        if STATE["pending_box_corner"] is None:
            STATE["pending_box_corner"] = [x, y]
            msg = f"{layer['name']}: set first corner, click opposite corner."
        else:
            px, py = STATE["pending_box_corner"]
            STATE["pending_box_corner"] = None
            prev_box = layer["box"]
            layer["box"] = [float(min(px, x)), float(min(py, y)),
                            float(max(px, x)), float(max(py, y))]
            layer["history"].append(("box", prev_box))
            _recompute(layer)
            msg = f"{layer['name']}: box set — {_prompt_summary(layer)}"
    elif mode == "refine":
        if STATE["pending_box_corner"] is None:
            STATE["pending_box_corner"] = [x, y]
            msg = f"{layer['name']}: refine — click opposite corner of the region to fine-tune."
        else:
            px, py = STATE["pending_box_corner"]
            STATE["pending_box_corner"] = None
            rbox = [float(min(px, x)), float(min(py, y)),
                    float(max(px, x)), float(max(py, y))]
            region = _refine_region(layer, rbox)
            if region is None:
                msg = (f"{layer['name']}: refine skipped (segment this layer with "
                       "points/box first, or draw a larger refine box).")
            else:
                layer.setdefault("refined_regions", []).append(region)
                layer["history"].append(("refine", None))
                cy, cx = region["alpha"].shape
                msg = (f"{layer['name']}: refined {cx}×{cy} px region with "
                       f"band={REFINE_BAND_PX}px. Cyan outline in preview.")
    elif mode == "move":
        thresh = max(W, H) / 40.0
        sel = STATE["move_selected"]
        if sel is None or sel[0] != layer["name"]:
            # First click: pick up nearest point.
            ni = _nearest_point_idx(layer, x, y, thresh)
            if ni >= 0:
                STATE["move_selected"] = (layer["name"], ni)
                msg = f"{layer['name']}: picked point #{ni}. Click to drop."
            else:
                msg = "Move: no point within reach to pick up."
        else:
            # Second click: drop the selected point here.
            _, ni = sel
            if 0 <= ni < len(layer["points"]):
                layer["points"][ni] = [x, y]
                _recompute(layer)
            STATE["move_selected"] = None
            msg = f"{layer['name']}: moved — {_prompt_summary(layer)}"
    elif mode == "erase":
        thresh = max(W, H) / 60.0
        ni = _nearest_point_idx(layer, x, y, thresh)
        if ni >= 0:
            layer["points"].pop(ni)
            layer["labels"].pop(ni)
            layer["history"].append(("point-removed", ni))
            _recompute(layer)
            msg = f"{layer['name']}: point removed — {_prompt_summary(layer)}"
        elif layer["box"] is not None and _inside_box(layer["box"], x, y):
            layer["history"].append(("box-removed", layer["box"]))
            layer["box"] = None
            _recompute(layer)
            msg = f"{layer['name']}: box removed — {_prompt_summary(layer)}"
        else:
            msg = "Erase: no nearby point or box."
    else:
        msg = f"Unknown mode {mode!r}"
    return layers_state, _render(layers_state, idx), msg, _layer_to_df(layer), gr.update()


def _inside_box(box: list[float], x: float, y: float) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def on_undo(active_layer_name, layers_state):
    idx = _find_idx(layers_state, active_layer_name)
    if idx < 0:
        return layers_state, _render(layers_state, -1), "No active layer.", []
    layer = layers_state[idx]
    if not layer["history"]:
        return layers_state, _render(layers_state, idx), "Nothing to undo.", _layer_to_df(layer)
    kind, data = layer["history"].pop()
    if kind == "point":
        layer["points"].pop()
        layer["labels"].pop()
    elif kind == "box":
        layer["box"] = data
    elif kind == "box-removed":
        layer["box"] = data
    elif kind == "refine":
        if layer.get("refined_regions"):
            layer["refined_regions"].pop()
    STATE["pending_box_corner"] = None
    _recompute(layer)
    return layers_state, _render(layers_state, idx), f"Undo — {_prompt_summary(layer)}", _layer_to_df(layer)


def on_active_layer_change(active_layer_name, layers_state):
    STATE["pending_box_corner"] = None
    STATE["move_selected"] = None
    idx = _find_idx(layers_state, active_layer_name)
    layer = layers_state[idx] if idx >= 0 else None
    return _render(layers_state, idx), _layer_to_df(layer)


def on_mode_change(_mode, active_layer_name, layers_state):
    """Clear pending interactions so a mode switch doesn't dangle state."""
    STATE["pending_box_corner"] = None
    STATE["move_selected"] = None
    idx = _find_idx(layers_state, active_layer_name)
    return _render(layers_state, idx)


def _ensure_sam3_pipeline(device_resolved: str):
    if STATE["sam3_pipe"] is None or STATE["sam3_device"] != device_resolved:
        model, processor = load_sam3_concept_pipeline(device_resolved)
        STATE["sam3_pipe"] = (model, processor)
        STATE["sam3_device"] = device_resolved
    return STATE["sam3_pipe"]


def on_concept_segment(concept_text, device, base_name, score_thresh, layers_state):
    """
    Run SAM 3 concept segmentation. Creates one new layer per returned instance
    named {base_name}-{i} (i=1..N). Adds them to layers_state with a pre-set mask
    (stored into STATE["cached_masks"]).
    """
    if STATE["rgb"] is None:
        return layers_state, gr.update(), None, "Upload an image first."
    text = (concept_text or "").strip()
    if not text:
        return layers_state, gr.update(), None, "Enter a concept like 'rope' or 'face'."
    device_resolved = detect_device(device)
    try:
        model, processor = _ensure_sam3_pipeline(device_resolved)
    except (ImportError, RuntimeError) as e:
        return (layers_state, gr.update(), None,
                f"**SAM 3 unavailable.** {e}")

    from PIL import Image as PILImage
    img = PILImage.fromarray(STATE["rgb"])
    try:
        results = segment_by_concept(model, processor, img, text,
                                     score_threshold=float(score_thresh))
    except Exception as e:
        return (layers_state, gr.update(), None,
                f"SAM 3 segmentation failed: {type(e).__name__}: {e}")

    if not results:
        return (layers_state, gr.update(), None,
                f"No instances of {text!r} found at score ≥ {score_thresh}. Try a different phrase or lower threshold.")

    base = (base_name or text).strip().replace(" ", "-") or "concept"
    # Avoid colliding with existing names.
    existing = {L["name"] for L in layers_state}
    created = []
    for i, r in enumerate(results, start=1):
        name = f"{base}-{i}"
        n = 1
        while name in existing:
            n += 1
            name = f"{base}-{i}-{n}"
        existing.add(name)
        layer = _new_layer(name)
        # Store SAM 3's mask as both the initial candidate and the prior so
        # future SAM 2 click prompts refine it rather than replacing it.
        layer["candidates"] = [r["mask"]]
        layer["candidate_scores"] = [r["score"]]
        layer["mask_prior"] = r["mask"]
        layer["prior_source"] = "sam3"
        layers_state = layers_state + [layer]
        STATE["cached_masks"][name] = r["mask"]
        created.append((name, r["score"]))

    choices = [L["name"] for L in layers_state]
    first_name = created[0][0]
    idx = _find_idx(layers_state, first_name)
    summary = ", ".join(f"{n} ({s:.2f})" for n, s in created)
    return (layers_state,
            gr.update(choices=choices, value=first_name),
            _render(layers_state, idx),
            f"SAM 3 created {len(created)} layer(s) for {text!r}: {summary}")


def on_cycle_mask(direction, active_layer_name, layers_state):
    """direction=+1 or -1. Cycles the active layer's SAM candidate."""
    idx = _find_idx(layers_state, active_layer_name)
    if idx < 0:
        return layers_state, _render(layers_state, -1), "No active layer.", []
    layer = layers_state[idx]
    cands = layer.get("candidates", [])
    if not cands:
        return (layers_state, _render(layers_state, idx),
                "No candidates yet — add a point or box on this layer first.",
                _layer_to_df(layer))
    n = len(cands)
    layer["active_candidate"] = (layer["active_candidate"] + direction) % n
    STATE["cached_masks"][layer["name"]] = cands[layer["active_candidate"]]
    i = layer["active_candidate"]
    score = layer["candidate_scores"][i]
    names = ["best", "alt 1", "alt 2"]
    label = names[i] if i < len(names) else f"alt {i}"
    return (layers_state, _render(layers_state, idx),
            f"{layer['name']}: candidate {i+1}/{n} ({label}, score {score:.2f}). "
            f"SAM returns subpart/part/whole; cycle to pick which you meant.",
            _layer_to_df(layer))


def on_zoom_button(active_layer_name, layers_state):
    """Toggle: start zoom-box capture, or exit an active zoom."""
    idx = _find_idx(layers_state, active_layer_name)
    if STATE["zoom_box"] is not None:
        STATE["zoom_box"] = None
        STATE["awaiting_zoom"] = False
        STATE["pending_zoom_corner"] = None
        return (_render(layers_state, idx),
                gr.update(value="Zoom to region"),
                "Zoomed out.")
    STATE["awaiting_zoom"] = True
    STATE["pending_zoom_corner"] = None
    return (_render(layers_state, idx),
            gr.update(value="Cancel zoom"),
            "Zoom: click two opposite corners on the canvas to define the zoomed region.")


def on_points_df_change(df_rows, active_layer_name, layers_state):
    """User edited the points dataframe: mutate the active layer and re-segment."""
    idx = _find_idx(layers_state, active_layer_name)
    if idx < 0:
        return layers_state, _render(layers_state, -1), "No active layer."
    layer = layers_state[idx]
    points, labels = _df_to_layer_fields(df_rows)
    # Skip no-op changes to avoid feedback loops.
    if points == layer["points"] and labels == layer["labels"]:
        return layers_state, _render(layers_state, idx), gr.update()
    layer["points"] = points
    layer["labels"] = labels
    _recompute(layer)
    return layers_state, _render(layers_state, idx), f"{layer['name']}: {_prompt_summary(layer)}"


def on_save(out_dir, edges, feather, matting_band, matting_algo, infill, include_bg, include_css, layers_state):
    if STATE["rgb"] is None:
        return "No image loaded.", "", None
    if not layers_state:
        return "No layers to save.", "", None
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rgb = STATE["rgb"]
    H, W, _ = rgb.shape

    claimed = np.zeros((H, W), dtype=bool)
    exclusive = []
    for L in layers_state:
        m = STATE["cached_masks"].get(L["name"])
        if m is None:
            continue
        m_excl = m & ~claimed
        exclusive.append((L["name"], m_excl))
        claimed |= m

    # Index layers by name to look up refined regions.
    layer_by_name = {L["name"]: L for L in layers_state}

    finals = []
    for name, m_excl in exclusive:
        if edges == "matting":
            try:
                alpha = matting_refine(rgb, m_excl, band_px=int(matting_band), algo=matting_algo)
            except ImportError:
                alpha = m_excl
        else:
            alpha = m_excl
        # Apply per-region refinements (local matting overrides).
        regions = layer_by_name.get(name, {}).get("refined_regions", [])
        if regions:
            alpha_f = _coerce_alpha(alpha).copy()
            for region in regions:
                x1, y1, x2, y2 = [int(v) for v in region["box"]]
                refined = region["alpha"]
                # Clip the refined alpha to this layer's exclusive territory so
                # a refine box can't steal pixels from a nearer layer.
                excl_crop = m_excl[y1:y2, x1:x2].astype(np.float32)
                alpha_f[y1:y2, x1:x2] = refined * excl_crop
            alpha = alpha_f
        finals.append((name, alpha))

    written = []
    for name, a in finals:
        fe = int(feather) if edges == "feather" else 0
        rgba = mask_to_rgba(rgb, a, fe)
        p = out / f"{name}.png"
        rgba.save(p, optimize=True)
        written.append(p.name)

    if include_bg:
        union = np.zeros((H, W), dtype=np.float32)
        for _n, a in finals:
            union = 1.0 - (1.0 - union) * (1.0 - _coerce_alpha(a))
        if infill == "none":
            bg_alpha = 1.0 - union
            fe = int(feather) if edges == "feather" else 0
            rgba = mask_to_rgba(rgb, bg_alpha, fe)
        else:
            hole = union > 0.5
            filled_rgb = infill_background(rgb, hole, method=infill)
            rgba = mask_to_rgba(filled_rgb, np.ones((H, W), dtype=np.float32), 0)
        p = out / "bg.png"
        rgba.save(p, optimize=True)
        written.append(p.name)

    (out / "points.json").write_text(json.dumps(
        [{"name": L["name"], "points": L["points"], "labels": L["labels"], "box": L["box"]}
         for L in layers_state],
        indent=2,
    ))

    css = ""
    if include_css:
        css = build_css_snippet([name for name, _ in finals], include_bg=include_bg)
        (out / "snippet.html").write_text(css)
        written.append("snippet.html")

    # Build inline output composite: stack bg + layers in depth order
    # (back-to-front so nearest layer lands on top).
    composite = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if include_bg and (out / "bg.png").exists():
        composite = Image.alpha_composite(
            composite, Image.open(out / "bg.png").convert("RGBA"),
        )
    for name, _ in reversed(finals):
        p = out / f"{name}.png"
        if p.exists():
            composite = Image.alpha_composite(composite, Image.open(p).convert("RGBA"))
    composite_np = np.asarray(composite.convert("RGB"))

    status = (f"Saved {len(written)} file(s) at {W}×{H} → `{out}`\n"
              f"Files: {', '.join(written)}\nSpec: points.json\nEdges: {edges}")
    return status, css, composite_np


def on_export_spec(layers_state):
    return json.dumps(
        [{"name": L["name"], "points": L["points"], "labels": L["labels"], "box": L["box"]}
         for L in layers_state],
        indent=2,
    )


# --- UI ---

def build_ui(args):
    with gr.Blocks(title="layercake — SAM 2 interactive") as demo:
        gr.Markdown(
            "# layercake\n"
            "**Photo → depth-ordered transparent PNG layers for CSS compositing.** "
            "Click-prompt with SAM 2, or describe a concept in text with SAM 3."
        )
        gr.HTML(
            """
            <div style="display:flex; gap:12px; margin:8px 0 16px 0; flex-wrap:wrap;">
              <div style="flex:1 1 0; min-width:120px; padding:10px 14px; border-radius:10px;
                          background:var(--panel-background-fill, rgba(127,127,127,0.08));
                          border:1px solid var(--border-color-primary, rgba(127,127,127,0.25));">
                <div style="font-size:11px; opacity:0.6; letter-spacing:0.04em;">STEP 1</div>
                <div style="font-weight:600;">Upload</div>
              </div>
              <div style="flex:1 1 0; min-width:120px; padding:10px 14px; border-radius:10px;
                          background:var(--panel-background-fill, rgba(127,127,127,0.08));
                          border:1px solid var(--border-color-primary, rgba(127,127,127,0.25));">
                <div style="font-size:11px; opacity:0.6; letter-spacing:0.04em;">STEP 2</div>
                <div style="font-weight:600;">Name a layer</div>
              </div>
              <div style="flex:1 1 0; min-width:120px; padding:10px 14px; border-radius:10px;
                          background:var(--panel-background-fill, rgba(127,127,127,0.08));
                          border:1px solid var(--border-color-primary, rgba(127,127,127,0.25));">
                <div style="font-size:11px; opacity:0.6; letter-spacing:0.04em;">STEP 3</div>
                <div style="font-weight:600;">Click points / draw box</div>
              </div>
              <div style="flex:1 1 0; min-width:120px; padding:10px 14px; border-radius:10px;
                          background:var(--panel-background-fill, rgba(127,127,127,0.08));
                          border:1px solid var(--border-color-primary, rgba(127,127,127,0.25));">
                <div style="font-size:11px; opacity:0.6; letter-spacing:0.04em;">STEP 4</div>
                <div style="font-weight:600;">Save</div>
              </div>
            </div>
            <div style="font-size:13px; opacity:0.7; margin-bottom:16px;">
              Tip — for complex scenes, many small named layers beat cramming many points into one.
              Use Erase to remove the nearest point/box.
            </div>
            """
        )
        layers_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Source image", height=220)

                with gr.Accordion("Segment by concept (SAM 3, text prompt)", open=False):
                    gr.Markdown(
                        "Type a short noun phrase — `rope`, `face`, `hand` — and SAM 3 "
                        "segments **all instances** of it, creating one layer per instance. "
                        "Gated model: requires `HF_TOKEN` + one-time access approval at "
                        "[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)."
                    )
                    with gr.Row():
                        concept_text = gr.Textbox(label="Concept", placeholder="rope", scale=2)
                        concept_base = gr.Textbox(label="Layer name prefix", placeholder="(defaults to concept)", scale=2)
                    concept_thresh = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                               label="Score threshold (higher = fewer, higher-confidence instances)")
                    concept_btn = gr.Button("Segment by concept", variant="primary", size="sm")

                gr.Markdown("### Layers")
                with gr.Row():
                    new_name = gr.Textbox(label="New layer name", placeholder="subject", scale=2)
                    add_btn = gr.Button("Add layer", scale=1)
                active_layer = gr.Dropdown(choices=[], label="Active layer")
                point_mode = gr.Radio(
                    ["include", "exclude", "move", "box", "refine", "erase"],
                    value="include", label="Click mode",
                    info="include/exclude: add point. move: 1st click picks, 2nd drops. "
                         "box: two clicks = corners. refine: two clicks = region to fine-tune "
                         "the edge via local matting. erase: remove nearest point/box.",
                )
                with gr.Row():
                    undo_btn = gr.Button("Undo last")
                    clear_btn = gr.Button("Clear layer")
                    remove_btn = gr.Button("Remove layer", variant="stop")
                with gr.Row():
                    mask_prev_btn = gr.Button("◀ Try alt mask", size="sm")
                    mask_next_btn = gr.Button("Try alt mask ▶", size="sm")

                points_df = gr.Dataframe(
                    headers=["x", "y", "±"],
                    datatype=["number", "number", "str"],
                    row_count=(0, "dynamic"),
                    column_count=(3, "fixed"),
                    label="Points (edit x/y to move, delete row to remove)",
                    interactive=True,
                )

                gr.Markdown("### Save")
                out_dir = gr.Textbox(value=str(Path.cwd() / "out"), label="Output dir")
                edges = gr.Radio(
                    ["feather", "matting"], value="feather", label="Edge quality",
                    info="feather = fast. matting = closed-form; best on hair/fine edges.",
                )
                feather = gr.Slider(0, 10, value=2, step=1, label="Feather (px)")
                with gr.Row():
                    include_bg = gr.Checkbox(value=True, label="Write bg.png")
                    include_css = gr.Checkbox(value=True, label="Write snippet.html")
                save_btn = gr.Button("Save layers", variant="primary", size="lg")

                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        model = gr.Dropdown(
                            ["facebook/sam3",
                             "sam2-hiera-large", "sam2-hiera-base-plus",
                             "sam2-hiera-small", "sam2-hiera-tiny"],
                            value=args.model, label="Model",
                        )
                        device = gr.Dropdown(["auto", "cpu", "cuda", "mps"],
                                             value=args.device, label="Device")
                    matting_band = gr.Slider(2, 24, value=8, step=1, label="Matting band (px)",
                                             info="unknown-region width for --edges matting")
                    matting_algo = gr.Dropdown(
                        ["cf", "lbdm", "knn"], value="cf", label="Matting solver",
                        info="cf = fastest + highest quality on typical problems. "
                             "lbdm/knn = for very large unknown regions.",
                    )
                    infill = gr.Dropdown(
                        ["none", "opencv", "lama"], value="none", label="Background infill",
                        info="none = transparent. opencv = Navier-Stokes (instant, simple bgs). "
                             "lama = LaMa model (plausible on complex, ~200MB first use).",
                    )

                export_btn = gr.Button("Show points.json")
                spec_out = gr.Code(label="points.json", language="json")
                css_out = gr.Code(label="CSS snippet", language="html")

            with gr.Column(scale=3):
                preview = gr.Image(
                    label="Click to add points / draw box for the active layer",
                    type="numpy",
                    format="png",
                    interactive=False,
                    elem_classes=["big-preview"],
                )
                with gr.Row():
                    zoom_btn = gr.Button("Zoom to region", size="sm")
                status = gr.Markdown("Upload an image to begin.")
                output_composite = gr.Image(
                    label="Output composite (how the CSS stack will render — updates on Save)",
                    type="numpy",
                    format="png",
                    interactive=False,
                )

        image_in.change(on_upload, [image_in, model, device, layers_state],
                        [preview, layers_state, active_layer, status, points_df])
        add_btn.click(on_add_layer, [new_name, layers_state],
                      [layers_state, active_layer, new_name, status])
        concept_btn.click(on_concept_segment,
                          [concept_text, device, concept_base, concept_thresh, layers_state],
                          [layers_state, active_layer, preview, status])
        remove_btn.click(on_remove_layer, [active_layer, layers_state],
                        [layers_state, active_layer, preview, status, points_df])
        clear_btn.click(on_clear_points, [active_layer, layers_state],
                        [layers_state, preview, status, points_df])
        undo_btn.click(on_undo, [active_layer, layers_state],
                       [layers_state, preview, status, points_df])
        mask_prev_btn.click(lambda a, s: on_cycle_mask(-1, a, s),
                            [active_layer, layers_state],
                            [layers_state, preview, status, points_df])
        mask_next_btn.click(lambda a, s: on_cycle_mask(+1, a, s),
                            [active_layer, layers_state],
                            [layers_state, preview, status, points_df])
        active_layer.change(on_active_layer_change, [active_layer, layers_state],
                            [preview, points_df])
        point_mode.change(on_mode_change, [point_mode, active_layer, layers_state], [preview])
        preview.select(on_image_click, [point_mode, active_layer, layers_state],
                       [layers_state, preview, status, points_df, zoom_btn])
        zoom_btn.click(on_zoom_button, [active_layer, layers_state],
                       [preview, zoom_btn, status])
        points_df.input(on_points_df_change, [points_df, active_layer, layers_state],
                        [layers_state, preview, status])
        save_btn.click(on_save,
                       [out_dir, edges, feather, matting_band, matting_algo, infill,
                        include_bg, include_css, layers_state],
                       [status, css_out, output_composite])
        export_btn.click(on_export_spec, [layers_state], [spec_out])

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--model", default="sam2-hiera-large")  # or "facebook/sam3"
    args = ap.parse_args()
    demo = build_ui(args)
    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
