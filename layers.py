#!/usr/bin/env python3
"""
layercake — cut a source photo into depth-ordered transparent PNG layers using SAM 2.

Every output PNG has the *source image's exact dimensions* so the layers stack cleanly
under CSS `object-fit: cover; object-position: center`. That's the whole point.

Example:
    python layers.py input.jpg --layers '[
      {"name": "foreground", "points": [[1200, 300], [850, 250]]},
      {"name": "subject",    "points": [[420, 400]]},
      {"name": "midground",  "points": [[1100, 800], [950, 950]]}
    ]' --out layers/

Depth order is the list order: the FIRST layer is treated as the nearest (rendered on
top), the LAST layer as the farthest before the inverse-union background. Earlier
layers occlude later layers in their overlap regions (because they're "in front").
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class LayerSpec:
    name: str
    points: list[list[float]]
    labels: Optional[list[int]] = None   # per-point label: 1=include, 0=exclude
    box: Optional[list[float]] = None    # [x1, y1, x2, y2] axis-aligned

    @classmethod
    def from_dict(cls, d: dict) -> "LayerSpec":
        pts = d.get("points") or []
        box = d.get("box")
        if not pts and not box:
            raise ValueError(f"layer {d.get('name')!r}: must have points or box")
        labels = d.get("labels")
        if labels is not None and len(labels) != len(pts):
            raise ValueError(
                f"layer {d.get('name')!r}: labels length {len(labels)} "
                f"must match points length {len(pts)}"
            )
        if box is not None and len(box) != 4:
            raise ValueError(f"layer {d.get('name')!r}: box must be [x1,y1,x2,y2]")
        return cls(name=d["name"], points=pts, labels=labels, box=box)


def detect_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_ctx(device: str):
    """
    fp16 autocast on MPS/CUDA ~halves SAM 2 encoder + predict latency with no
    visible mask-quality cost. No-op on CPU (fp16 CPU kernels are marginal).
    """
    import torch
    from contextlib import nullcontext

    if device in ("mps", "cuda"):
        try:
            return torch.autocast(device_type=device, dtype=torch.float16)
        except (RuntimeError, TypeError):
            return nullcontext()
    return nullcontext()


def bool_mask_to_sam2_prior(mask: np.ndarray, strength: float = 5.0) -> np.ndarray:
    """
    Convert a full-resolution bool mask into SAM 2's mask_input format —
    a (1, 256, 256) float32 array in logit space. Used to seed SAM 2's
    `predict(mask_input=...)` with a prior from SAM 3 (Level 2 chaining):
    SAM 3's concept mask becomes a warm start that subsequent SAM 2 click
    prompts refine instead of replacing.

    `strength` controls how heavily the prior biases SAM 2. 5.0 is moderate
    (clicks can still meaningfully modify the mask); 10.0+ locks the prior;
    2-3 is a softer suggestion.
    """
    pil = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    resized = np.asarray(pil.resize((256, 256), Image.BILINEAR)).astype(np.float32) / 255.0
    logits = np.where(resized > 0.5, strength, -strength).astype(np.float32)
    return logits[None, ...]  # shape (1, 256, 256)

class Sam3TrackerPredictor:
    """
    Adapter exposing the SAM2ImagePredictor surface (.set_image / .predict /
    .device) on top of transformers' Sam3TrackerModel. Lets layercake swap
    SAM 2 -> SAM 3 (Tracker / PVS) with zero changes to segment_layer or the app.

    Supported predict() kwargs mirror what layercake actually uses:
      point_coords (N,2), point_labels (N,), box (4,), mask_input (1,256,256),
      multimask_output (bool), return_logits (bool).
    Returns (masks, scores, low_res_logits-or-None) like SAM 2.
    """

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self._image = None              # PIL image for the current frame
        self._orig_size = None          # (H, W)
        self._image_embeddings = None   # cached encoder output

    def set_image(self, rgb: np.ndarray):
        import torch
        self._image = Image.fromarray(rgb, mode="RGB")
        self._orig_size = (rgb.shape[0], rgb.shape[1])
        # Pre-compute vision features once per image (the expensive step).
        inputs = self.processor(images=self._image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self._image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])


    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ):
        import torch
        if self._image is None:
            raise RuntimeError("set_image() must be called before predict().")

        # SAM2 layout: point_coords (N,2), point_labels (N,)
        # SAM3Tracker layout: input_points  (img, obj, pts_per_obj, 2)
        #                     input_labels  (img, obj, pts_per_obj)
        input_points = input_labels = input_boxes = None
        if point_coords is not None and len(point_coords) > 0:
            input_points = [[[[float(x), float(y)] for x, y in point_coords]]]
            if point_labels is None:
                point_labels = np.ones(len(point_coords), dtype=np.int32)
            input_labels = [[[int(v) for v in point_labels]]]
        if box is not None:
            x1, y1, x2, y2 = (float(v) for v in box)
            input_boxes = [[[x1, y1, x2, y2]]]

        proc_kwargs = dict(return_tensors="pt", original_sizes=[list(self._orig_size)])
        if input_points is not None:
            proc_kwargs.update(input_points=input_points, input_labels=input_labels)
        if input_boxes is not None:
            proc_kwargs["input_boxes"] = input_boxes
        inputs = self.processor(**proc_kwargs).to(self.device)

        # logit-space mask prior (1,256,256) -> tensor; reuse SAM3's prompt encoder
        input_masks = None
        if mask_input is not None:
            input_masks = torch.as_tensor(mask_input, dtype=torch.float32, device=self.device)

        fwd = {k: v for k, v in inputs.items() if k != "pixel_values"}
        with torch.no_grad():
            outputs = self.model(
                **fwd,
                input_masks=input_masks,
                image_embeddings=self._image_embeddings,
                multimask_output=multimask_output,
            )

        # pred_masks: (batch, point_batch, num_masks, h, w). Upscale to source dims.
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            binarize=not return_logits,
        )[0]  # -> (num_masks, H, W) for our single image/object

        # Collapse leading object dim if present, to match SAM2's (N, H, W).
        masks_np = masks.numpy() if hasattr(masks, "numpy") else np.asarray(masks)
        if masks_np.ndim == 4:        # (obj, num_masks, H, W) -> (num_masks, H, W)
            masks_np = masks_np[0]
        scores = outputs.iou_scores.squeeze().detach().cpu().numpy().reshape(-1)

        if return_logits:
            return masks_np.astype(np.float32), scores, None
        return masks_np.astype(bool), scores, None

def load_sam3_tracker_predictor(model: str, device: str):
    """Load the SAM 3 Tracker (PVS) predictor behind the SAM2ImagePredictor API."""
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    hf_id = model if "/" in model else f"facebook/{model}"
    try:
        m = Sam3TrackerModel.from_pretrained(hf_id, token=token).to(device)
        proc = Sam3TrackerProcessor.from_pretrained(hf_id, token=token)
    except Exception as e:
        msg = str(e)
        if any(s in msg.lower() for s in ("gated", "401", "403", "access")):
            raise RuntimeError(f"{msg}\n\n{SAM3_AUTH_HINT}") from e
        raise
    return Sam3TrackerPredictor(m, proc, device)

def load_sam2_predictor(model: str, device: str):
    """Load an image predictor. Routes sam3* ids to the SAM 3 Tracker adapter,
    otherwise loads SAM 2. Name kept for backward compat with the app."""
    if "sam3" in model.lower():
        return load_sam3_tracker_predictor(model, device)
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    hf_id = model if "/" in model else f"facebook/{model}"
    return SAM2ImagePredictor.from_pretrained(hf_id, device=device)

# --- SAM 3 concept-segmentation path ---------------------------------------
#
# SAM 3 (Nov 2025) introduced Promptable Concept Segmentation: given a short
# text phrase like "rope" or "face", return masks + boxes + scores for every
# instance of that concept in the image. This is exactly the capability SAM 2
# can't give you. We use the HuggingFace transformers integration (avoids the
# `triton` dep that the official sam3 package needs and macOS doesn't ship).
#
# `facebook/sam3` on HuggingFace is a gated model — users must accept terms
# on the model page and have HF_TOKEN set in their environment. We surface a
# clear error if auth is missing rather than letting transformers error deep
# in its innards.

SAM3_AUTH_HINT = (
    "SAM 3 weights are gated. One-time setup:\n"
    "  1. Visit https://huggingface.co/facebook/sam3 and click 'Agree and access'.\n"
    "  2. Create a token at https://huggingface.co/settings/tokens (read scope).\n"
    "  3. Export HF_TOKEN=hf_xxx  (or run `huggingface-cli login`).\n"
    "Then relaunch."
)


def load_sam3_concept_pipeline(device: str, model_id: str = "facebook/sam3"):
    """
    Load the HuggingFace SAM 3 text-prompted concept segmenter.

    Returns (model, processor) ready to run concept segmentation. Raises
    ImportError if transformers isn't new enough to have SAM 3, or RuntimeError
    with a clear message if HF access isn't granted.
    """
    try:
        from transformers import Sam3Model, Sam3Processor  # type: ignore
    except ImportError as e:
        raise ImportError(
            "transformers is missing SAM 3 support. Upgrade: "
            "`uv pip install -U transformers`"
        ) from e

    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    try:
        processor = Sam3Processor.from_pretrained(model_id, token=token)
        model = Sam3Model.from_pretrained(model_id, token=token)
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "401" in msg or "403" in msg or "access" in msg.lower():
            raise RuntimeError(f"{msg}\n\n{SAM3_AUTH_HINT}") from e
        raise

    import torch
    model.to(device)
    model.eval()
    return model, processor


def segment_by_concept(
    model,
    processor,
    image: Image.Image,
    text: str,
    score_threshold: float = 0.4,
    max_instances: int = 10,
) -> list[dict]:
    """
    Run SAM 3 concept segmentation: find every instance of `text` in `image`.

    Returns a list of {"mask": bool HxW, "box": [x1,y1,x2,y2], "score": float},
    sorted best-first and filtered by score_threshold + max_instances.
    """
    import torch

    rgb = image.convert("RGB")
    inputs = processor(images=rgb, text=text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode(), autocast_ctx(str(device).split(":")[0]):
        outputs = model(**inputs)

    # Post-process to per-instance mask+box+score at source dims.
    # The processor exposes post_process_instance_segmentation.
    W, H = rgb.size
    orig_sizes = [(H, W)]
    try:
        results = processor.post_process_instance_segmentation(
            outputs, target_sizes=orig_sizes, score_thresh=score_threshold,
        )[0]
    except Exception:
        # Fallback: read raw outputs and threshold ourselves.
        import torch.nn.functional as F
        pred_masks = outputs.pred_masks[0]  # (N, H', W')
        pred_boxes = outputs.pred_boxes[0] if hasattr(outputs, "pred_boxes") else None
        pred_scores = outputs.pred_scores[0] if hasattr(outputs, "pred_scores") else None
        if pred_scores is None:
            raise
        # Keep top-k above threshold
        keep = pred_scores >= score_threshold
        pred_masks = pred_masks[keep]
        pred_scores = pred_scores[keep]
        # Resize masks to source dims
        m = F.interpolate(pred_masks.float().unsqueeze(0), size=(H, W),
                          mode="bilinear", align_corners=False).squeeze(0)
        results = {
            "masks": (m > 0).cpu().numpy(),
            "scores": pred_scores.cpu().numpy(),
            "boxes": pred_boxes[keep].cpu().numpy() if pred_boxes is not None else None,
        }

    out = []
    masks = np.asarray(results["masks"]) if not isinstance(results["masks"], np.ndarray) else results["masks"]
    scores = np.asarray(results.get("scores", []))
    boxes = results.get("boxes")
    if boxes is not None and not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes)
    order = np.argsort(-scores) if scores.size else range(len(masks))
    for rank, i in enumerate(order):
        if rank >= max_instances:
            break
        out.append({
            "mask": np.asarray(masks[i]).astype(bool),
            "box": boxes[i].tolist() if boxes is not None else None,
            "score": float(scores[i]) if scores.size else 1.0,
        })
    return out


def segment_layer(
    predictor,
    layer: LayerSpec,
    img_wh: tuple[int, int],
    return_soft: bool = False,
) -> np.ndarray:
    """
    Return a HxW mask at source image dimensions. Honors the layer's points,
    labels, and optional axis-aligned box prompt.

    If return_soft is False (default), returns a bool mask.
    If return_soft is True, returns a float32 array in [0, 1] from SAM 2's
    sigmoid(logits) — use this for slightly softer edges out of the gate.
    """
    W, H = img_wh
    pts_list = layer.points or []
    has_pts = len(pts_list) > 0
    pts = np.asarray(pts_list, dtype=np.float32) if has_pts else None
    if has_pts:
        if layer.labels is None:
            lbls = np.ones(len(pts_list), dtype=np.int32)
        else:
            lbls = np.asarray(layer.labels, dtype=np.int32)
    else:
        lbls = None
    box = np.asarray(layer.box, dtype=np.float32) if layer.box is not None else None

    device = getattr(predictor, "device", None)
    device_str = str(device).split(":")[0] if device else "cpu"
    with autocast_ctx(device_str):
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=lbls,
            box=box,
            multimask_output=True,
            return_logits=return_soft,
        )
    best = int(np.argmax(scores))
    m = np.asarray(masks[best])
    if m.shape != (H, W):
        raise RuntimeError(
            f"predicter returned mask. "
            "Check that the predictor was set on the correct image."
        )
    if return_soft:
        return (1.0 / (1.0 + np.exp(-m.astype(np.float32)))).astype(np.float32)
    return m.astype(bool)


def feather_alpha(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Feather a uint8 alpha channel by `radius` pixels via PIL GaussianBlur."""
    if radius <= 0:
        return alpha
    im = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(im)


def matting_refine(
    rgb: np.ndarray,
    hard_mask: np.ndarray,
    band_px: int = 8,
    algo: str = "cf",
) -> np.ndarray:
    """
    Refine a hard bool mask into a soft float32 alpha via alpha matting.

    Builds a trimap: known-fg = erode(mask, band), known-bg = dilate(mask, band)^c,
    unknown = the band between them. Solves soft alpha in the unknown band so the
    boundary respects image edges (hair, thin filaments, fur).

    algo:
      - "cf"   (default): Closed-form matting — near-gold-standard quality,
                fastest in practice on small-to-moderate unknown regions.
      - "lbdm": Learning-Based Digital — approximation designed for large
                unknown regions; has heavier setup so often slower on typical
                problem sizes. Try when cf struggles on huge complex edges.
      - "knn":  KNN matting — different kernel, sometimes useful on textured fur.

    Requires `pymatting` and `scipy`. Raises ImportError if unavailable.
    """
    import pymatting  # type: ignore
    from scipy.ndimage import binary_erosion, binary_dilation  # type: ignore

    solvers = {
        "lbdm": pymatting.estimate_alpha_lbdm,
        "cf": pymatting.estimate_alpha_cf,
        "knn": pymatting.estimate_alpha_knn,
    }
    solve = solvers.get(algo)
    if solve is None:
        raise ValueError(f"unknown matting algo {algo!r}; choose {sorted(solvers)}")

    fg = binary_erosion(hard_mask, iterations=band_px)
    bg_not = binary_dilation(hard_mask, iterations=band_px)
    trimap = np.full(hard_mask.shape, 0.5, dtype=np.float64)
    trimap[fg] = 1.0
    trimap[~bg_not] = 0.0
    img_f = rgb.astype(np.float64) / 255.0
    if img_f.ndim == 3 and img_f.shape[2] == 4:
        img_f = img_f[..., :3]
    alpha = solve(np.ascontiguousarray(img_f), np.ascontiguousarray(trimap))
    return alpha.astype(np.float32)


def infill_background(
    rgb: np.ndarray,
    hole_mask: np.ndarray,
    method: str = "lama",
) -> np.ndarray:
    """
    Fill a "hole" region of `rgb` (where hole_mask=True) with plausible content
    so the resulting image can be used as a full-coverage backdrop behind the
    foreground layer stack.

    method:
      - "opencv": cv2.INPAINT_NS (Navier-Stokes). Near-instant, no ML deps.
                  Good for simple/smooth backgrounds (walls, gradients, sky).
      - "lama":   LaMa model via `simple-lama-inpainting`. Plausible on
                  complex scenes; downloads a ~200 MB checkpoint on first use.

    Returns a uint8 (H, W, 3) RGB array at source dims.
    """
    if hole_mask.dtype != bool:
        hole_mask = hole_mask.astype(bool)
    if method == "opencv":
        import cv2  # type: ignore
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask_u8 = (hole_mask.astype(np.uint8)) * 255
        filled_bgr = cv2.inpaint(bgr, mask_u8, 5, cv2.INPAINT_NS)
        return cv2.cvtColor(filled_bgr, cv2.COLOR_BGR2RGB)
    elif method == "lama":
        from simple_lama_inpainting import SimpleLama  # type: ignore
        global _LAMA
        if _LAMA is None:
            _LAMA = SimpleLama()
        mask_pil = Image.fromarray((hole_mask.astype(np.uint8)) * 255, mode="L")
        src_pil = Image.fromarray(rgb, mode="RGB")
        filled = _LAMA(src_pil, mask_pil)
        if filled.size != src_pil.size:
            filled = filled.resize(src_pil.size, Image.Resampling.BICUBIC)
        return np.asarray(filled.convert("RGB"))
    else:
        raise ValueError(f"unknown infill method {method!r}")


_LAMA = None  # cache the SimpleLama instance


def _coerce_alpha(mask_or_alpha: np.ndarray) -> np.ndarray:
    """Convert bool/uint8/float mask into a float32 alpha array in [0, 1]."""
    if mask_or_alpha.dtype == bool:
        return mask_or_alpha.astype(np.float32)
    if mask_or_alpha.dtype == np.uint8:
        return mask_or_alpha.astype(np.float32) / 255.0
    return np.clip(mask_or_alpha.astype(np.float32), 0.0, 1.0)


def mask_to_rgba(
    rgb: np.ndarray,
    mask_or_alpha: np.ndarray,
    feather: int = 0,
) -> Image.Image:
    """Compose RGB + mask/alpha → RGBA PIL image. Feather is only applied if > 0."""
    H, W = mask_or_alpha.shape[:2]
    alpha_f = _coerce_alpha(mask_or_alpha)
    alpha_u8 = (alpha_f * 255.0).astype(np.uint8)
    if feather > 0:
        alpha_u8 = feather_alpha(alpha_u8, feather)
    out = np.empty((H, W, 4), dtype=np.uint8)
    out[..., :3] = rgb
    out[..., 3] = alpha_u8
    return Image.fromarray(out, mode="RGBA")


def run_depth(img: Image.Image, device: str) -> Optional[Image.Image]:
    """Return a uint8 L-mode depth map at source dims. None if the model isn't installed."""
    try:
        from transformers import pipeline  # type: ignore
    except ImportError:
        return None

    try:
        pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device if device != "mps" else -1,  # pipeline mps can be flaky; fall back
        )
    except Exception as e:
        print(f"[depth] Depth Anything V2 unavailable: {e}", file=sys.stderr)
        return None

    result = pipe(img)
    depth = result.get("depth") if isinstance(result, dict) else result[0]["depth"]
    if not isinstance(depth, Image.Image):
        depth = Image.fromarray(np.asarray(depth))
    depth = depth.convert("L").resize(img.size, Image.Resampling.BICUBIC)
    return depth


PREVIEW_TINTS = [
    (255, 80, 80),    # red    — nearest
    (80, 200, 120),   # green
    (80, 140, 255),   # blue
    (230, 190, 80),   # amber
    (200, 100, 230),  # magenta
    (80, 220, 220),   # cyan
]


def build_preview(rgb: np.ndarray, ordered_layers: list[tuple[str, np.ndarray]]) -> Image.Image:
    """Tint each layer's region and composite back-to-front onto the source image."""
    H, W, _ = rgb.shape
    canvas = Image.fromarray(rgb).convert("RGBA")
    for idx, (_name, m) in enumerate(reversed(ordered_layers)):
        tint = PREVIEW_TINTS[(len(ordered_layers) - 1 - idx) % len(PREVIEW_TINTS)]
        a = _coerce_alpha(m)
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        overlay[..., :3] = tint
        overlay[..., 3] = (a * 110).astype(np.uint8)
        canvas = Image.alpha_composite(canvas, Image.fromarray(overlay, "RGBA"))
    return canvas.convert("RGB")


def build_css_snippet(layer_names: list[str], include_bg: bool = True, wordmark: bool = True) -> str:
    """Return a ready-to-paste HTML+CSS snippet that stacks the output PNGs."""
    ordered = (["bg"] if include_bg else []) + list(reversed(layer_names))
    html_lines = ['<div class="hero">']
    if include_bg:
        html_lines.append('  <img src="images/bg.png" class="layer l-bg" alt="">')
    if wordmark:
        html_lines.append('  <h1 class="hero-wordmark">your wordmark</h1>')
    for name in list(reversed(layer_names)):
        safe = name.replace(" ", "-")
        html_lines.append(f'  <img src="images/{name}.png" class="layer l-{safe}" alt="">')
    html_lines.append('</div>')

    css_lines = [
        '.hero { position: relative; isolation: isolate; overflow: hidden; }',
        '.hero .layer { position: absolute; inset: 0; width: 100%; height: 100%;',
        '               object-fit: cover; object-position: center; pointer-events: none; }',
    ]
    z = 1
    if include_bg:
        css_lines.append(f'.hero .l-bg {{ z-index: {z}; }}')
        z += 1
    if wordmark:
        css_lines.append(f'.hero .hero-wordmark {{ z-index: {z}; position: relative; /* your text styles here */ }}')
        z += 1
    for name in list(reversed(layer_names)):
        safe = name.replace(" ", "-")
        css_lines.append(f'.hero .l-{safe} {{ z-index: {z}; }}')
        z += 1

    return "<!-- HTML -->\n" + "\n".join(html_lines) + "\n\n/* CSS */\n" + "\n".join(css_lines) + "\n"


def parse_layers_arg(spec: str) -> list[LayerSpec]:
    # Accept either a JSON string or a path to a JSON file.
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("--layers must be a JSON list of {name, points} objects")
    return [LayerSpec.from_dict(d) for d in data]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="source image (JPEG/PNG)")
    ap.add_argument("--layers", required=True,
                    help="JSON string or path to JSON file. List of {name, points, labels?}. "
                         "Order = depth order (first = nearest).")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--feather", type=int, default=2, help="Gaussian feather radius (px) for --edges feather")
    ap.add_argument("--edges", default="feather", choices=["feather", "sam-soft", "matting"],
                    help="edge quality: feather (hard+blur, fast), sam-soft (SAM's own soft logits), "
                         "matting (pymatting closed-form, slowest + best on hair/fine edges)")
    ap.add_argument("--matting-band", type=int, default=8, help="unknown-region band width in px for --edges matting")
    ap.add_argument("--matting-algo", default="cf", choices=["cf", "lbdm", "knn"],
                    help="matting solver: cf (default, measured fastest + highest quality on typical "
                         "problems), lbdm (approximation for very large unknown regions), knn")
    ap.add_argument("--preview", action="store_true", help="write _preview.png composite")
    ap.add_argument("--no-depth", action="store_true", help="skip Depth Anything V2")
    ap.add_argument("--no-bg", action="store_true", help="skip writing bg.png (inverse union)")
    ap.add_argument("--no-css", action="store_true", help="skip writing snippet.html")
    ap.add_argument("--infill", default="none", choices=["none", "opencv", "lama"],
                    help="fill the bg hole (where foreground layers sit) with plausible content. "
                         "opencv = Navier-Stokes (instant, simple bgs). lama = LaMa model (better, ~200MB). "
                         "When set, bg.png is opaque everywhere and contains the infilled backdrop.")
    ap.add_argument("--model", default="sam2-hiera-large",
                    help="Segmentation model id. SAM 2: sam2-hiera-{large,base-plus,"
                         "small,tiny}. SAM 3 (better, gated): facebook/sam3.")
    args = ap.parse_args()

    src_path = Path(args.image).expanduser()
    if not src_path.exists():
        print(f"error: {src_path} not found", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layers_arg(args.layers)
    if not layers:
        print("error: --layers produced zero layers", file=sys.stderr)
        return 2

    device = detect_device(args.device)
    print(f"[device] {device}")

    img = Image.open(src_path).convert("RGB")
    W, H = img.size
    rgb = np.asarray(img)
    print(f"[source] {src_path.name} {W}x{H}")

    print(f"[sam2] loading {args.model} on {device} (weights auto-download on first run)")
    predictor = load_sam2_predictor(args.model, device)
    predictor.set_image(rgb)

    # Depth-ordered bool masks first (for exclusivity logic).
    hard_masks: list[tuple[str, np.ndarray]] = []
    for layer in layers:
        print(f"[sam2] segmenting {layer.name!r} from {len(layer.points)} point(s)")
        m = segment_layer(predictor, layer, (W, H))
        hard_masks.append((layer.name, m))

    claimed = np.zeros((H, W), dtype=bool)
    exclusive_bool: list[tuple[str, np.ndarray]] = []
    for name, m in hard_masks:
        m_excl = m & ~claimed
        exclusive_bool.append((name, m_excl))
        claimed |= m

    # Turn each exclusive hard mask into a final alpha according to --edges.
    final: list[tuple[str, np.ndarray]] = []
    for name, m_excl in exclusive_bool:
        if args.edges == "matting":
            print(f"[matting] refining {name!r} (band={args.matting_band}px, algo={args.matting_algo})")
            try:
                alpha = matting_refine(rgb, m_excl, band_px=args.matting_band, algo=args.matting_algo)
            except ImportError as e:
                print(f"[matting] unavailable ({e}); falling back to feather", file=sys.stderr)
                alpha = m_excl
        elif args.edges == "sam-soft":
            # Re-run this layer with return_logits and gate by the exclusive hard region
            # so soft alpha only lives within the pixels this layer owns.
            layer = next(L for L in layers if L.name == name)
            soft = segment_layer(predictor, layer, (W, H), return_soft=True)
            alpha = soft * m_excl.astype(np.float32)
        else:
            alpha = m_excl
        final.append((name, alpha))

    # Write per-layer RGBA PNGs.
    for name, a in final:
        feather = args.feather if args.edges == "feather" else 0
        rgba = mask_to_rgba(rgb, a, feather)
        out_path = out_dir / f"{name}.png"
        rgba.save(out_path, optimize=True)
        a_f = _coerce_alpha(a)
        coverage = float(a_f.mean())
        print(f"[write] {out_path.name} size={rgba.size} coverage={coverage*100:.1f}%")

    # Background = inverse of the soft union over all layer alphas. We use the
    # Porter-Duff "over" inverse so overlapping soft alphas combine correctly.
    if not args.no_bg:
        union = np.zeros((H, W), dtype=np.float32)
        for _name, a in final:
            union = 1.0 - (1.0 - union) * (1.0 - _coerce_alpha(a))
        if args.infill == "none":
            bg_alpha = 1.0 - union
            feather = args.feather if args.edges == "feather" else 0
            rgba_bg = mask_to_rgba(rgb, bg_alpha, feather)
            out_path = out_dir / "bg.png"
            rgba_bg.save(out_path, optimize=True)
            print(f"[write] {out_path.name} size={rgba_bg.size} coverage={float(bg_alpha.mean())*100:.1f}%")
        else:
            hole = union > 0.5  # binary hole where foreground sits
            print(f"[infill] {args.infill} filling {float(hole.mean())*100:.1f}% of frame")
            filled_rgb = infill_background(rgb, hole, method=args.infill)
            opaque_alpha = np.ones((H, W), dtype=np.float32)
            rgba_bg = mask_to_rgba(filled_rgb, opaque_alpha, 0)
            out_path = out_dir / "bg.png"
            rgba_bg.save(out_path, optimize=True)
            print(f"[write] {out_path.name} size={rgba_bg.size} (infilled, opaque)")

    # Depth map.
    if not args.no_depth:
        print("[depth] running Depth Anything V2 (small-hf)")
        d = run_depth(img, device)
        if d is not None:
            out_path = out_dir / "depth.png"
            d.save(out_path, optimize=True)
            print(f"[write] {out_path.name} size={d.size}")
        else:
            print("[depth] skipped (install `transformers` to enable)")

    # Preview composite.
    if args.preview:
        preview = build_preview(rgb, final)
        out_path = out_dir / "_preview.png"
        preview.save(out_path, optimize=True)
        print(f"[write] {out_path.name} size={preview.size}")

    # CSS snippet.
    if not args.no_css:
        snippet = build_css_snippet([name for name, _ in final], include_bg=not args.no_bg)
        (out_dir / "snippet.html").write_text(snippet)
        print(f"[write] snippet.html ({len(final)} layer(s))")

    # Dimension sanity check.
    ok = True
    for f in sorted(out_dir.glob("*.png")):
        with Image.open(f) as im:
            if im.size != (W, H):
                ok = False
                print(f"[FAIL] {f.name} is {im.size}, expected {(W, H)}")
    if ok:
        print(f"[ok] all PNGs match source dims {W}x{H}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
