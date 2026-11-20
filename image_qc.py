"""Image quality checks and display enhancements for radiographs."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def qc(img: np.ndarray, meta: dict) -> tuple[dict, list[dict]]:
    h, w = img.shape
    p1, p99 = np.percentile(img, [1, 99])
    sat_lo, sat_hi = float((img <= 1).mean() * 100), float((img >= 254).mean() * 100)
    # orientation / inversion heuristic: on a normal PA chest film the mediastinum (centre) is brighter than the lungs
    cy0, cy1 = int(h * .25), int(h * .7)
    centre = img[cy0:cy1, int(w * .45):int(w * .55)].mean()
    lungs = (img[cy0:cy1, int(w * .18):int(w * .35)].mean() + img[cy0:cy1, int(w * .65):int(w * .82)].mean()) / 2
    left = img[:, : w // 2]
    right = img[:, w - w // 2:][:, ::-1]
    sym = float(np.corrcoef(left.ravel()[::7], right.ravel()[::7])[0, 1])
    m = {"width": w, "height": h, "aspect": round(w / h, 2), "dynamic_range": round(p99 - p1, 1), "contrast_sd": round(float(img.std()), 1),
         "mean_brightness": round(float(img.mean()), 1), "pct_black": round(sat_lo, 2), "pct_white": round(sat_hi, 2),
         "mediastinum_minus_lungs": round(float(centre - lungs), 1), "left_right_symmetry": round(sym, 3)}
    f = []
    if min(h, w) < 256:
        f.append(("fail", f"Low resolution ({w}×{h}); the model needs ≥ 224 px and diagnostic images are usually ≥ 2000 px."))
    elif min(h, w) < 512:
        f.append(("warn", f"Moderate resolution ({w}×{h}) - fine for AI, limited for human reading."))
    if m["dynamic_range"] < 80:
        f.append(("warn", "Low contrast / narrow dynamic range - possible under- or over-exposure."))
    if sat_hi > 8 or sat_lo > 25:
        f.append(("warn", f"Clipped pixels (white {sat_hi:.1f}%, black {sat_lo:.1f}%) - detail may be lost."))
    if m["mediastinum_minus_lungs"] < -10:
        f.append(("fail", "Lungs look brighter than the mediastinum - the image may be inverted (negative). Use 'Invert' or check "
                          "the DICOM photometric interpretation."))
    if not .6 <= m["aspect"] <= 1.6:
        f.append(("warn", "Unusual aspect ratio - is this a chest radiograph? The model only works on frontal chest X-rays."))
    if sym < .5:
        f.append(("warn", "Low left/right symmetry - rotated patient, lateral view, or a non-chest image."))
    view = meta.get("ViewPosition")
    if view and view.upper() == "AP":
        f.append(("info", "AP view: the heart looks larger than on PA, so cardiomegaly / CTR must be interpreted with caution."))
    if view and view.upper() in ("LL", "RL", "LATERAL", "LAT"):
        f.append(("fail", "Lateral view - the model was trained on frontal (PA/AP) radiographs."))
    if not f:
        f.append(("good", "Image quality checks passed."))
    return m, [{"level": a, "text": b} for a, b in f]


def enhance(img: np.ndarray, mode: str = "Original", window: tuple[float, float] = (0, 255), gamma: float = 1.0, invert=False) -> np.ndarray:
    a = img.astype(np.float64)
    lo, hi = window
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if gamma != 1.0:
        a = a ** (1 / gamma)
    if mode == "CLAHE (local contrast)":
        from skimage import exposure
        a = exposure.equalize_adapthist(a, clip_limit=.015)
    elif mode == "Histogram equalisation":
        from skimage import exposure
        a = exposure.equalize_hist(a)
    elif mode == "Edge enhance (unsharp)":
        from skimage.filters import unsharp_mask
        a = unsharp_mask(a, radius=3, amount=1.5)
    if invert:
        a = 1 - a
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def draw_ctr(img: np.ndarray, heart: tuple[float, float], thorax: tuple[float, float], heart_y=.62, thorax_y=.72) -> Image.Image:
    """Draw cardiothoracic-ratio lines (positions as fractions of width / height)."""
    im = Image.fromarray(img.astype(np.uint8)).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, w // 300)
    for (x0, x1), y, col in ((heart, heart_y, (255, 80, 80)), (thorax, thorax_y, (80, 200, 255))):
        yy = int(h * y)
        d.line([(int(w * x0), yy), (int(w * x1), yy)], fill=col, width=lw)
        for x in (x0, x1):
            d.line([(int(w * x), yy - h // 40), (int(w * x), yy + h // 40)], fill=col, width=lw)
    return im
