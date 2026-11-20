"""Chest X-ray AI: DenseNet-121 from TorchXRayVision (Cohen et al., MIDL 2022), exported to ONNX.

Trained on 8 public datasets (NIH ChestX-ray14, PadChest, CheXpert, MIMIC-CXR, Google NIH labels, OpenI,
RSNA/Kaggle). Outputs 18 findings. Scores are calibrated so that 0.5 equals the model's operating
threshold (above 0.5 = model calls the finding positive).

The ONNX weights are stored as two ~14 MB parts so they can be uploaded to GitHub through the browser.
"""
from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "cxr_model_meta.json").read_text())
PATHOLOGIES: list[str] = META["pathologies"]
OP = np.array(META["op_threshs"], dtype=np.float64)
W = np.array(META["classifier_weight"], dtype=np.float32)  # (18, 1024)

# clinical grouping used for triage
PRIORITY = {"Pneumothorax": "STAT", "Edema": "Urgent", "Consolidation": "Urgent", "Pneumonia": "Urgent", "Effusion": "Urgent",
            "Mass": "Urgent", "Lung Lesion": "Urgent", "Fracture": "Urgent", "Cardiomegaly": "Routine",
            "Enlarged Cardiomediastinum": "Urgent", "Atelectasis": "Routine", "Infiltration": "Routine", "Emphysema": "Routine",
            "Fibrosis": "Routine", "Pleural_Thickening": "Routine", "Nodule": "Routine", "Hernia": "Routine", "Lung Opacity": "Routine"}
RANK = {"STAT": 0, "Urgent": 1, "Routine": 2, "Normal": 3}
DESCRIPTION = {
    "Atelectasis": "partial collapse of lung tissue", "Consolidation": "air spaces filled with fluid/pus (e.g. pneumonia)",
    "Infiltration": "ill-defined increased lung density", "Pneumothorax": "air in the pleural space - can be an emergency",
    "Edema": "fluid in the lungs, often from heart failure", "Emphysema": "destroyed, over-inflated alveoli (COPD)",
    "Fibrosis": "scarring of lung tissue", "Effusion": "fluid around the lung (pleural effusion)",
    "Pneumonia": "lung infection", "Pleural_Thickening": "thickened pleura", "Cardiomegaly": "enlarged heart (CTR > 0.5 on PA film)",
    "Nodule": "small rounded opacity (< 3 cm)", "Mass": "rounded opacity ≥ 3 cm", "Hernia": "hiatal / diaphragmatic hernia",
    "Lung Lesion": "focal lung abnormality", "Fracture": "rib / clavicle fracture", "Lung Opacity": "any area of increased density",
    "Enlarged Cardiomediastinum": "widened heart/mediastinal silhouette"}


@lru_cache(maxsize=1)
def session() -> ort.InferenceSession:
    blob = b"".join((HERE / f"cxr_model_part{i}.bin").read_bytes() for i in (1, 2))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    return ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])


# ------------------------------------------------------------------ preprocessing
def to_gray_uint(arr: np.ndarray) -> np.ndarray:
    """Any image array -> 2-D float image scaled 0..255."""
    a = arr.astype(np.float64)
    if a.ndim == 3:
        a = a[..., :3].mean(2)
    lo, hi = np.percentile(a, .1), np.percentile(a, 99.9)
    if a.max() > 255 or a.min() < 0:
        a = (np.clip(a, lo, hi) - lo) / max(hi - lo, 1e-6) * 255
    return a


def load_image(data: bytes, name: str) -> tuple[np.ndarray, dict]:
    if name.lower().endswith((".dcm", ".dicom")) or data[128:132] == b"DICM":
        import dicom_io
        return dicom_io.read(data)
    img = Image.open(io.BytesIO(data))
    meta = {"format": img.format, "mode": img.mode, "size": f"{img.width}×{img.height}"}
    return to_gray_uint(np.array(img)), meta


def center_square(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = img.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s], (y0, x0)


def preprocess(img255: np.ndarray, size=224) -> np.ndarray:
    """Identical to TorchXRayVision: scale to [-1024, 1024], centre-crop, skimage resize to 224."""
    from skimage.transform import resize
    x = (2 * (img255 / 255.0) - 1.0) * 1024
    sq, _ = center_square(x)
    r = resize(sq, (size, size), mode="constant", preserve_range=True)
    return r.astype(np.float32)[None, None]


# ------------------------------------------------------------------ inference
def calibrate(logits: np.ndarray) -> np.ndarray:
    p = 1 / (1 + np.exp(-logits.astype(np.float64)))
    return np.where(p < OP, p / (OP * 2), 1 - (1 - p) / ((1 - OP) * 2))


def predict(img255: np.ndarray, triage_cutoff: float = .55) -> dict:
    x = preprocess(img255)
    feats, logits = session().run(None, {"image": x})
    score = calibrate(logits[0])
    raw = 1 / (1 + np.exp(-logits[0]))
    df = pd.DataFrame({"finding": PATHOLOGIES, "score": score, "raw_probability": raw, "threshold": OP})
    df["positive"] = df.score >= .5
    df["confidence"] = pd.cut(df.score, [-1, .4, .5, triage_cutoff, .7, 2], labels=["unlikely", "low", "borderline", "possible", "likely"])
    df["priority"] = [PRIORITY[f] if s >= triage_cutoff else "-" for f, s in zip(df.finding, df.score)]
    df["description"] = df.finding.map(DESCRIPTION)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    pos = df[df.score >= triage_cutoff]
    triage = min((PRIORITY[f] for f in pos.finding), key=RANK.get, default="Normal")
    return {"table": df, "features": feats[0], "triage": triage, "input": x[0, 0]}


def cam(features: np.ndarray, finding: str, size=224) -> np.ndarray:
    """Class activation map: classifier weights · last conv features (exact for DenseNet + global-avg-pool)."""
    k = PATHOLOGIES.index(finding)
    m = np.tensordot(W[k], features, axes=(0, 0))  # 7x7
    m = np.maximum(m, 0)
    m = m / m.max() if m.max() > 0 else m
    return np.array(Image.fromarray(m.astype(np.float32)).resize((size, size), Image.BICUBIC)).clip(0, 1)


def occlusion(img255: np.ndarray, finding: str, patch=48, stride=24, size=224) -> np.ndarray:
    """Model-agnostic saliency: how much the score drops when each region is hidden."""
    x = preprocess(img255, size)[0, 0]
    k = PATHOLOGIES.index(finding)
    base = calibrate(session().run(None, {"image": x[None, None]})[1][0])[k]
    coords = [(y, xx) for y in range(0, size - patch + 1, stride) for xx in range(0, size - patch + 1, stride)]
    batch = np.repeat(x[None, None], len(coords), 0)
    for i, (y, xx) in enumerate(coords):
        batch[i, 0, y:y + patch, xx:xx + patch] = 0.0
    scores = []
    for i in range(0, len(batch), 16):
        scores.append(calibrate(session().run(None, {"image": batch[i:i + 16]})[1])[:, k])
    scores = np.concatenate(scores)
    heat, cnt = np.zeros((size, size)), np.zeros((size, size))
    for (y, xx), s in zip(coords, scores):
        heat[y:y + patch, xx:xx + patch] += max(base - s, 0)
        cnt[y:y + patch, xx:xx + patch] += 1
    heat = heat / np.maximum(cnt, 1)
    return heat / heat.max() if heat.max() > 0 else heat


# ------------------------------------------------------------------ visualisation helpers
_CMAP = np.array([[0, 0, 128], [0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 0], [128, 0, 0]], dtype=np.float32)


def colormap(h: np.ndarray) -> np.ndarray:
    pos = np.clip(h, 0, 1) * (len(_CMAP) - 1)
    i = np.floor(pos).astype(int).clip(0, len(_CMAP) - 2)
    f = (pos - i)[..., None]
    return (_CMAP[i] * (1 - f) + _CMAP[i + 1] * f).astype(np.uint8)


def overlay(img255: np.ndarray, heat: np.ndarray, alpha=.45, size=512) -> Image.Image:
    sq, _ = center_square(img255)
    base = np.array(Image.fromarray(sq.astype(np.uint8)).resize((size, size)).convert("RGB"), dtype=np.float32)
    h = np.array(Image.fromarray((heat * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)) / 255.0
    col = colormap(h).astype(np.float32)
    a = (alpha * h)[..., None]  # transparent where the map is low
    return Image.fromarray((base * (1 - a) + col * a).clip(0, 255).astype(np.uint8))


def to_png(img: np.ndarray | Image.Image, max_side=1024) -> bytes:
    im = img if isinstance(img, Image.Image) else Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    im.thumbnail((max_side, max_side))
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()
