"""Generic image classifier (brain MRI, skin lesion): MobileNetV3-Large fine-tuned in this project, run with ONNX Runtime.

Each model ships as <name>.onnx (+ split parts if large) and <name>_meta.json holding class names, normalisation,
final-layer weights (for class activation maps) and the validation / test metrics measured after training.
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


@lru_cache(maxsize=4)
def meta(name: str) -> dict:
    return json.loads((HERE / f"{name}_meta.json").read_text())


@lru_cache(maxsize=4)
def session(name: str) -> ort.InferenceSession:
    single = HERE / f"{name}.onnx"
    blob = single.read_bytes() if single.exists() else b"".join(
        (HERE / f"{name}_part{i}.bin").read_bytes() for i in (1, 2))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    return ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])


def load_rgb(data: bytes, name: str) -> tuple[np.ndarray, dict]:
    if name.lower().endswith((".dcm", ".dicom")) or data[128:132] == b"DICM":
        import dicom_io
        g, m = dicom_io.read(data)
        return np.repeat(g[..., None], 3, 2).astype(np.uint8), m
    im = Image.open(io.BytesIO(data))
    return np.asarray(im.convert("RGB")), {"format": im.format, "size": f"{im.width}×{im.height}"}


def preprocess(rgb: np.ndarray, name: str) -> np.ndarray:
    m = meta(name)
    s = m["size"]
    x = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((s, s), Image.BILINEAR), np.float32) / 255.0
    x = (x - np.array(m["mean"], np.float32)) / np.array(m["std"], np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def predict(rgb: np.ndarray, name: str) -> dict:
    f, lg = session(name).run(None, {"image": preprocess(rgb, name)})
    p = softmax(lg[0])
    cls = meta(name)["classes"]
    df = pd.DataFrame({"class": cls, "probability": p}).sort_values("probability", ascending=False).reset_index(drop=True)
    return {"table": df, "features": f[0], "top": df["class"][0], "top_p": float(df.probability[0])}


def cam(features: np.ndarray, name: str, cls: str, out=224) -> np.ndarray:
    m = meta(name)
    w = np.array(m["fc_weight"], np.float32)[m["classes"].index(cls)]
    h = np.maximum(np.tensordot(w, features, axes=(0, 0)), 0)
    h = h / h.max() if h.max() > 0 else h
    return np.array(Image.fromarray(h.astype(np.float32)).resize((out, out), Image.BICUBIC)).clip(0, 1)


def occlusion(rgb: np.ndarray, name: str, cls: str, patch=48, stride=24) -> np.ndarray:
    m = meta(name)
    s = m["size"]
    x = preprocess(rgb, name)[0]
    k = m["classes"].index(cls)
    base = softmax(session(name).run(None, {"image": x[None]})[1][0])[k]
    coords = [(y, xx) for y in range(0, s - patch + 1, stride) for xx in range(0, s - patch + 1, stride)]
    batch = np.repeat(x[None], len(coords), 0)
    for i, (y, xx) in enumerate(coords):
        batch[i, :, y:y + patch, xx:xx + patch] = 0.0  # mean colour after normalisation
    sc = np.concatenate([softmax(session(name).run(None, {"image": batch[i:i + 16]})[1])[:, k] for i in range(0, len(batch), 16)])
    heat, cnt = np.zeros((s, s)), np.zeros((s, s))
    for (y, xx), v in zip(coords, sc):
        heat[y:y + patch, xx:xx + patch] += max(base - v, 0)
        cnt[y:y + patch, xx:xx + patch] += 1
    heat /= np.maximum(cnt, 1)
    return heat / heat.max() if heat.max() > 0 else heat


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha=.5, size=512) -> Image.Image:
    from cxr_model import colormap
    base = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((size, size)).convert("RGB"), np.float32)
    h = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), np.float32) / 255
    a = (alpha * h)[..., None]
    return Image.fromarray((base * (1 - a) + colormap(h).astype(np.float32) * a).clip(0, 255).astype(np.uint8))
