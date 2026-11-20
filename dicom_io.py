"""DICOM reading: VOI LUT / windowing, MONOCHROME1 inversion, metadata, simple anonymisation."""
from __future__ import annotations

import io

import numpy as np

TAGS = ["PatientName", "PatientID", "PatientSex", "PatientAge", "PatientBirthDate", "StudyDate", "StudyDescription",
        "Modality", "BodyPartExamined", "ViewPosition", "Manufacturer", "InstitutionName", "PixelSpacing", "ImagerPixelSpacing",
        "Rows", "Columns", "BitsStored", "PhotometricInterpretation", "KVP", "Exposure"]


def read(data: bytes) -> tuple[np.ndarray, dict]:
    import pydicom
    from pydicom.pixels import apply_voi_lut
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array.astype(np.float64)
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float64)
    except Exception:
        pass
    if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr  # MONOCHROME1: high values are dark
    if arr.ndim == 3:  # multi-frame or RGB -> first frame / grey
        arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr.mean(2)
    lo, hi = np.percentile(arr, .5), np.percentile(arr, 99.5)
    img = (np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-6) * 255
    meta = {"format": "DICOM"}
    for t in TAGS:
        v = ds.get(t)
        if v is not None and str(v) != "":
            meta[t] = str(v)
    sp = ds.get("PixelSpacing") or ds.get("ImagerPixelSpacing")
    if sp:
        try:
            meta["mm_per_pixel"] = float(sp[0])
        except Exception:
            pass
    return img, meta


def anonymise(meta: dict) -> dict:
    hide = {"PatientName", "PatientID", "PatientBirthDate", "InstitutionName"}
    return {k: ("***" if k in hide else v) for k, v in meta.items()}
