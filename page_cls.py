"""Shared study page for the brain-MRI and skin-lesion classifiers."""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import cls_model as C
import cxr_model

HERE = Path(__file__).resolve().parent

MODALITIES = {
    "brain": {
        "model": "brain_mri", "title": "🧠 Brain MRI tumour AI", "icon": "🧠",
        "labels": {"glioma_tumor": "Glioma", "meningioma_tumor": "Meningioma", "no_tumor": "No tumour", "pituitary_tumor": "Pituitary tumour"},
        "explain": {"glioma_tumor": "Tumour of glial cells inside the brain; grade varies from low to high (glioblastoma).",
                    "meningioma_tumor": "Usually benign tumour of the meninges (brain coverings), often slow-growing.",
                    "no_tumor": "No tumour pattern recognised by the model.",
                    "pituitary_tumor": "Tumour of the pituitary gland (mostly benign adenomas); may affect hormones or vision."},
        "action": {"glioma_tumor": ("Urgent", "Suspected intra-axial tumour → urgent neuro-oncology / neurosurgery referral; contrast-enhanced MRI (± perfusion/spectroscopy)."),
                   "meningioma_tumor": ("Soon", "Suspected extra-axial lesion → neurosurgical review; contrast MRI; small asymptomatic lesions may be monitored."),
                   "pituitary_tumor": ("Soon", "Suspected sellar lesion → endocrinology (hormone panel) + visual fields; dedicated pituitary MRI protocol."),
                   "no_tumor": ("Routine", "No tumour pattern detected by AI → routine report; correlate clinically.")},
        "upload_hint": "Single 2-D MRI slice (axial, coronal or sagittal; T1 / T1-contrast / T2) as PNG / JPG / DICOM.",
        "source": "Brain Tumor Classification (MRI) dataset — S. Bhuvaji et al., 2020 (Kaggle / GitHub). 3,264 slices, 4 classes.",
        "limits": "Trained on single 2-D slices from one public dataset; no sequence information, no 3-D context, no tumour grading or "
                  "segmentation, and other diagnoses (stroke, abscess, metastasis, MS) are not covered. The official test split of this dataset "
                  "is known to differ from its training split, which lowers test accuracy.",
        "demos": {"Glioma (test split)": "demo_mri_glioma.jpg", "Glioma the model gets WRONG (dataset shift example)": "demo_mri_glioma_missed.jpg",
                  "Meningioma (test split)": "demo_mri_meningioma.jpg",
                  "Pituitary (test split)": "demo_mri_pituitary.jpg", "No tumour (test split)": "demo_mri_normal.jpg"},
        "accent": "#4A235A",
    },
    "skin": {
        "model": "skin_lesion", "title": "🔬 Skin lesion (dermoscopy) AI", "icon": "🔬",
        "labels": {"akiec": "Actinic keratosis / Bowen's (AKIEC)", "bcc": "Basal cell carcinoma (BCC)", "bkl": "Benign keratosis (BKL)",
                   "df": "Dermatofibroma (DF)", "mel": "Melanoma (MEL)", "nv": "Melanocytic naevus / mole (NV)", "vasc": "Vascular lesion (VASC)"},
        "explain": {"akiec": "Pre-cancerous / in-situ squamous lesion from sun damage.", "bcc": "Most common skin cancer; grows slowly, rarely spreads.",
                    "bkl": "Benign: seborrhoeic keratosis, solar lentigo, lichen-planus-like keratosis.", "df": "Benign fibrous skin nodule.",
                    "mel": "Melanoma - the most dangerous skin cancer; early excision is curative.", "nv": "Common benign mole.",
                    "vasc": "Benign vascular lesion (cherry angioma, angiokeratoma, pyogenic granuloma)."},
        "action": {"mel": ("Urgent", "Suspected melanoma → urgent (2-week-wait) dermatology referral for excision biopsy."),
                   "bcc": ("Soon", "Suspected BCC → dermatology referral (routine-soon); excision / Mohs / topical depending on site."),
                   "akiec": ("Soon", "Actinic keratosis / Bowen's → dermatology or GP treatment (cryotherapy, 5-FU, PDT)."),
                   "bkl": ("Routine", "Likely benign keratosis → reassurance; review if changing."),
                   "df": ("Routine", "Likely dermatofibroma → reassurance."),
                   "nv": ("Routine", "Likely benign naevus → reassurance; monitor with ABCDE / photos."),
                   "vasc": ("Routine", "Likely vascular lesion → reassurance.")},
        "upload_hint": "Dermoscopic photo of one lesion, centred (as from a dermatoscope). Ordinary phone photos are out of distribution.",
        "source": "HAM10000 — Tschandl, Rosendahl & Kittler, Sci Data 2018 (CC BY-NC 4.0). 10,015 dermoscopic images, 7 classes.",
        "limits": "Dermoscopic images only, mostly fair-skinned European patients (Austria / Australia); performance on darker skin types, "
                  "clinical (non-dermoscopic) photos, nails, mucosa and rare lesion types is unknown. The data are imbalanced (67% naevi).",
        "demos": {"Melanoma (test split)": "demo_skin_mel.jpg", "Naevus (test split)": "demo_skin_nv.jpg",
                  "Basal cell carcinoma (test split)": "demo_skin_bcc.jpg", "Benign keratosis (test split)": "demo_skin_bkl.jpg"},
        "accent": "#7E5109", "low_conf": .45,
        "malignant": ["mel", "bcc", "akiec"],
    },
}
URG = {"Urgent": "🔴", "Soon": "🟠", "Routine": "🟢"}


@st.cache_data(show_spinner=False)
def _run(data: bytes, name: str, model: str):
    rgb, m = C.load_rgb(data, name)
    return rgb, m, C.predict(rgb, model)


def render(kind: str):
    cfg = MODALITIES[kind]
    model = cfg["model"]
    meta = C.meta(model)
    key = f"mx_{kind}"
    st.title(cfg["title"])
    st.caption(cfg["upload_hint"])
    a, b = st.columns([2, 3])
    with a:
        demo = st.selectbox("Demo image", list(cfg["demos"]), index=None, placeholder="choose a demo...", key=f"{key}_demo")
        if demo and st.button("Open demo", type="primary", key=f"{key}_open"):
            p = HERE / cfg["demos"][demo]
            st.session_state[key] = {"name": p.name, "data": p.read_bytes(), "decision": "not reviewed"}
    with b:
        up = st.file_uploader("…or upload", type=["png", "jpg", "jpeg", "dcm", "dicom"], key=f"{key}_up")
        if up is not None and st.button("Analyse upload", type="primary", key=f"{key}_go"):
            st.session_state[key] = {"name": up.name, "data": up.getvalue(), "decision": "not reviewed"}
    s = st.session_state.get(key)
    if not s:
        st.info("Open a demo or upload an image.")
        model_card(cfg, meta)
        return
    try:
        rgb, imeta, res = _run(s["data"], s["name"], model)
    except Exception as e:
        st.error(f"Could not analyse the image: {e}")
        return
    df = res["table"].copy()
    df["label"] = df["class"].map(cfg["labels"])
    top = res["top"]
    urg, action = cfg["action"][top]
    if kind == "skin":
        mal = float(df[df["class"].isin(cfg["malignant"])].probability.sum())
        pmel = float(df.set_index("class").probability["mel"])
        c = st.columns(3)
        c[0].metric("Malignant / pre-malignant probability", f"{mal:.0%}", help="MEL + BCC + AKIEC")
        c[1].metric("Melanoma probability", f"{pmel:.0%}")
        thr = c[2].slider("Melanoma alert threshold", .02, .9, float(meta.get("mel_threshold", .1)), .01, key=f"{key}_thr",
                          help="Default 0.10 was chosen on the validation set (sensitivity 86%, specificity 80%). "
                               "Lower = fewer missed melanomas but more false alarms.")
        if pmel >= thr and top != "mel":
            st.error(f"Melanoma probability {pmel:.0%} is above the {thr:.0%} alert threshold even though the top class is "
                     f"{cfg['labels'][top]} → treat as suspicious and refer.")
            urg, action = "Urgent", cfg["action"]["mel"][1]
    st.subheader(f"{URG[urg]} AI result: **{cfg['labels'][top]}** ({res['top_p']:.0%}) — {s['name']}")
    if top != "no_tumor" or urg != "Routine":
        (st.error if urg == "Urgent" else st.warning if urg == "Soon" else st.success)(f"**Suggested action ({urg}):** {action}")
    else:
        st.success(f"**Suggested action:** {action}")
    low = cfg.get("low_conf", .6)
    if res["top_p"] < low:
        st.warning(f"Low confidence (< {low:.0%}): the model is unsure — the image may be atypical or out of distribution.")
    t1, t2, t3, t4 = st.tabs(["🖼️ Image & probabilities", "🔥 Heat-maps", "📝 Review & report", "📊 Model card & accuracy"])
    with t1:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.image(rgb, caption=f"{imeta.get('size', '')} {imeta.get('format', '')}", width="stretch")
        with c2:
            fig = px.bar(df.sort_values("probability"), x="probability", y="label", orientation="h", range_x=[0, 1],
                         color="probability", color_continuous_scale="Reds", title="Class probabilities (softmax)")
            st.plotly_chart(fig, key=f"{key}_bar")
            df["meaning"] = df["class"].map(cfg["explain"])
            st.dataframe(df[["label", "probability", "meaning"]], hide_index=True,
                         column_config={"probability": st.column_config.ProgressColumn("probability", min_value=0, max_value=1, format="%.3f")})
    with t2:
        cls = st.selectbox("Explain class", list(df["class"]), format_func=lambda x: cfg["labels"][x], key=f"{key}_cls")
        alpha = st.slider("Overlay strength", .1, .9, .5, .05, key=f"{key}_alpha")
        c1, c2 = st.columns(2)
        with c1:
            st.image(C.overlay(rgb, C.cam(res["features"], model, cls), alpha), caption=f"Class activation map — {cfg['labels'][cls]}",
                     width="stretch")
        with c2:
            ok = f"{key}_occ_{s['name']}_{cls}"
            if st.button("Compute occlusion map (~5 s)", key=f"{key}_occbtn"):
                st.session_state[ok] = C.occlusion(rgb, model, cls)
            if ok in st.session_state:
                st.image(C.overlay(rgb, st.session_state[ok], alpha), caption=f"Occlusion sensitivity — {cfg['labels'][cls]}", width="stretch")
        st.caption("Heat-maps show which regions drove the prediction. Attention on rulers, ink marks, hair or image borders "
                   "(skin) or on the skull / text (MRI) means the model may be using shortcuts.")
    with t3:
        p = s.setdefault("patient", {"name": "", "id": "", "age": "", "sex": "", "site": "", "indication": ""})
        x = st.columns(4)
        p["name"] = x[0].text_input("Patient", p["name"], key=f"{key}_pn")
        p["id"] = x[1].text_input("ID / MRN", p["id"], key=f"{key}_pid")
        p["age"] = x[2].text_input("Age", p["age"], key=f"{key}_age")
        p["sex"] = x[3].text_input("Sex", p["sex"], key=f"{key}_sex")
        p["site"] = st.text_input("Lesion site" if kind == "skin" else "MRI sequence / plane", p["site"], key=f"{key}_site")
        p["indication"] = st.text_input("Clinical indication", p["indication"], key=f"{key}_ind")
        s["decision"] = st.radio("Clinician review of AI result", ["not reviewed", "Agree", "Disagree", "Uncertain - needs further imaging / biopsy"],
                                 horizontal=True, key=f"{key}_dec")
        default = f"AI suggests {cfg['labels'][top]} ({res['top_p']:.0%}). {action}"
        impression = st.text_area("Impression / plan (clinician)", default, height=100, key=f"{key}_imp")
        clinician = st.text_input("Reported by", "", key=f"{key}_by")
        pdf = report(cfg, s, df, rgb, res, impression, clinician, urg)
        st.download_button("⬇️ Download report (PDF)", pdf, f"{kind}_AI_report_{p['id'] or 'case'}.pdf", "application/pdf", type="primary",
                           key=f"{key}_pdf")
    with t4:
        model_card(cfg, meta)


def model_card(cfg, meta):
    st.markdown(f"**Model:** {meta['backbone']} fine-tuned in this project · input {meta['size']}×{meta['size']} · "
                f"{len(meta['classes'])} classes  \n**Training data:** {cfg['source']}")
    c = st.columns(4)
    c[0].metric("Validation accuracy", f"{meta['val_acc']:.1%}", help=f"n = {meta['n_val']}")
    c[1].metric("Validation macro-F1", f"{meta['val_macro_f1']:.2f}")
    if "test_acc" in meta:
        c[2].metric("Held-out test accuracy", f"{meta['test_acc']:.1%}", help=f"n = {meta['n_test']}")
        c[3].metric("Test macro-F1", f"{meta['test_macro_f1']:.2f}")
    if meta.get("extra"):
        st.write(meta["extra"])
    cm = np.array(meta.get("test_confusion", meta["val_confusion"]))
    lab = [cfg["labels"][k] for k in meta["classes"]]
    st.plotly_chart(px.imshow(cm, x=lab, y=lab, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="True"),
                              title="Confusion matrix — " + ("held-out test set" if "test_confusion" in meta else "validation set")),
                    key=f"cm_{cfg['model']}")
    rec = cm.diagonal() / cm.sum(1).clip(min=1)
    st.dataframe(pd.DataFrame({"class": lab, "sensitivity (recall)": rec.round(3), "n": cm.sum(1)}), hide_index=True)
    st.warning("**Limitations:** " + cfg["limits"] + " Research / education only — not a medical device.")


def report(cfg, s, df, rgb, res, impression, clinician, urg) -> bytes:
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor(cfg["accent"]))
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor(cfg["accent"]))
    B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    S = ParagraphStyle("s", parent=B, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#555555"))
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=14 * mm, bottomMargin=14 * mm)
    p = s.get("patient", {})
    el = [Paragraph(cfg["title"].split(" ", 1)[1] + " — AI-assisted report", H1),
          Table([["Patient", p.get("name", ""), "ID", p.get("id", "")], ["Age / sex", f"{p.get('age', '')} / {p.get('sex', '')}", "Date", str(dt.date.today())],
                 ["Site / sequence", p.get("site", ""), "Indication", p.get("indication", "")]], colWidths=[30 * mm, 62 * mm, 25 * mm, 65 * mm],
                style=[("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), .25, colors.grey)]), Spacer(1, 6)]
    el.append(Paragraph(f"<b>AI result:</b> {cfg['labels'][res['top']]} ({res['top_p']:.0%}) &nbsp; · &nbsp; <b>Priority:</b> {urg} &nbsp; · "
                        f"&nbsp; <b>Clinician review:</b> {s.get('decision')}", B))
    el.append(Paragraph("Impression / plan", H2))
    el.append(Paragraph(impression.replace("\n", "<br/>"), B))
    im1, im2 = io.BytesIO(cxr_model.to_png(np.asarray(rgb), 700)), io.BytesIO(cxr_model.to_png(C.overlay(rgb, C.cam(res["features"], cfg["model"], res["top"]), .5), 700))
    el.append(Spacer(1, 6))
    el.append(Table([[RLImage(im1, 80 * mm, 80 * mm), RLImage(im2, 80 * mm, 80 * mm)], [Paragraph("Image", S), Paragraph("AI heat-map (top class)", S)]]))
    el.append(Paragraph("Class probabilities", H2))
    rows = [["Class", "Probability"]] + [[r.label, f"{r.probability:.3f}"] for r in df.itertuples()]
    el.append(Table(rows, colWidths=[110 * mm, 40 * mm], style=[("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), .25, colors.grey),
                                                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE"))]))
    m = C.meta(cfg["model"])
    el.append(Paragraph("About the model", H2))
    el.append(Paragraph(f"{m['backbone']} fine-tuned on {cfg['source']} Held-out test accuracy "
                        f"{m.get('test_acc', m['val_acc']):.1%}. {cfg['limits']}", S))
    el.append(Paragraph("<b>Not a medical device.</b> Decision support for education/research; the clinician is responsible for the diagnosis.", S))
    el.append(Spacer(1, 10))
    el.append(Paragraph(f"Reported by: {clinician or '________________'} &nbsp;&nbsp; Signature: ________________", B))
    doc.build(el)
    return buf.getvalue()
