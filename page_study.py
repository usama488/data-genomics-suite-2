import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import cxr_model as M
import dicom_io
import image_qc as Q
import report_pdf

HERE = Path(__file__).resolve().parent
DEMOS = {"Demo 1 — NIH ChestX-ray14 (PNG)": "demo_cxr_nih_1.png", "Demo 2 — NIH ChestX-ray14 (PNG)": "demo_cxr_nih_2.png",
         "Demo 3 — DICOM (CR, PA, male 62)": "demo_cxr_dicom.dcm",
         "Demo 4 — QC test: wrongly tagged DICOM (shows as negative)": "demo_cxr_qc_inverted.dcm"}
LEVEL = {"fail": st.error, "warn": st.warning, "info": st.info, "good": st.success}


@st.cache_data(show_spinner=False)
def _analyse(data: bytes, name: str, cutoff: float):
    img, meta = M.load_image(data, name)
    res = M.predict(img, cutoff)
    return img, meta, res


def load(data: bytes, name: str):
    cutoff = st.session_state.get("mx_cutoff", .55)
    img, meta, res = _analyse(data, name, cutoff)
    st.session_state.mx_study = {"name": name, "data": data, "img": img, "meta": meta, "res": res, "decisions": {}, "ctr": None,
                                 "patient": {"name": meta.get("PatientName", ""), "id": meta.get("PatientID", Path(name).stem[:20]),
                                             "sex": meta.get("PatientSex", ""), "age": meta.get("PatientAge", ""),
                                             "view": meta.get("ViewPosition", "PA"), "date": meta.get("StudyDate", str(dt.date.today())),
                                             "exam": "Chest X-ray", "indication": "", "clinician": ""}}


def render():
    st.session_state.setdefault("mx_cutoff", .55)
    st.title("🩻 Chest X-ray AI — study viewer")
    st.caption("Upload a frontal chest radiograph (PNG / JPG / DICOM) or open a demo. Everything runs locally in the app — no image leaves the server.")
    a, b = st.columns([2, 3])
    with a:
        demo = st.selectbox("Demo image", list(DEMOS), index=None, placeholder="choose a demo...")
        if demo and st.button("Open demo", type="primary"):
            load((HERE / DEMOS[demo]).read_bytes(), DEMOS[demo])
    with b:
        up = st.file_uploader("…or upload an X-ray", type=["png", "jpg", "jpeg", "dcm", "dicom"])
        if up is not None and st.button("Analyse upload", type="primary"):
            try:
                load(up.getvalue(), up.name)
            except Exception as e:
                st.error(f"Could not read the image: {e}")
    s = st.session_state.get("mx_study")
    if not s:
        st.info("Open a demo or upload an image to start.")
        return
    if s.get("cut") != st.session_state.mx_cutoff:  # triage depends on the cut-off
        s["res"] = _analyse(s["data"], s["name"], st.session_state.mx_cutoff)[2]
        s["cut"] = st.session_state.mx_cutoff
    img, meta, res = s["img"], s["meta"], s["res"]
    df = res["table"]
    tri = res["triage"]
    colour = {"STAT": "🔴", "Urgent": "🟠", "Routine": "🔵", "Normal": "🟢"}[tri]
    top = df[df.score >= st.session_state.get("mx_cutoff", .55)]
    st.subheader(f"{colour} AI triage: **{tri}** — {s['name']}")
    st.write("Most likely findings: " + (", ".join(f"**{f.replace('_', ' ')}** ({sc:.2f})" for f, sc in zip(top.finding, top.score))
                                         if len(top) else "none above the triage cut-off"))
    tabs = st.tabs(["🖼️ Viewer & QC", "🤖 AI findings", "🔥 Heat-maps", "📏 CTR measurement", "📝 Review & report"])

    with tabs[0]:
        qm, flags = Q.qc(img, meta)
        c1, c2 = st.columns([3, 2])
        with c2:
            mode = st.selectbox("Enhancement", ["Original", "CLAHE (local contrast)", "Histogram equalisation", "Edge enhance (unsharp)"])
            win = st.slider("Window (min / max)", 0, 255, (0, 255))
            gamma = st.slider("Gamma", .4, 2.5, 1.0, .05)
            inv = st.checkbox("Invert (negative)")
            for f in flags:
                LEVEL[f["level"]](f["text"])
            st.dataframe(pd.DataFrame(qm.items(), columns=["metric", "value"]).astype(str), hide_index=True)
            if meta.get("format") == "DICOM":
                with st.expander("DICOM header (identifiers masked)"):
                    st.json(dicom_io.anonymise(meta))
        with c1:
            view = Q.enhance(img, mode, win, gamma, inv)
            st.image(view, caption=f"{mode} · {img.shape[1]}×{img.shape[0]} px", width="stretch")
            st.plotly_chart(px.histogram(x=img.ravel()[::17], nbins=64, title="Pixel intensity histogram", height=220,
                                         labels={"x": "intensity"}), key="hist")

    with tabs[1]:
        cut = st.slider("Triage cut-off (score)", .5, .8, step=.01, key="mx_cutoff",
                        help="0.5 is the model's operating threshold. A higher cut-off reduces false alarms but may miss findings.")
        d = df.copy()
        d["label"] = d.finding.str.replace("_", " ")
        fig = px.bar(d.sort_values("score"), x="score", y="label", orientation="h", color="confidence", range_x=[0, 1], height=560,
                     color_discrete_map={"likely": "#C0392B", "possible": "#E67E22", "borderline": "#F4D03F", "low": "#AED6F1", "unlikely": "#D5D8DC"},
                     title="AI score per finding (0.5 = operating threshold)")
        fig.add_vline(x=.5, line_dash="dash", annotation_text="threshold")
        fig.add_vline(x=cut, line_dash="dot", line_color="red", annotation_text="triage cut-off")
        st.plotly_chart(fig, key="bars")
        st.dataframe(d[["finding", "score", "confidence", "priority", "raw_probability", "threshold", "description"]], hide_index=True,
                     column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=1, format="%.3f"),
                                    "raw_probability": st.column_config.NumberColumn("raw prob.", format="%.3f"),
                                    "threshold": st.column_config.NumberColumn("model threshold", format="%.3f")})
        st.caption("Raw probability is the sigmoid output; the score rescales it so the model's own operating threshold sits at 0.5, "
                   "making findings comparable. Several labels overlap (e.g. Lung Opacity ⊃ Consolidation ⊃ Pneumonia).")

    with tabs[2]:
        f = st.selectbox("Finding", list(df.finding), format_func=lambda x: f"{x.replace('_', ' ')} ({df.set_index('finding').score[x]:.2f})")
        alpha = st.slider("Overlay strength", .1, .9, .5, .05)
        c1, c2 = st.columns(2)
        with c1:
            h = M.cam(res["features"], f)
            st.image(M.overlay(img, h, alpha), caption=f"Class activation map — {f}", width="stretch")
            st.caption("CAM: weights of the final layer × the last convolutional feature maps (7×7 grid). Fast and exact for this "
                       "architecture, but coarse.")
        with c2:
            key = f"occ_{s['name']}_{f}"
            if st.button("Compute occlusion map (~10 s)"):
                st.session_state[key] = M.occlusion(img, f)
            if key in st.session_state:
                st.image(M.overlay(img, st.session_state[key], alpha), caption=f"Occlusion sensitivity — {f}", width="stretch")
                st.caption("Occlusion: grey squares are slid over the image; red = covering this region lowers the score most.")
            else:
                st.info("Occlusion sensitivity is model-agnostic and slower: it hides each region and measures the score drop.")
        st.warning("Heat-maps show where the model looked, not the lesion outline. Attention outside the lungs (tubes, text "
                   "markers) can reveal shortcut learning.")

    with tabs[3]:
        st.write("Mark the widest **heart** diameter and the widest inner **thoracic** diameter (fractions of image width).")
        c1, c2 = st.columns([3, 2])
        with c2:
            heart = st.slider("Heart borders (red)", 0.0, 1.0, (0.36, 0.66), .005)
            hy = st.slider("Heart line height", .3, .9, .62, .01)
            thorax = st.slider("Thorax inner borders (blue)", 0.0, 1.0, (0.12, 0.88), .005)
            ty = st.slider("Thorax line height", .3, .95, .72, .01)
            ctr = (heart[1] - heart[0]) / max(thorax[1] - thorax[0], 1e-6)
            s["ctr"] = ctr
            st.metric("Cardiothoracic ratio", f"{ctr:.2f}", "enlarged (> 0.50)" if ctr > .5 else "normal (≤ 0.50)",
                      delta_color="inverse" if ctr > .5 else "normal")
            if s["patient"].get("view", "").upper() == "AP":
                st.warning("AP film: the heart is magnified, so CTR over-estimates size.")
            mm = meta.get("mm_per_pixel")
            if mm:
                st.write(f"Heart width ≈ {(heart[1] - heart[0]) * img.shape[1] * mm / 10:.1f} cm · "
                         f"thorax ≈ {(thorax[1] - thorax[0]) * img.shape[1] * mm / 10:.1f} cm (from DICOM pixel spacing)")
            ai = df.set_index("finding").score["Cardiomegaly"]
            st.caption(f"AI cardiomegaly score for comparison: {ai:.2f}")
        with c1:
            st.image(Q.draw_ctr(Q.enhance(img), heart, thorax, hy, ty), width="stretch")

    with tabs[4]:
        p = s["patient"]
        x = st.columns(4)
        p["name"] = x[0].text_input("Patient name", p.get("name", ""))
        p["id"] = x[1].text_input("Patient ID / MRN", p.get("id", ""))
        p["age"] = x[2].text_input("Age", p.get("age", ""))
        p["sex"] = x[3].text_input("Sex", p.get("sex", ""))
        x = st.columns(3)
        p["view"] = x[0].selectbox("View", ["PA", "AP", "AP supine"], index=["PA", "AP", "AP supine"].index(p["view"]) if p.get("view") in ("PA", "AP", "AP supine") else 0)
        p["clinician"] = x[1].text_input("Referring clinician", p.get("clinician", ""))
        p["date"] = x[2].text_input("Study date", p.get("date", ""))
        p["indication"] = st.text_input("Clinical indication", p.get("indication", ""))
        st.markdown("**Confirm or reject each AI finding (score ≥ 0.4)**")
        for r in df[df.score >= .4].itertuples():
            s["decisions"][r.finding] = st.radio(f"{r.finding.replace('_', ' ')} — AI {r.score:.2f} ({r.confidence})",
                                                 ["not reviewed", "Confirmed", "Rejected", "Uncertain"], horizontal=True,
                                                 index=["not reviewed", "Confirmed", "Rejected", "Uncertain"].index(s["decisions"].get(r.finding, "not reviewed")),
                                                 key=f"dec_{s['name']}_{r.finding}")
        conf = [f.replace("_", " ").lower() for f, d in s["decisions"].items() if d == "Confirmed"]
        default = ("Findings: " + ", ".join(conf) + ". Clinical correlation recommended." if conf else
                   "No acute cardiopulmonary abnormality identified.")
        if s.get("ctr"):
            default += f" Cardiothoracic ratio {s['ctr']:.2f}."
        impression = st.text_area("Impression (radiologist)", default, height=110)
        st.session_state.setdefault("mx_lab", {"name": "Department of Radiology", "address": "", "radiologist": ""})
        lab = st.session_state.mx_lab
        y = st.columns(3)
        lab["name"] = y[0].text_input("Hospital / department", lab["name"])
        lab["address"] = y[1].text_input("Address", lab["address"])
        lab["radiologist"] = y[2].text_input("Reporting radiologist", lab["radiologist"])
        best = df.finding[0]
        imgs = [(M.to_png(Q.enhance(img), 900), "Original radiograph"),
                (M.to_png(M.overlay(img, M.cam(res["features"], best), .5), 900), f"AI heat-map: {best.replace('_', ' ')} ({df.score[0]:.2f})")]
        pdf = report_pdf.build(s, lab, df, s["decisions"], impression, tri, imgs, s.get("ctr"), Q.qc(img, meta)[1])
        st.download_button("⬇️ Download radiology report (PDF)", pdf, f"CXR_report_{p.get('id') or 'study'}.pdf", "application/pdf", type="primary")
        st.download_button("⬇️ AI scores (CSV)", df.to_csv(index=False).encode(), "cxr_ai_scores.csv")
