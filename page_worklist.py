from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import cxr_model as M
import page_study

HERE = Path(__file__).resolve().parent


def render():
    st.title("📋 Triage worklist (batch)")
    st.caption("Upload many chest X-rays: the AI scores all of them and sorts the list so STAT / urgent studies are read first.")
    ups = st.file_uploader("X-rays (PNG / JPG / DICOM)", type=["png", "jpg", "jpeg", "dcm", "dicom"], accept_multiple_files=True)
    use_demo = st.checkbox("Include the demo images", value=not ups)
    cut = st.session_state.setdefault("mx_cutoff", .55)
    if st.button("▶️ Score worklist", type="primary"):
        items = [(u.name, u.getvalue()) for u in (ups or [])]
        if use_demo:
            items += [(f, (HERE / f).read_bytes()) for f in page_study.DEMOS.values()]
        rows, bar = [], st.progress(0.0)
        for i, (name, data) in enumerate(items):
            try:
                img, meta, res = page_study._analyse(data, name, cut)
                d = res["table"]
                pos = d[d.score >= cut]
                rows.append({"study": name, "triage": res["triage"], "n_findings": len(pos),
                             "top findings": ", ".join(f"{f} ({s:.2f})" for f, s in zip(pos.finding[:4], pos.score[:4])) or "-",
                             "max score": float(d.score.max()), "view": meta.get("ViewPosition", ""),
                             **{f: float(s) for f, s in zip(d.finding, d.score)}, "_data": data})
            except Exception as e:
                rows.append({"study": name, "triage": "ERROR", "top findings": str(e)[:80], "_data": data})
            bar.progress((i + 1) / len(items), name)
        st.session_state.mx_worklist = pd.DataFrame(rows)
    wl = st.session_state.get("mx_worklist")
    if wl is None or wl.empty:
        return
    wl = wl.assign(_r=wl.triage.map({**M.RANK, "ERROR": 9})).sort_values(["_r", "max score"], ascending=[True, False])
    icon = {"STAT": "🔴 STAT", "Urgent": "🟠 Urgent", "Routine": "🔵 Routine", "Normal": "🟢 Normal", "ERROR": "⚠️ error"}
    show = wl.assign(triage=wl.triage.map(icon))[["triage", "study", "n_findings", "top findings", "max score", "view"]]
    st.dataframe(show, hide_index=True)
    c = wl.triage.value_counts().reset_index()
    c.columns = ["triage", "studies"]
    st.plotly_chart(px.pie(c, names="triage", values="studies", hole=.5, title="Worklist by priority",
                           color="triage", color_discrete_map={"STAT": "#C0392B", "Urgent": "#E67E22", "Routine": "#2E86C1", "Normal": "#27AE60"}),
                    key="wlpie")
    finding_cols = [f for f in M.PATHOLOGIES if f in wl]
    if finding_cols:
        st.plotly_chart(px.imshow(wl.set_index("study")[finding_cols], color_continuous_scale="Reds", zmin=0, zmax=1, aspect="auto",
                                  title="AI score heat-map (studies × findings)"), key="wlheat")
    pick = st.selectbox("Open a study in the viewer", list(wl.study))
    if st.button("Open in study viewer"):
        page_study.load(wl.set_index("study").loc[pick, "_data"], pick)
        st.success("Opened — go to **🩻 Study viewer**.")
    st.download_button("⬇️ Worklist (CSV)", wl.drop(columns=["_data", "_r"]).to_csv(index=False).encode(), "cxr_worklist.csv")
