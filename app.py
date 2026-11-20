"""Medical Imaging AI — chest X-ray, brain MRI and skin-lesion assistants.  Run: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Medical Imaging AI", page_icon="🩻", layout="wide")

import page_cls  # noqa: E402
import page_learn  # noqa: E402
import page_study  # noqa: E402
import page_worklist  # noqa: E402

PAGES = {"🩻 Chest X-ray": page_study.render, "📋 X-ray triage worklist": page_worklist.render,
         "🧠 Brain MRI": lambda: page_cls.render("brain"), "🔬 Skin lesion": lambda: page_cls.render("skin"),
         "📚 Learn & model cards": page_learn.render}


def main():
    st.sidebar.title("🩻 Medical Imaging AI")
    page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
    s = st.session_state.get("mx_study")
    if s:
        st.sidebar.caption(f"Open study: {s['name']} · triage {s['res']['triage']}")
    st.sidebar.caption("⚠️ Research/education only — not a medical device.")
    PAGES[page]()


main()
