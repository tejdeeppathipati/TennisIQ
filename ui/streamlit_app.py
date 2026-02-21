from __future__ import annotations

from pathlib import Path

import streamlit as st


st.set_page_config(page_title="TennisIQ", layout="wide")
st.title("TennisIQ Viewer")

run_dir = st.text_input("Run directory", value="outputs/runs")
path = Path(run_dir)

if path.exists():
    st.success(f"Found: {path}")
    overlay = path / "overlay.mp4"
    if overlay.exists():
        st.video(str(overlay))
else:
    st.warning("Run directory not found.")
