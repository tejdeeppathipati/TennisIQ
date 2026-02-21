from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="TennisIQ Dashboard", layout="wide")
st.title("TennisIQ Point Timeline Dashboard")


def _load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _latest_run_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.name)[-1]


def _points_list(points_payload: Dict[str, Any] | List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    if points_payload is None:
        return []
    if isinstance(points_payload, list):
        return points_payload
    return points_payload.get("points", [])


def _clips_list(points_payload: Dict[str, Any] | List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    if points_payload is None or isinstance(points_payload, list):
        return []
    return points_payload.get("clips", [])


def _build_timeline_df(points_payload: Dict[str, Any] | List[Dict[str, Any]] | None) -> pd.DataFrame:
    points = _points_list(points_payload)
    rows = []
    for p in points:
        start = float(p.get("start_sec", 0.0))
        end = float(p.get("end_sec", 0.0))
        rows.append(
            {
                "point_id": int(p.get("point_id", -1)),
                "start_sec": start,
                "end_sec": end,
                "duration_sec": max(0.0, end - start),
                "end_reason": p.get("end_reason", "UNKNOWN"),
                "serve_zone": p.get("serve_zone", None),
                "confidence": float(p.get("confidence", 0.0)),
                "rally_hit_count": int(p.get("rally_hit_count", 0)),
            }
        )
    return pd.DataFrame(rows)


def _point_clip_for_id(points_payload: Dict[str, Any] | List[Dict[str, Any]] | None, point_id: int) -> Path | None:
    for c in _clips_list(points_payload):
        if int(c.get("point_id", -1)) == point_id:
            return Path(c["clip"])
    return None


def _fallback_insights(stats: Dict[str, Any], points_df: pd.DataFrame) -> List[str]:
    serve_in_pct = float(stats.get("serve_in_pct", 0.0))
    reason = "OUT"
    if not points_df.empty:
        reason = str(points_df["end_reason"].value_counts().index[0])
    return [
        f"Serve in percentage is {serve_in_pct:.1f}%.",
        f"Most common point-ending reason: {reason}.",
    ]


def _plot_serve_map(serve_points: List[Dict[str, Any]]):
    fig, ax = plt.subplots(figsize=(6, 4))
    xs, ys, colors = [], [], []
    cmap = {
        "top_left": "tab:blue",
        "top_right": "tab:orange",
        "bottom_left": "tab:green",
        "bottom_right": "tab:red",
        None: "gray",
    }
    for p in serve_points:
        sb = p.get("serve_bounce", (None, None))
        if sb[0] is None or sb[1] is None:
            continue
        xs.append(sb[0])
        ys.append(sb[1])
        colors.append(cmap.get(p.get("zone"), "gray"))

    if xs:
        ax.scatter(xs, ys, c=colors, s=24)
    ax.set_title("Serve Placement Map")
    ax.set_xlabel("Court X")
    ax.set_ylabel("Court Y")
    ax.grid(alpha=0.3)
    st.pyplot(fig)


def _plot_error_heatmap(error_points: List[List[float]]):
    fig, ax = plt.subplots(figsize=(6, 4))
    if error_points:
        arr = np.array(error_points, dtype=float)
        ax.hist2d(arr[:, 0], arr[:, 1], bins=16)
    ax.set_title("Error Heatmap")
    ax.set_xlabel("Court X")
    ax.set_ylabel("Court Y")
    st.pyplot(fig)


def main():
    runs_root = Path("outputs/runs")
    latest = _latest_run_dir(runs_root)
    default_run = str(latest) if latest else "outputs/runs"

    run_dir = st.text_input("Run directory", value=default_run)
    run_path = Path(run_dir)

    if not run_path.exists():
        st.warning("Run directory not found.")
        return

    st.success(f"Loaded run: {run_path}")

    frames = _load_jsonl(run_path / "frames.jsonl")
    tracks = _load_json(run_path / "tracks.json") or {}
    points_payload = _load_json(run_path / "points.json") or {"points": [], "clips": []}
    insights_payload = _load_json(run_path / "insights.json") or {}

    overlay = run_path / "overlay.mp4"
    if overlay.exists():
        st.video(str(overlay))

    stats = insights_payload.get("stats", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Serve In %", f"{float(stats.get('serve_in_pct', 0.0)):.1f}")
    c2.metric("Points", int(stats.get("num_points", len(_points_list(points_payload)))))
    c3.metric("Bounces", int(stats.get("num_bounces", 0)))

    timeline_df = _build_timeline_df(points_payload)
    st.subheader("Point Timeline")
    if timeline_df.empty:
        st.info("No points found in points.json")
    else:
        st.dataframe(timeline_df, use_container_width=True)

        options = timeline_df["point_id"].tolist()
        selected_id = st.selectbox("Select point", options=options, index=0)
        selected = next((p for p in _points_list(points_payload) if int(p.get("point_id", -1)) == int(selected_id)), None)
        clip_path = _point_clip_for_id(points_payload, int(selected_id))

        st.subheader(f"Point {selected_id}")
        if selected:
            st.json(selected)

        if clip_path is not None and clip_path.exists():
            st.video(str(clip_path))
        else:
            st.warning("Clip not found for selected point.")

    st.subheader("Charts")
    visuals = insights_payload.get("visuals", {})
    serve_points = visuals.get("serve_placement_points", [])
    error_points = visuals.get("error_heatmap_points", [])

    col_l, col_r = st.columns(2)
    with col_l:
        _plot_serve_map(serve_points)
    with col_r:
        _plot_error_heatmap(error_points)

    st.subheader("Insights")
    insights = insights_payload.get("insights") or _fallback_insights(stats, timeline_df)
    for s in insights:
        st.write(f"- {s}")

    st.subheader("Point Cards")
    cards = insights_payload.get("point_cards", [])
    if not cards and not timeline_df.empty:
        # Minimal fallback cards.
        for _, row in timeline_df.head(5).iterrows():
            cards.append(
                {
                    "point_id": int(row["point_id"]),
                    "why": f"Point ended with {row['end_reason']}.",
                    "try_instead": "Use a higher-margin target and recover to center early.",
                }
            )

    for c in cards:
        st.markdown(f"**Point {c['point_id']}**")
        st.write(f"Why: {c['why']}")
        st.write(f"Try instead: {c['try_instead']}")

    with st.expander("Debug Summary"):
        st.write(f"frames.jsonl rows: {len(frames)}")
        st.write(f"tracks keys: {list(tracks.keys())}")


if __name__ == "__main__":
    main()
