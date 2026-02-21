from __future__ import annotations

from tennisiq.events.segmentation import run_point_state_machine


def _frames(n: int):
    out = []
    for i in range(n):
        out.append(
            {
                "ball_visible": i >= 3,
                "ball_xy": (100.0 + i, 200.0 + i) if i >= 3 else (None, None),
                "ball_court_xy": (800.0, 1800.0) if i >= 3 else (None, None),
                "ball_speed": 700.0 if i == 4 else 200.0,
                "homography_confidence": 0.9,
                "ball_inout": "in",
            }
        )
    return out


def test_state_machine_out_reason():
    frames = _frames(20)
    frames[8]["ball_inout"] = "out"

    bounces = [{"frame_idx": 8, "score": 0.9}]
    hits = [{"frame_idx": 4, "score": 0.8}]

    points = run_point_state_machine(
        frames,
        bounce_events=bounces,
        hit_events=hits,
        fps=30,
        inactivity_frames=2,
        ball_lost_frames=5,
        serve_speed_thresh=500.0,
        net_y=1748.0,
    )

    assert len(points) == 1
    assert points[0]["end_reason"] == "OUT"


def test_state_machine_ball_lost_reason():
    frames = _frames(18)
    for i in range(10, 18):
        frames[i]["ball_visible"] = False
        frames[i]["ball_xy"] = (None, None)
        frames[i]["ball_court_xy"] = (None, None)

    points = run_point_state_machine(
        frames,
        bounce_events=[],
        hit_events=[{"frame_idx": 4, "score": 0.8}],
        fps=30,
        inactivity_frames=2,
        ball_lost_frames=3,
        serve_speed_thresh=500.0,
        net_y=1748.0,
    )

    assert len(points) == 1
    assert points[0]["end_reason"] == "BALL_LOST"
