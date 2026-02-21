from __future__ import annotations

from tennisiq.events.features import FEATURE_NAMES, build_candidate_feature_row, compute_track_kinematics, feature_matrix


def test_feature_shape_and_kinematics():
    ball_track = [(100.0, 100.0), (110.0, 120.0), (120.0, 130.0), (118.0, 122.0), (115.0, 108.0)]
    mapped = [
        {
            "ball_court_xy": p,
            "playerA_court_xy": (130.0, 300.0),
            "playerB_court_xy": (130.0, 50.0),
            "homography_confidence": 0.9,
        }
        for p in ball_track
    ]
    cand = {"frame_idx": 2, "rule_score": 0.7, "reversal": True, "speed_drop": True}

    kin = compute_track_kinematics(ball_track, fps=30)
    row = build_candidate_feature_row(2, cand, kin, ball_track, mapped, net_y=1748.0)
    mat = feature_matrix([row])

    assert len(kin) == len(ball_track)
    assert set(FEATURE_NAMES).issubset(row.keys())
    assert mat.shape == (1, len(FEATURE_NAMES))
