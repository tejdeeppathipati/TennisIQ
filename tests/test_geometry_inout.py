from __future__ import annotations

from tennisiq.geometry.court_reference import CourtReference
from tennisiq.geometry.polygons import build_court_geometry, classify_point_in_out_line


def test_in_out_line_classification():
    ref = CourtReference()
    geom = build_court_geometry(ref)

    inside_pt = (float(ref.middle_line[0][0]), float(ref.net[0][1]))
    outside_pt = (float(ref.left_court_line[0][0] - 120), float(ref.net[0][1]))
    line_pt = (float(ref.left_inner_line[0][0]), float(ref.net[0][1]))

    in_label, _ = classify_point_in_out_line(inside_pt, geom, line_margin=6.0)
    out_label, _ = classify_point_in_out_line(outside_pt, geom, line_margin=6.0)
    line_label, meta = classify_point_in_out_line(line_pt, geom, line_margin=8.0)

    assert in_label in {"in", "line"}
    assert out_label == "out"
    assert line_label == "line"
    assert meta["line_distance"] <= 8.0
