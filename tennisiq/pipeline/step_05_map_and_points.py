from __future__ import annotations

from typing import List

from tennisiq.events.inout import classify_in_out
from tennisiq.geometry.polygons import default_court_polygon


def run_step_05_map_and_points(frame_records: List[dict]):
    court_poly = default_court_polygon()
    enriched = []
    for row in frame_records:
        row = dict(row)
        row["ball_inout"] = classify_in_out(tuple(row["ball_point"]), court_poly)
        enriched.append(row)
    return enriched
