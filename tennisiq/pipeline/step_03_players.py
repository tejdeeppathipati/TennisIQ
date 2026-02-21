from __future__ import annotations

from tennisiq.cv.players.infer import detect_players


def run_step_03_players(frames):
    return detect_players(frames)
