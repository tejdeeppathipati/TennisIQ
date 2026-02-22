"""
Standard tennis court reference geometry.

All coordinates are in a top-down 2-D space (units = arbitrary but proportional
to real meters).  The 14 keypoints match the ordering produced by
CourtKeypointNet (indices 0-13).

Reference court dimensions (ITF):
    Singles court:  23.77 m  x  8.23 m
    Doubles court:  23.77 m  x 10.97 m
    Service box:     6.40 m  deep on each side
    Centre line splits service boxes

The coordinate system places (0, 0) at the top-left corner of the doubles
court, with +x going right and +y going down.

Source reference: https://github.com/yastrebksv/TennisCourtDetector
"""
import numpy as np


class CourtReference:
    """Canonical court geometry with 14 keypoints in a fixed coordinate frame."""

    def __init__(self):
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))

        self.key_points = [
            *self.baseline_top,
            *self.baseline_bottom,
            *self.left_inner_line,
            *self.right_inner_line,
            *self.top_inner_line,
            *self.bottom_inner_line,
            *self.middle_line,
        ]

        self.court_conf = {
            1: [*self.baseline_top, *self.baseline_bottom],
            2: [
                self.left_inner_line[0], self.right_inner_line[0],
                self.left_inner_line[1], self.right_inner_line[1],
            ],
            3: [
                self.left_inner_line[0], self.right_court_line[0],
                self.left_inner_line[1], self.right_court_line[1],
            ],
            4: [
                self.left_court_line[0], self.right_inner_line[0],
                self.left_court_line[1], self.right_inner_line[1],
            ],
            5: [*self.top_inner_line, *self.bottom_inner_line],
            6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
            7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
            8: [
                self.right_inner_line[0], self.right_court_line[0],
                self.right_inner_line[1], self.right_court_line[1],
            ],
            9: [
                self.left_court_line[0], self.left_inner_line[0],
                self.left_court_line[1], self.left_inner_line[1],
            ],
            10: [
                self.top_inner_line[0], self.middle_line[0],
                self.bottom_inner_line[0], self.middle_line[1],
            ],
            11: [
                self.middle_line[0], self.top_inner_line[1],
                self.middle_line[1], self.bottom_inner_line[1],
            ],
            12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
        }

        self.court_width = 1117
        self.court_height = 2408
        self.top_bottom_border = 549
        self.right_left_border = 274
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_height + self.top_bottom_border * 2

        # ITF doubles court: 10.97m wide, 23.77m long
        self.real_width_m = 10.97
        self.real_length_m = 23.77
        self.meters_per_unit = self.real_width_m / self.court_width  # ~0.00982

    def get_keypoints_array(self) -> np.ndarray:
        """Return the 14 reference keypoints as shape (14, 2) float32 array."""
        return np.array(self.key_points, dtype=np.float32)

    def get_court_configurations(self) -> dict[int, list[tuple]]:
        """Return all 12 four-point court configurations for homography search."""
        return self.court_conf

    def get_conf_indices(self) -> dict[int, list[int]]:
        """Map each court configuration to its keypoint indices (0-13)."""
        conf_indices = {}
        for conf_id, conf_points in self.court_conf.items():
            indices = []
            for pt in conf_points:
                indices.append(self.key_points.index(pt))
            conf_indices[conf_id] = indices
        return conf_indices
