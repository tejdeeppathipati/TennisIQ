from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import distance

from tennisiq.geometry.court_reference import CourtReference


Point = Tuple[Optional[float], Optional[float]]

court_ref = CourtReference()
refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

court_conf_ind = {}
for i in range(len(court_ref.court_conf)):
    conf = court_ref.court_conf[i + 1]
    inds = []
    for j in range(4):
        inds.append(court_ref.key_points.index(conf[j]))
    court_conf_ind[i + 1] = inds


def get_trans_matrix(points: List[Point]) -> Optional[np.ndarray]:
    matrix_trans = None
    dist_max = np.inf
    for conf_ind in range(1, 13):
        conf = court_ref.court_conf[conf_ind]
        inds = court_conf_ind[conf_ind]
        inters = [points[inds[0]], points[inds[1]], points[inds[2]], points[inds[3]]]
        if any((p[0] is None or p[1] is None) for p in inters):
            continue
        matrix, _ = cv2.findHomography(np.float32(conf), np.float32(inters), method=0)
        if matrix is None:
            continue
        trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
        dists = []
        for i in range(12):
            if i in inds:
                continue
            if points[i][0] is None or points[i][1] is None:
                continue
            dists.append(distance.euclidean(points[i], np.squeeze(trans_kps[i])))
        dist_mean = np.mean(dists) if dists else 0.0
        if dist_mean < dist_max:
            matrix_trans = matrix
            dist_max = dist_mean
    return matrix_trans
