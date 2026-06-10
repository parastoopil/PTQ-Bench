"""ByteTrack-inspired multi-object tracker using IoU matching and a Kalman filter.

Reference: ByteTrack — Multi-Object Tracking by Associating Every Detection Box
           (Zhang et al., ECCV 2022)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from yolo_ptq_bench.detector import DetectionResult


@dataclass
class KalmanState:
    """2D bounding box Kalman state: [cx, cy, s, r, dx, dy, ds]."""
    x: np.ndarray  # state vector (7,)
    P: np.ndarray  # covariance (7, 7)


@dataclass
class Track:
    """A tracked object across frames."""
    track_id: int
    bbox: np.ndarray      # xyxy
    score: float
    class_id: int
    class_name: str
    age: int = 0          # frames since last match
    hits: int = 1         # total frames matched
    trail: List[np.ndarray] = field(default_factory=list)


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU matrix between two sets of xyxy boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    return inter_area / np.clip(union_area, 1e-6, None)


class ByteTracker:
    """
    Multi-object tracker following the ByteTrack two-stage association strategy.

    Stage 1: Match high-confidence detections to existing tracks via IoU.
    Stage 2: Match low-confidence detections to unmatched tracks (rescues
             partially-occluded objects that produce weak detections).

    Args:
        high_thresh: Confidence threshold separating high/low detections.
        low_thresh:  Minimum confidence to consider at all.
        iou_thresh:  IoU threshold for a match to be accepted.
        max_age:     Frames a track survives without a detection hit.
        min_hits:    Frames before a track is considered confirmed.
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        iou_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits

        self._tracks: List[Track] = []
        self._next_id: int = 1
        self._frame_count: int = 0

    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.hits >= self.min_hits]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0

    def _hungarian_match(
        self, detections: np.ndarray, tracks: List[Track], iou_thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Return (matches, unmatched_det_ids, unmatched_trk_ids)."""
        if len(detections) == 0 or len(tracks) == 0:
            return [], list(range(len(detections))), list(range(len(tracks)))

        trk_boxes = np.array([t.bbox for t in tracks])
        iou = _iou_matrix(detections, trk_boxes)
        cost = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost)

        matches, unmatched_dets, unmatched_trks = [], [], []
        matched_d, matched_t = set(), set()

        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= iou_thresh:
                matches.append((r, c))
                matched_d.add(r)
                matched_t.add(c)

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_d]
        unmatched_trks = [i for i in range(len(tracks)) if i not in matched_t]

        return matches, unmatched_dets, unmatched_trks

    def update(self, result: DetectionResult) -> List[Track]:
        """
        Update tracker with detections from a single frame.
        Returns the list of confirmed tracks for this frame.
        """
        self._frame_count += 1

        if result.num_detections == 0:
            for t in self._tracks:
                t.age += 1
            self._tracks = [t for t in self._tracks if t.age <= self.max_age]
            return self.active_tracks

        # Split detections by confidence
        high_mask = result.scores >= self.high_thresh
        low_mask = (result.scores >= self.low_thresh) & ~high_mask

        high_dets = result.boxes[high_mask]
        high_scores = result.scores[high_mask]
        high_cls = result.class_ids[high_mask]
        high_names = [result.class_names[i] for i, m in enumerate(high_mask) if m]

        low_dets = result.boxes[low_mask]
        low_scores = result.scores[low_mask]
        low_cls = result.class_ids[low_mask]
        low_names = [result.class_names[i] for i, m in enumerate(low_mask) if m]

        # Stage 1: match high-conf detections to all tracks
        matches1, unmatched_high, unmatched_trks1 = self._hungarian_match(
            high_dets, self._tracks, self.iou_thresh
        )

        for det_idx, trk_idx in matches1:
            t = self._tracks[trk_idx]
            t.bbox = high_dets[det_idx]
            t.score = float(high_scores[det_idx])
            t.class_id = int(high_cls[det_idx])
            t.class_name = high_names[det_idx]
            t.hits += 1
            t.age = 0
            t.trail.append(t.bbox.copy())
            if len(t.trail) > 50:
                t.trail.pop(0)

        # Stage 2: match low-conf detections to still-unmatched tracks
        remaining_trks = [self._tracks[i] for i in unmatched_trks1]
        if len(low_dets) > 0 and len(remaining_trks) > 0:
            matches2, _, still_unmatched = self._hungarian_match(
                low_dets, remaining_trks, self.iou_thresh
            )
            for det_idx, trk_idx in matches2:
                t = remaining_trks[trk_idx]
                t.bbox = low_dets[det_idx]
                t.score = float(low_scores[det_idx])
                t.hits += 1
                t.age = 0
                unmatched_trks1.remove(self._tracks.index(t))

        # Age unmatched tracks and prune dead ones
        for i in unmatched_trks1:
            self._tracks[i].age += 1
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

        # Spawn new tracks for unmatched high-conf detections
        for i in unmatched_high:
            new_track = Track(
                track_id=self._next_id,
                bbox=high_dets[i].copy(),
                score=float(high_scores[i]),
                class_id=int(high_cls[i]),
                class_name=high_names[i],
                age=0,
                hits=1,
                trail=[high_dets[i].copy()],
            )
            self._tracks.append(new_track)
            self._next_id += 1

        return self.active_tracks
