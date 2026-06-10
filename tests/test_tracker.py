"""Tests for ByteTracker."""

import numpy as np
import pytest

from yolo_ptq_bench.detector import DetectionResult
from yolo_ptq_bench.tracker import ByteTracker, _iou_matrix


def make_result(boxes, scores, class_ids=None):
    if class_ids is None:
        class_ids = np.zeros(len(boxes), dtype=int)
    return DetectionResult(
        boxes=np.array(boxes, dtype=np.float32),
        scores=np.array(scores, dtype=np.float32),
        class_ids=class_ids,
        class_names=["obj"] * len(boxes),
    )


class TestIoUMatrix:
    def test_perfect_overlap(self):
        box = np.array([[0, 0, 10, 10]], dtype=float)
        iou = _iou_matrix(box, box)
        assert iou[0, 0] == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.array([[0, 0, 5, 5]], dtype=float)
        b = np.array([[10, 10, 20, 20]], dtype=float)
        iou = _iou_matrix(a, b)
        assert iou[0, 0] == pytest.approx(0.0)

    def test_empty_inputs(self):
        iou = _iou_matrix(np.empty((0, 4)), np.array([[0, 0, 1, 1]]))
        assert iou.shape == (0, 1)


class TestByteTracker:
    def test_no_detections_returns_empty(self):
        tracker = ByteTracker()
        result = make_result([], [])
        tracks = tracker.update(result)
        assert tracks == []

    def test_new_track_spawned(self):
        tracker = ByteTracker(min_hits=1)
        result = make_result([[10, 10, 50, 50]], [0.9])
        tracks = tracker.update(result)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    def test_track_persists_across_frames(self):
        tracker = ByteTracker(min_hits=1)
        for _ in range(5):
            result = make_result([[10, 10, 50, 50]], [0.9])
            tracks = tracker.update(result)
        assert len(tracks) == 1
        assert tracks[0].hits == 5

    def test_track_dies_after_max_age(self):
        tracker = ByteTracker(min_hits=1, max_age=3)
        result = make_result([[10, 10, 50, 50]], [0.9])
        tracker.update(result)

        empty = make_result([], [])
        for _ in range(5):
            tracks = tracker.update(empty)
        assert tracks == []

    def test_multiple_tracks(self):
        tracker = ByteTracker(min_hits=1)
        boxes = [[10, 10, 50, 50], [200, 200, 250, 250]]
        result = make_result(boxes, [0.9, 0.85])
        tracks = tracker.update(result)
        assert len(tracks) == 2

    def test_low_confidence_rescue(self):
        tracker = ByteTracker(high_thresh=0.5, low_thresh=0.1, min_hits=1)
        result_high = make_result([[10, 10, 50, 50]], [0.8])
        tracker.update(result_high)

        # Low-confidence detection in same location — should keep track alive
        result_low = make_result([[12, 12, 52, 52]], [0.2])
        tracks = tracker.update(result_low)
        assert len(tracks) == 1

    def test_reset_clears_state(self):
        tracker = ByteTracker(min_hits=1)
        tracker.update(make_result([[10, 10, 50, 50]], [0.9]))
        tracker.reset()
        assert tracker._next_id == 1
        assert tracker._tracks == []

    def test_trail_grows(self):
        tracker = ByteTracker(min_hits=1)
        for i in range(10):
            result = make_result([[10 + i, 10, 50 + i, 50]], [0.9])
            tracker.update(result)
        active = tracker.active_tracks
        assert len(active[0].trail) == 10
