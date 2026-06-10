"""Tests for DetectionMetrics and helper functions."""

import numpy as np
import pytest

from yolo_ptq_bench.metrics import (
    DetectionMetrics,
    compute_ap,
    compute_iou_matrix,
    compute_precision_recall,
)


class TestComputeIoU:
    def test_identical_boxes(self):
        box = np.array([[0, 0, 10, 10]], dtype=float)
        iou = compute_iou_matrix(box, box)
        assert iou[0, 0] == pytest.approx(1.0)

    def test_half_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=float)
        b = np.array([[5, 0, 15, 10]], dtype=float)
        iou = compute_iou_matrix(a, b)
        # intersection = 5×10=50, union = 10×10 + 10×10 - 50 = 150
        assert iou[0, 0] == pytest.approx(50 / 150, abs=1e-4)

    def test_no_overlap(self):
        a = np.array([[0, 0, 5, 5]], dtype=float)
        b = np.array([[10, 10, 20, 20]], dtype=float)
        assert compute_iou_matrix(a, b)[0, 0] == pytest.approx(0.0)

    def test_empty_a(self):
        iou = compute_iou_matrix(np.empty((0, 4)), np.zeros((3, 4)))
        assert iou.shape == (0, 3)


class TestComputeAP:
    def test_perfect_precision(self):
        recalls = np.linspace(0, 1, 11)
        precisions = np.ones(11)
        ap = compute_ap(recalls, precisions)
        assert ap == pytest.approx(1.0, abs=0.01)

    def test_zero_precision(self):
        recalls = np.linspace(0, 1, 11)
        precisions = np.zeros(11)
        ap = compute_ap(recalls, precisions)
        assert ap == pytest.approx(0.0, abs=0.01)


class TestPrecisionRecall:
    def test_perfect_match(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=float)
        rec, prec = compute_precision_recall(boxes, np.array([0.9]), boxes, iou_threshold=0.5)
        assert prec[-1] == pytest.approx(1.0, abs=0.01)
        assert rec[-1] == pytest.approx(1.0, abs=0.01)

    def test_no_gt(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=float)
        rec, prec = compute_precision_recall(boxes, np.array([0.9]), np.empty((0, 4)))
        assert prec[-1] == pytest.approx(0.0, abs=0.01)

    def test_no_pred(self):
        rec, prec = compute_precision_recall(
            np.empty((0, 4)), np.empty(0), np.array([[0, 0, 10, 10]])
        )
        assert rec[0] == pytest.approx(0.0, abs=0.01)


class TestDetectionMetrics:
    def test_map_perfect_detector(self):
        classes = ["cat", "dog"]
        metrics = DetectionMetrics(classes)
        gt_boxes = np.array([[0, 0, 10, 10], [50, 50, 100, 100]], dtype=float)
        gt_cls = np.array([0, 1])

        metrics.update(gt_boxes, np.ones(2), gt_cls, gt_boxes, gt_cls)
        result = metrics.compute([0.5])
        assert result.map50 == pytest.approx(1.0, abs=0.05)

    def test_map_empty_predictions(self):
        metrics = DetectionMetrics(["cat"])
        gt_boxes = np.array([[0, 0, 10, 10]], dtype=float)
        metrics.update(np.empty((0, 4)), np.empty(0), np.empty(0, int), gt_boxes, np.array([0]))
        result = metrics.compute([0.5])
        assert result.map50 == pytest.approx(0.0, abs=0.05)
