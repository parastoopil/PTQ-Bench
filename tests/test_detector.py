"""Tests for YOLODetector."""

import numpy as np
import pytest

from yolo_ptq_bench.detector import DetectionResult, YOLODetector


@pytest.fixture(scope="module")
def fp32_detector(device):
    return YOLODetector("yolov8n", device=device, precision="fp32")


@pytest.fixture(scope="module")
def fp16_detector(device):
    return YOLODetector("yolov8n", device=device, precision="fp16")


class TestDetectionResult:
    def test_num_detections_empty(self):
        r = DetectionResult(
            boxes=np.empty((0, 4)), scores=np.empty(0),
            class_ids=np.empty(0, dtype=int), class_names=[],
        )
        assert r.num_detections == 0

    def test_num_detections_nonempty(self):
        r = DetectionResult(
            boxes=np.zeros((3, 4)), scores=np.ones(3),
            class_ids=np.zeros(3, dtype=int), class_names=["a", "b", "c"],
        )
        assert r.num_detections == 3

    def test_filter_by_class(self):
        r = DetectionResult(
            boxes=np.zeros((3, 4)), scores=np.ones(3),
            class_ids=np.array([0, 1, 0]), class_names=["cat", "dog", "cat"],
        )
        cats = r.filter_by_class("cat")
        assert cats.num_detections == 2
        assert all(n == "cat" for n in cats.class_names)


class TestYOLODetector:
    def test_invalid_precision(self, device):
        with pytest.raises(ValueError, match="fp32"):
            YOLODetector("yolov8n", device=device, precision="bf16")

    def test_detect_returns_result(self, fp32_detector, dummy_image):
        result = fp32_detector.detect(dummy_image)
        assert isinstance(result, DetectionResult)
        assert result.image_shape == (640, 640)
        assert result.inference_ms > 0

    def test_boxes_shape(self, fp32_detector, dummy_image):
        result = fp32_detector.detect(dummy_image)
        if result.num_detections > 0:
            assert result.boxes.shape[1] == 4
            assert result.scores.shape == (result.num_detections,)
            assert result.class_ids.shape == (result.num_detections,)

    def test_fp16_produces_result(self, fp16_detector, dummy_image):
        result = fp16_detector.detect(dummy_image)
        assert isinstance(result, DetectionResult)

    def test_scores_in_range(self, fp32_detector, dummy_image):
        result = fp32_detector.detect(dummy_image)
        if result.num_detections > 0:
            assert np.all(result.scores >= 0.0)
            assert np.all(result.scores <= 1.0)

    def test_batch_detect_length(self, fp32_detector, batch_images):
        results = fp32_detector.detect_batch(batch_images)
        assert len(results) == len(batch_images)

    def test_warmup_does_not_raise(self, fp32_detector):
        fp32_detector.warmup(n_warmup=2)

    def test_repr(self, fp32_detector):
        r = repr(fp32_detector)
        assert "yolov8n" in r
        assert "fp32" in r
