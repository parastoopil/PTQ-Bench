"""YOLOv8 detector wrapper with multi-precision GPU inference support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class DetectionResult:
    """Single-image detection output."""
    boxes: np.ndarray        # (N, 4) xyxy
    scores: np.ndarray       # (N,)
    class_ids: np.ndarray    # (N,)
    class_names: List[str]   # (N,)
    inference_ms: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)

    @property
    def num_detections(self) -> int:
        return len(self.boxes)

    def filter_by_class(self, class_name: str) -> "DetectionResult":
        mask = np.array([n == class_name for n in self.class_names])
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            class_names=[n for n, m in zip(self.class_names, mask) if m],
            inference_ms=self.inference_ms,
            image_shape=self.image_shape,
        )


class YOLODetector:
    """
    YOLOv8 detector supporting FP32 and FP16 precision on GPU.

    Example:
        detector = YOLODetector("yolov8n", device="cuda", precision="fp16")
        result = detector.detect(image_array)
    """

    SUPPORTED_MODELS = {
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
        "yolov8n-seg", "yolov8s-seg", "yolov8m-seg",
    }

    def __init__(
        self,
        model_name: str = "yolov8n",
        device: str = "cuda",
        precision: str = "fp32",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
    ) -> None:
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"Unsupported precision '{precision}'. Use 'fp32' or 'fp16'.")

        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size

        self._model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        model_file = f"{self.model_name}.pt"
        self._model = YOLO(model_file)
        self._model.to(self.device)

        if self.precision == "fp16":
            self._model.model.half()

    @property
    def class_names(self) -> dict:
        return self._model.names

    def warmup(self, n_warmup: int = 5) -> None:
        """Run dummy inferences to warm up GPU kernels."""
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        for _ in range(n_warmup):
            self.detect(dummy)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run detection on a single HxWxC BGR/RGB numpy image."""
        t0 = time.perf_counter()
        with torch.inference_mode():
            results = self._model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                verbose=False,
                device=self.device,
                half=(self.precision == "fp16"),
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty(0, dtype=np.float32),
                class_ids=np.empty(0, dtype=np.int32),
                class_names=[],
                inference_ms=elapsed_ms,
                image_shape=image.shape[:2],
            )

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = boxes.conf.cpu().numpy().astype(np.float32)
        cls_ids = boxes.cls.cpu().numpy().astype(np.int32)
        names = [self._model.names[c] for c in cls_ids]

        return DetectionResult(
            boxes=xyxy,
            scores=scores,
            class_ids=cls_ids,
            class_names=names,
            inference_ms=elapsed_ms,
            image_shape=image.shape[:2],
        )

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """Detect on a list of images (batched inference)."""
        return [self.detect(img) for img in images]

    def __repr__(self) -> str:
        return (
            f"YOLODetector(model={self.model_name}, "
            f"precision={self.precision}, device={self.device})"
        )
