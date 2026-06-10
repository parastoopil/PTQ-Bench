"""Detection evaluation metrics: mAP, precision, recall, F1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ClassMetrics:
    class_name: str
    ap50: float        # AP at IoU=0.50
    ap50_95: float     # AP averaged over IoU=[0.50:0.05:0.95]
    precision: float
    recall: float
    f1: float


@dataclass
class DatasetMetrics:
    map50: float
    map50_95: float
    per_class: Dict[str, ClassMetrics]

    @property
    def mean_precision(self) -> float:
        if not self.per_class:
            return 0.0
        return float(np.mean([c.precision for c in self.per_class.values()]))

    @property
    def mean_recall(self) -> float:
        if not self.per_class:
            return 0.0
        return float(np.mean([c.recall for c in self.per_class.values()]))


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of xyxy bounding boxes.

    Returns:
        IoU matrix of shape (len(boxes_a), len(boxes_b)).
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

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

    return (inter_area / np.clip(union_area, 1e-6, None)).astype(np.float32)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using the 101-point interpolation (COCO-style)."""
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    ap = float(np.mean(np.interp(recall_thresholds, recalls, precisions)))
    return ap


def compute_precision_recall(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve for a single class on a single image.

    Args:
        pred_boxes:  (M, 4) predicted xyxy boxes.
        pred_scores: (M,) confidence scores.
        gt_boxes:    (G, 4) ground-truth xyxy boxes.
        iou_threshold: IoU threshold for a TP match.

    Returns:
        (recalls, precisions) both sorted by descending score.
    """
    if len(pred_boxes) == 0:
        return np.array([0.0]), np.array([0.0])

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    tps = np.zeros(len(pred_boxes))
    fps = np.zeros(len(pred_boxes))

    if len(gt_boxes) > 0:
        iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
        for i, iou_row in enumerate(iou_mat):
            best_j = int(np.argmax(iou_row))
            if iou_row[best_j] >= iou_threshold and not matched_gt[best_j]:
                tps[i] = 1
                matched_gt[best_j] = True
            else:
                fps[i] = 1
    else:
        fps[:] = 1

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    n_gt = max(len(gt_boxes), 1)

    recalls = cum_tp / n_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1)

    return recalls, precisions


class DetectionMetrics:
    """
    Accumulate predictions and ground-truth across a dataset, then compute
    mAP@50 and mAP@50:95 per COCO evaluation protocol.
    """

    def __init__(self, class_names: List[str]) -> None:
        self.class_names = class_names
        self._preds: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            c: [] for c in class_names
        }
        self._gts: Dict[str, List[np.ndarray]] = {c: [] for c in class_names}

    def update(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ) -> None:
        """Add predictions and ground-truth for a single image."""
        for cls_id, cls_name in enumerate(self.class_names):
            mask_pred = pred_classes == cls_id
            mask_gt = gt_classes == cls_id
            self._preds[cls_name].append(
                (pred_boxes[mask_pred], pred_scores[mask_pred], pred_boxes[mask_pred])
            )
            self._gts[cls_name].append(gt_boxes[mask_gt])

    def compute(self, iou_thresholds: Optional[List[float]] = None) -> DatasetMetrics:
        """Compute mAP across all classes and IoU thresholds."""
        if iou_thresholds is None:
            iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50:0.95

        per_class: Dict[str, ClassMetrics] = {}
        all_ap50: List[float] = []
        all_ap50_95: List[float] = []

        for cls_name in self.class_names:
            preds = self._preds[cls_name]
            gts = self._gts[cls_name]

            all_pred_boxes = np.concatenate([p[0] for p in preds if len(p[0]) > 0], axis=0) if any(len(p[0]) > 0 for p in preds) else np.empty((0, 4))
            all_pred_scores = np.concatenate([p[1] for p in preds if len(p[1]) > 0]) if any(len(p[1]) > 0 for p in preds) else np.empty(0)
            all_gt_boxes = np.concatenate([g for g in gts if len(g) > 0], axis=0) if any(len(g) > 0 for g in gts) else np.empty((0, 4))

            ap_per_iou: List[float] = []
            for iou_thr in iou_thresholds:
                rec, prec = compute_precision_recall(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thr)
                ap_per_iou.append(compute_ap(rec, prec))

            ap50 = ap_per_iou[0]
            ap50_95 = float(np.mean(ap_per_iou))

            rec50, prec50 = compute_precision_recall(all_pred_boxes, all_pred_scores, all_gt_boxes, 0.5)
            best_prec = float(prec50[-1]) if len(prec50) > 0 else 0.0
            best_rec = float(rec50[-1]) if len(rec50) > 0 else 0.0
            f1 = 2 * best_prec * best_rec / max(best_prec + best_rec, 1e-6)

            per_class[cls_name] = ClassMetrics(
                class_name=cls_name,
                ap50=ap50,
                ap50_95=ap50_95,
                precision=best_prec,
                recall=best_rec,
                f1=f1,
            )
            all_ap50.append(ap50)
            all_ap50_95.append(ap50_95)

        return DatasetMetrics(
            map50=float(np.mean(all_ap50)),
            map50_95=float(np.mean(all_ap50_95)),
            per_class=per_class,
        )
