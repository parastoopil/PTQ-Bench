#!/usr/bin/env python3
"""
Single-image or webcam detection demo with FP32 vs FP16 side-by-side comparison.

Usage:
    python scripts/run_detection.py --source path/to/image.jpg
    python scripts/run_detection.py --source path/to/video.mp4 --model yolov8s
    python scripts/run_detection.py --source 0  # webcam
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_ptq_bench.detector import YOLODetector
from yolo_ptq_bench.visualizer import ResultVisualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO detection demo with FP32 vs FP16 comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", required=True,
                   help="Image/video path or webcam index (0).")
    p.add_argument("--model", default="yolov8n")
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--save", default=None,
                   help="Output path to save annotated result.")
    p.add_argument("--compare", action="store_true",
                   help="Save side-by-side FP32 vs FP16 comparison image.")
    return p.parse_args()


def load_image(source: str) -> np.ndarray:
    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {source}")
    return img


def run_image(args: argparse.Namespace) -> None:
    image = load_image(args.source)

    detectors = {
        "FP32": YOLODetector(args.model, args.device, "fp32", args.conf, args.iou, args.image_size),
        "FP16": YOLODetector(args.model, args.device, "fp16", args.conf, args.iou, args.image_size),
    }

    print(f"Warming up detectors...")
    for d in detectors.values():
        d.warmup(n_warmup=3)

    annotated: dict[str, np.ndarray] = {}
    for name, det in detectors.items():
        t0 = time.perf_counter()
        result = det.detect(image)
        latency = (time.perf_counter() - t0) * 1000

        print(
            f"[{name:4s}] {result.num_detections:3d} detections | "
            f"latency: {latency:6.1f} ms | "
            f"fps: {1000/max(latency,1):.0f}"
        )

        ann = ResultVisualizer.draw_detections(
            image,
            result.boxes,
            result.scores,
            result.class_names,
            color=(0, 200, 80) if name == "FP32" else (0, 140, 255),
        )

        cv2.putText(
            ann, f"{name} | {latency:.1f}ms | {result.num_detections} obj",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        annotated[name] = ann

    if args.compare:
        side_by_side = np.concatenate(list(annotated.values()), axis=1)
        out_path = args.save or "results/comparison.jpg"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, side_by_side)
        print(f"Saved comparison → {out_path}")
    elif args.save:
        cv2.imwrite(args.save, annotated["FP32"])
        print(f"Saved → {args.save}")


def run_video(args: argparse.Namespace) -> None:
    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    detector = YOLODetector(args.model, args.device, "fp16", args.conf, args.iou, args.image_size)
    detector.warmup(n_warmup=5)

    fps_buffer: list[float] = []
    print("Running detection (press Q to quit)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        result = detector.detect(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        fps_buffer.append(1000.0 / max(elapsed_ms, 1))
        avg_fps = sum(fps_buffer[-30:]) / len(fps_buffer[-30:])

        ann = ResultVisualizer.draw_detections(frame, result.boxes, result.scores, result.class_names)
        cv2.putText(
            ann,
            f"FP16 | {elapsed_ms:.1f}ms | {avg_fps:.0f} FPS | {result.num_detections} obj",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        cv2.imshow("YOLO-PTQ-Bench Detection", ann)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    src = args.source

    is_image = src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"))
    is_webcam = src.isdigit()

    if is_image:
        run_image(args)
    else:
        run_video(args)


if __name__ == "__main__":
    main()
