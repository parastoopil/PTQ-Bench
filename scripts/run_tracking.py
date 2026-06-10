#!/usr/bin/env python3
"""
Multi-object tracking demo using ByteTracker + YOLOv8 FP16 detection.

Usage:
    python scripts/run_tracking.py --source path/to/video.mp4
    python scripts/run_tracking.py --source 0 --model yolov8s
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_ptq_bench.detector import YOLODetector
from yolo_ptq_bench.tracker import ByteTracker
from yolo_ptq_bench.visualizer import ResultVisualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ByteTrack multi-object tracking with YOLOv8 FP16",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", required=True, help="Video path or webcam index.")
    p.add_argument("--model", default="yolov8n")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--high-thresh", type=float, default=0.5,
                   help="ByteTrack high-confidence threshold.")
    p.add_argument("--low-thresh", type=float, default=0.1,
                   help="ByteTrack low-confidence threshold.")
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--save", default=None, help="Save output video to this path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: cannot open source {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    detector = YOLODetector(
        args.model, args.device, args.precision,
        args.conf, 0.45, args.image_size,
    )
    detector.warmup(n_warmup=5)

    tracker = ByteTracker(
        high_thresh=args.high_thresh,
        low_thresh=args.low_thresh,
        max_age=args.max_age,
    )

    frame_count = 0
    fps_buffer: list[float] = []
    class_counts: defaultdict[str, int] = defaultdict(int)

    print(f"Tracking with {args.model} [{args.precision.upper()}] (Q to quit)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        detection_result = detector.detect(frame)
        tracks = tracker.update(detection_result)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        fps_buffer.append(1000.0 / max(elapsed_ms, 1))
        avg_fps = sum(fps_buffer[-30:]) / len(fps_buffer[-30:])

        for t in tracks:
            class_counts[t.class_name] += 1

        annotated = ResultVisualizer.draw_tracks(frame, tracks)

        cv2.putText(
            annotated,
            f"[{args.precision.upper()}] {avg_fps:.0f} FPS | {len(tracks)} tracked",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.putText(
            annotated,
            f"Frame {frame_count} | Total IDs: {tracker._next_id - 1}",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )

        if writer:
            writer.write(annotated)

        cv2.imshow("YOLO-PTQ-Bench Tracker", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    if writer:
        writer.release()
        print(f"Saved tracking video → {args.save}")
    cv2.destroyAllWindows()

    print(f"\nTracking Summary:")
    print(f"  Frames processed : {frame_count}")
    print(f"  Total unique IDs : {tracker._next_id - 1}")
    print(f"  Average FPS      : {sum(fps_buffer)/max(len(fps_buffer),1):.1f}")
    if class_counts:
        print(f"  Detections by class:")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {cnt}")


if __name__ == "__main__":
    main()
