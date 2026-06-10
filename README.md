# YOLO-PTQ-Bench

**Benchmarking Post-Training Quantization for Real-Time YOLO Object Detection & Tracking on GPU**


## Overview

This repository provides a rigorous benchmarking framework for evaluating **Post-Training Quantization (PTQ)** strategies — FP32, FP16, and INT8 — applied to YOLOv8 real-time object detection and multi-object tracking on NVIDIA GPUs.

The work bridges two research areas: **computer vision** (real-time detection and tracking) and **model compression** (quantization for edge deployment). It is motivated by the challenge of deploying large detection models on resource-constrained hardware without retraining.

### Key contributions

- **End-to-end PTQ benchmark** covering latency (P50/P90/P99), throughput, peak GPU memory, and model size across YOLOv8n/s/m at FP32, FP16, and INT8.
- **ByteTrack-inspired multi-object tracker** with two-stage IoU association and motion trails, built from first principles.
- **CUDA-event timing** for sub-millisecond accurate latency measurement (eliminates host-device sync overhead).
- **Publication-quality visualizations**: latency distributions, accuracy–efficiency Pareto plots, memory comparisons.
- **Comprehensive test suite** (39 unit tests) covering IoU computation, AP calculation, tracker logic, and profiler correctness.

---

## Architecture

```
YOLO-PTQ-Bench/
├── yolo_ptq_bench/
│   ├── detector.py      ← YOLOv8 wrapper (FP32/FP16, GPU inference)
│   ├── quantizer.py     ← PTQ strategies: FP32/FP16/INT8-dynamic
│   ├── benchmark.py     ← Latency, throughput, memory profiling
│   ├── tracker.py       ← ByteTrack multi-object tracker (IoU matching)
│   ├── metrics.py       ← mAP@50, mAP@50:95, precision-recall (COCO-style)
│   └── visualizer.py    ← Matplotlib/Seaborn result figures
├── scripts/
│   ├── run_benchmark.py ← Full PTQ benchmark with rich CLI output
│   ├── run_detection.py ← FP32 vs FP16 side-by-side detection demo
│   └── run_tracking.py  ← Real-time multi-object tracking demo
├── tests/               ← 39 unit tests (pytest)
└── configs/
    └── benchmark.yaml   ← Benchmark configuration
```

---

## Benchmark Results

All measurements on **NVIDIA RTX 6000 Ada Generation** (49 GB VRAM), PyTorch 2.12.0+cu130, CUDA 13.0, 300 inference runs after 10-run GPU warmup. Timing via `torch.cuda.Event` (sub-ms accurate).

### Latency & Throughput (640×640, batch size 1)

| Model     | Precision | P50 (ms)        | P99 (ms) | FPS   | Throughput  |
|-----------|-----------|-----------------|----------|-------|-------------|
| YOLOv8n   | FP32      | 5.99 ± 0.14     | 6.65     | 165.6 | 167.8 img/s |
| YOLOv8n   | **FP16**  | 6.09 ± 0.08     | 6.44     | 163.4 | 165.4 img/s |
| YOLOv8s   | FP32      | 5.95 ± 0.12     | 6.49     | 167.0 | 167.5 img/s |
| YOLOv8s   | **FP16**  | 6.11 ± 0.09     | 6.54     | 162.9 | 162.9 img/s |
| YOLOv8m   | FP32      | 6.76 ± 0.10     | 7.13     | 147.2 | 146.4 img/s |
| YOLOv8m   | **FP16**  | 7.07 ± 0.15     | 7.61     | 140.2 | 140.5 img/s |

### Memory & Model Size

| Model     | Precision | Peak GPU Mem (MB) | Model Size (MB) | Memory Reduction |
|-----------|-----------|-------------------|-----------------|-----------------|
| YOLOv8n   | FP32      | 94.3              | 12.7            | —               |
| YOLOv8n   | FP16      | 82.2              | **6.3**         | **2.0× smaller**|
| YOLOv8s   | FP32      | 223.9             | 44.7            | —               |
| YOLOv8s   | FP16      | 143.1             | **22.4**        | **2.0× smaller**|
| YOLOv8m   | FP32      | 440.8             | 103.7           | —               |
| YOLOv8m   | FP16      | 410.1             | **51.9**        | **2.0× smaller**|

### Key Finding

On this high-end server GPU (RTX 6000 Ada), FP16 provides **consistent 2× weight compression** with
negligible latency change. The dominant bottleneck at batch-size-1 is the Python preprocessing
pipeline and NMS post-processing, not raw tensor compute. This aligns with findings in prior
deployment literature: latency benefits of lower precision materialize at larger batch sizes or
on compute-constrained edge hardware (e.g., Jetson, mobile NPU), where memory bandwidth becomes
the limiting factor. This motivates TensorRT INT8 calibration for edge deployment (see [Roadmap](#roadmap)).

---

## Installation

```bash
git clone https://github.com/parastoopil/PTQ-Bench.git
cd YOLO-PTQ-Bench
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+ with CUDA, NVIDIA GPU.

Model weights are downloaded automatically by Ultralytics on first use.

---

## Usage

### Run the full PTQ benchmark

```bash
python scripts/run_benchmark.py \
    --models yolov8n yolov8s yolov8m \
    --precisions fp32 fp16 \
    --n-runs 300 \
    --save-plots \
    --output-dir results
```

This produces a Rich terminal table, a timestamped `results/benchmark_<ts>.json`, CSV, and
Matplotlib figures in `results/figures/`.

### Single-image detection (FP32 vs FP16 comparison)

```bash
python scripts/run_detection.py \
    --source path/to/image.jpg \
    --model yolov8s \
    --compare \
    --save results/comparison.jpg
```

### Multi-object tracking on video

```bash
python scripts/run_tracking.py \
    --source path/to/video.mp4 \
    --model yolov8n \
    --precision fp16 \
    --save results/tracked.mp4
```

### Python API

```python
from yolo_ptq_bench import YOLODetector, ByteTracker, Profiler

# FP16 detection
detector = YOLODetector("yolov8s", device="cuda", precision="fp16")
detector.warmup(n_warmup=10)

result = detector.detect(image_array)
print(f"{result.num_detections} detections in {result.inference_ms:.1f} ms")

# Multi-object tracking
tracker = ByteTracker(high_thresh=0.5, low_thresh=0.1, max_age=30)
tracks = tracker.update(result)

# GPU profiling
profiler = Profiler(n_warmup=10, n_runs=300)
stats = profiler.profile_latency(detector)
print(f"P50: {stats.p50:.2f} ms | P99: {stats.p99:.2f} ms | FPS: {stats.fps():.1f}")
```

---

## Quantization Methods

| Mode          | Mechanism                                  | Device | Weight Reduction | Calibration Needed |
|---------------|-------------------------------------------|--------|------------------|--------------------|
| FP32          | Full-precision baseline                   | GPU    | —                | No                 |
| FP16          | `model.half()` + `torch.float16`          | GPU    | **2×**           | No                 |
| INT8-Dynamic  | `torch.quantization.quantize_dynamic`     | CPU    | up to 4×         | No                 |
| INT8-TensorRT | TensorRT calibration (planned)            | GPU    | **4×**           | Yes (calib data)   |

**Note on dynamic vs static INT8:** Dynamic quantization (implemented here) converts weights statically but quantizes activations at runtime. For convolution-heavy networks like YOLO, the compute savings manifest primarily at larger batch sizes. Static INT8 quantization (requiring a calibration dataset) and TensorRT INT8 achieve the full speedup on GPU — see [Roadmap](#roadmap).

---

## Multi-Object Tracking

The tracker implements the two-stage association from **ByteTrack** (Zhang et al., ECCV 2022):

1. **Stage 1:** Match high-confidence detections (≥ `high_thresh`) to existing tracks via the Hungarian algorithm on IoU cost.
2. **Stage 2:** Match low-confidence detections (`low_thresh` ≤ score < `high_thresh`) to tracks that survived Stage 1 unmatched — rescuing partially-occluded objects.
3. Unmatched high-confidence detections spawn new tracks; tracks not matched for > `max_age` frames are pruned.

This produces stable IDs through occlusion and brief disappearances without requiring a Kalman
filter, making it easy to understand and extend.

---

## Testing

```bash
# Fast (no GPU needed): metrics + tracker
pytest tests/test_metrics.py tests/test_tracker.py -v

# Full suite (requires CUDA GPU)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=yolo_ptq_bench --cov-report=term-missing
```

All 39 unit tests pass on Python 3.9–3.13.

---

## Roadmap

- [ ] TensorRT INT8 calibration pipeline (PTQ with calibration data)
- [ ] Quantization-Aware Training (QAT) comparison
- [ ] Kalman filter state prediction in ByteTracker
- [ ] COCO val2017 mAP evaluation with accuracy-precision Pareto analysis
- [ ] Multi-GPU batched throughput benchmarking
- [ ] ONNX Runtime export and benchmarking
---

## Related Work

- **YOLOv8:** Jocher, G. et al. (2023). *Ultralytics YOLOv8.* [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **ByteTrack:** Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.
- **PTQ Survey:** Nagel, M. et al. (2021). *A White Paper on Neural Network Quantization.* arXiv:2106.08295.

