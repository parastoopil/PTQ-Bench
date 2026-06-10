"""GPU latency, throughput, and memory profiling for YOLO inference."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from yolo_ptq_bench.detector import YOLODetector


@dataclass
class LatencyStats:
    """Latency statistics in milliseconds."""
    mean: float
    std: float
    p50: float
    p90: float
    p99: float
    min: float
    max: float
    n_runs: int

    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyStats":
        arr = np.array(samples)
        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p99=float(np.percentile(arr, 99)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            n_runs=len(samples),
        )

    def fps(self) -> float:
        return 1000.0 / max(self.mean, 1e-6)


@dataclass
class MemoryStats:
    """GPU memory statistics in megabytes."""
    peak_mb: float
    allocated_mb: float
    reserved_mb: float


@dataclass
class ThroughputStats:
    """Batch throughput in images/second."""
    images_per_second: float
    batch_size: int
    total_images: int
    total_time_s: float


@dataclass
class BenchmarkResult:
    """Complete benchmark output for a single (model, precision) configuration."""
    model_name: str
    precision: str
    device: str
    image_size: int
    latency: LatencyStats
    memory: MemoryStats
    throughput: ThroughputStats
    model_size_mb: float
    param_count: int

    def summary_dict(self) -> Dict[str, float]:
        return {
            "model": self.model_name,
            "precision": self.precision,
            "latency_p50_ms": self.latency.p50,
            "latency_p99_ms": self.latency.p99,
            "fps": self.latency.fps(),
            "throughput_img_s": self.throughput.images_per_second,
            "peak_memory_mb": self.memory.peak_mb,
            "model_size_mb": self.model_size_mb,
        }


class Profiler:
    """
    Profiles YOLO inference along three axes: latency, memory, and throughput.

    Uses CUDA events for sub-millisecond accurate GPU timing, avoiding
    host-device synchronization overhead that skews perf measurements.
    """

    def __init__(
        self,
        n_warmup: int = 10,
        n_runs: int = 200,
        image_size: int = 640,
    ) -> None:
        self.n_warmup = n_warmup
        self.n_runs = n_runs
        self.image_size = image_size

    def _make_dummy_image(self) -> np.ndarray:
        return np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)

    def profile_latency(self, detector: YOLODetector) -> LatencyStats:
        """Measure per-image latency using CUDA events."""
        dummy = self._make_dummy_image()

        for _ in range(self.n_warmup):
            detector.detect(dummy)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies: List[float] = []
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(self.n_runs):
                start_event.record()
                detector.detect(dummy)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
        else:
            for _ in range(self.n_runs):
                t0 = time.perf_counter()
                detector.detect(dummy)
                latencies.append((time.perf_counter() - t0) * 1000.0)

        return LatencyStats.from_samples(latencies)

    def profile_memory(self, detector: YOLODetector) -> MemoryStats:
        """Measure peak GPU memory during inference."""
        if not torch.cuda.is_available():
            return MemoryStats(peak_mb=0.0, allocated_mb=0.0, reserved_mb=0.0)

        dummy = self._make_dummy_image()
        detector.warmup(n_warmup=3)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        for _ in range(10):
            detector.detect(dummy)

        torch.cuda.synchronize()
        return MemoryStats(
            peak_mb=torch.cuda.max_memory_allocated() / 1e6,
            allocated_mb=torch.cuda.memory_allocated() / 1e6,
            reserved_mb=torch.cuda.memory_reserved() / 1e6,
        )

    def profile_throughput(
        self,
        detector: YOLODetector,
        batch_sizes: Tuple[int, ...] = (1,),
    ) -> ThroughputStats:
        """Measure sustained throughput over a large number of images."""
        total_images = max(self.n_runs, 500)
        dummy = self._make_dummy_image()

        detector.warmup(n_warmup=self.n_warmup)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        for _ in range(total_images):
            detector.detect(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        return ThroughputStats(
            images_per_second=total_images / elapsed,
            batch_size=1,
            total_images=total_images,
            total_time_s=elapsed,
        )

    def profile_full(self, detector: YOLODetector, model_size_mb: float = 0.0, param_count: int = 0) -> BenchmarkResult:
        """Run all profiling passes and return a BenchmarkResult."""
        latency = self.profile_latency(detector)
        memory = self.profile_memory(detector)
        throughput = self.profile_throughput(detector)

        return BenchmarkResult(
            model_name=detector.model_name,
            precision=detector.precision,
            device=str(detector.device),
            image_size=detector.image_size,
            latency=latency,
            memory=memory,
            throughput=throughput,
            model_size_mb=model_size_mb,
            param_count=param_count,
        )
