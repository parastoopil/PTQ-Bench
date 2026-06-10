"""Tests for Profiler and BenchmarkResult."""

import numpy as np
import pytest

from yolo_ptq_bench.benchmark import BenchmarkResult, LatencyStats, Profiler
from yolo_ptq_bench.detector import YOLODetector


@pytest.fixture(scope="module")
def detector(device):
    return YOLODetector("yolov8n", device=device, precision="fp32")


@pytest.fixture(scope="module")
def profiler():
    return Profiler(n_warmup=2, n_runs=10, image_size=640)


class TestLatencyStats:
    def test_from_samples(self):
        samples = list(range(1, 101))
        stats = LatencyStats.from_samples(samples)
        assert stats.n_runs == 100
        assert stats.mean == pytest.approx(50.5, abs=0.1)
        assert stats.min == 1.0
        assert stats.max == 100.0
        assert stats.p50 == pytest.approx(50.5, abs=1.0)
        assert stats.p99 == pytest.approx(99.0, abs=1.0)

    def test_fps(self):
        stats = LatencyStats(mean=10.0, std=0, p50=10, p90=12, p99=15, min=8, max=16, n_runs=10)
        assert stats.fps() == pytest.approx(100.0, abs=0.1)


class TestProfiler:
    def test_profile_latency_returns_stats(self, profiler, detector):
        stats = profiler.profile_latency(detector)
        assert stats.n_runs == 10
        assert stats.mean > 0
        assert stats.p99 >= stats.p50 >= stats.min

    def test_profile_memory_returns_stats(self, profiler, detector):
        stats = profiler.profile_memory(detector)
        assert stats.peak_mb >= 0

    def test_profile_full_returns_result(self, profiler, detector):
        result = profiler.profile_full(detector, model_size_mb=10.0, param_count=123456)
        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "yolov8n"
        assert result.precision == "fp32"
        assert result.latency.fps() > 0

    def test_summary_dict_keys(self, profiler, detector):
        result = profiler.profile_full(detector)
        d = result.summary_dict()
        for key in ["latency_p50_ms", "fps", "peak_memory_mb"]:
            assert key in d
