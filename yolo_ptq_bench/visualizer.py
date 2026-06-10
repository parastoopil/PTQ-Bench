"""Publication-quality visualizations for benchmark results and detections."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from yolo_ptq_bench.benchmark import BenchmarkResult, LatencyStats


PALETTE = {
    "FP32": "#4C72B0",
    "FP16": "#DD8452",
    "INT8": "#55A868",
}

STYLE_KWARGS = dict(style="whitegrid", context="paper", font_scale=1.2)


def _apply_style() -> None:
    sns.set_theme(**STYLE_KWARGS)
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


class ResultVisualizer:
    """Generate and save plots from BenchmarkResult collections."""

    def __init__(self, output_dir: str = "results/figures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_latency_comparison(
        self,
        results: List[BenchmarkResult],
        filename: str = "latency_comparison.png",
    ) -> Path:
        """Bar chart comparing P50/P99 latency across model × precision."""
        _apply_style()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        labels = [f"{r.model_name}\n{r.precision.upper()}" for r in results]
        p50 = [r.latency.p50 for r in results]
        p99 = [r.latency.p99 for r in results]
        colors = [PALETTE.get(r.precision.upper(), "#888888") for r in results]

        for ax, values, title in zip(axes, [p50, p99], ["P50 Latency (ms)", "P99 Latency (ms)"]):
            bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("Latency (ms)")
            ax.set_xlabel("Model / Precision")
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.05,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9,
                )

        patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
        fig.legend(handles=patches, loc="upper right", title="Precision")
        fig.suptitle("Inference Latency: FP32 vs FP16 vs INT8", fontsize=14, fontweight="bold")
        fig.tight_layout()

        out = self.output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_throughput_vs_accuracy(
        self,
        results: List[BenchmarkResult],
        map50_values: Dict[str, float],
        filename: str = "accuracy_efficiency.png",
    ) -> Path:
        """
        Accuracy–efficiency Pareto plot: FPS on x-axis, mAP@50 on y-axis.
        Each point is a (model, precision) configuration.
        """
        _apply_style()
        fig, ax = plt.subplots(figsize=(9, 6))

        for r in results:
            key = f"{r.model_name}_{r.precision}"
            map_val = map50_values.get(key, None)
            if map_val is None:
                continue
            fps = r.latency.fps()
            color = PALETTE.get(r.precision.upper(), "#888888")
            ax.scatter(fps, map_val, color=color, s=120, zorder=5)
            ax.annotate(
                f"{r.model_name}\n{r.precision.upper()}",
                (fps, map_val),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )

        patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
        ax.legend(handles=patches, title="Precision")
        ax.set_xlabel("Throughput (FPS)", fontweight="bold")
        ax.set_ylabel("mAP@50 (COCO)", fontweight="bold")
        ax.set_title("Accuracy–Efficiency Trade-off", fontsize=14, fontweight="bold")
        fig.tight_layout()

        out = self.output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_memory_comparison(
        self,
        results: List[BenchmarkResult],
        filename: str = "memory_comparison.png",
    ) -> Path:
        """Stacked bar chart: GPU memory vs model size per configuration."""
        _apply_style()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        labels = [f"{r.model_name}\n{r.precision.upper()}" for r in results]
        peak_mem = [r.memory.peak_mb for r in results]
        model_sizes = [r.model_size_mb for r in results]
        colors = [PALETTE.get(r.precision.upper(), "#888888") for r in results]

        for ax, values, title, ylabel in zip(
            axes,
            [peak_mem, model_sizes],
            ["Peak GPU Memory", "Model Size on Disk"],
            ["Memory (MB)", "Size (MB)"],
        ):
            bars = ax.bar(labels, values, color=colors, edgecolor="white")
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel(ylabel)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.5,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=9,
                )

        patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
        fig.legend(handles=patches, loc="upper right", title="Precision")
        fig.suptitle("Memory Footprint by Precision", fontsize=14, fontweight="bold")
        fig.tight_layout()

        out = self.output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_latency_distribution(
        self,
        latency_samples: Dict[str, List[float]],
        filename: str = "latency_distribution.png",
    ) -> Path:
        """Violin/box plot of full latency distributions."""
        _apply_style()
        fig, ax = plt.subplots(figsize=(10, 5))

        labels, data, colors = [], [], []
        for label, samples in latency_samples.items():
            labels.append(label)
            data.append(samples)
            precision = label.split("_")[-1].upper() if "_" in label else "FP32"
            colors.append(PALETTE.get(precision, "#888888"))

        parts = ax.violinplot(data, positions=range(len(labels)), showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_xlabel("Model / Precision")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Distribution Across Precision Modes", fontweight="bold")
        fig.tight_layout()

        out = self.output_dir / filename
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_names: List[str],
        color: Tuple[int, int, int] = (0, 200, 80),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and labels onto a BGR image (in-place copy)."""
        import cv2
        img = image.copy()
        for box, score, name in zip(boxes, scores, class_names):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label = f"{name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    @staticmethod
    def draw_tracks(
        image: np.ndarray,
        tracks: list,
        trail_length: int = 20,
    ) -> np.ndarray:
        """Draw tracked objects with ID labels and motion trails."""
        import cv2
        img = image.copy()
        rng = np.random.default_rng(42)
        id_colors: Dict[int, Tuple[int, int, int]] = {}

        for track in tracks:
            tid = track.track_id
            if tid not in id_colors:
                c = rng.integers(80, 230, size=3).tolist()
                id_colors[tid] = tuple(c)
            color = id_colors[tid]

            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"#{tid} {track.class_name} {track.score:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            trail = track.trail[-trail_length:]
            for i in range(1, len(trail)):
                cx_prev = int((trail[i - 1][0] + trail[i - 1][2]) / 2)
                cy_prev = int((trail[i - 1][1] + trail[i - 1][3]) / 2)
                cx_curr = int((trail[i][0] + trail[i][2]) / 2)
                cy_curr = int((trail[i][1] + trail[i][3]) / 2)
                alpha = i / len(trail)
                faded = tuple(int(c * alpha) for c in color)
                cv2.line(img, (cx_prev, cy_prev), (cx_curr, cy_curr), faded, 1)

        return img
