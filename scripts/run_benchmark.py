#!/usr/bin/env python3
"""
Full PTQ benchmark: profile FP32, FP16, and INT8 latency/throughput/memory
across one or more YOLO model sizes.

Usage:
    python scripts/run_benchmark.py --models yolov8n yolov8s --device cuda
    python scripts/run_benchmark.py --models yolov8n --n-runs 500 --save-plots
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from yolo_ptq_bench.benchmark import BenchmarkResult, Profiler
from yolo_ptq_bench.detector import YOLODetector
from yolo_ptq_bench.quantizer import PrecisionMode, QuantizedDetector
from yolo_ptq_bench.visualizer import ResultVisualizer

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO PTQ Benchmark: FP32 / FP16 / INT8 on GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s"],
                   help="YOLO model names to benchmark.")
    p.add_argument("--precisions", nargs="+", default=["fp32", "fp16"],
                   choices=["fp32", "fp16"],
                   help="Precision modes to benchmark (INT8 via --include-int8).")
    p.add_argument("--include-int8", action="store_true",
                   help="Include INT8 dynamic quantization (CPU, for comparison).")
    p.add_argument("--device", default="cuda",
                   help="Compute device: 'cuda' or 'cpu'.")
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-runs", type=int, default=300)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--save-plots", action="store_true",
                   help="Generate and save visualisation figures.")
    p.add_argument("--output-dir", default="results",
                   help="Directory for CSV, JSON, and plot outputs.")
    return p.parse_args()


def print_header() -> None:
    console.print(Panel(
        "[bold cyan]YOLO-PTQ-Bench[/bold cyan]\n"
        "[dim]Post-Training Quantization Benchmark for Real-Time Object Detection[/dim]\n"
        f"[dim]PyTorch {torch.__version__} | "
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}[/dim]",
        expand=False,
    ))


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    profiler = Profiler(
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        image_size=args.image_size,
    )

    configs = [
        (model, prec)
        for model in args.models
        for prec in args.precisions
    ]

    results: list[BenchmarkResult] = []

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking...", total=len(configs))

        for model_name, precision in configs:
            progress.update(
                task,
                description=f"[cyan]{model_name}[/cyan] [{precision.upper()}]",
            )

            try:
                detector = YOLODetector(
                    model_name=model_name,
                    device=args.device,
                    precision=precision,
                    image_size=args.image_size,
                )

                # Get model size before profiling
                from yolo_ptq_bench.quantizer import QuantizedDetector
                qd = QuantizedDetector(model_name, args.device)
                if precision == "fp32":
                    _, stats = qd.build_fp32()
                else:
                    _, stats = qd.build_fp16()

                result = profiler.profile_full(
                    detector,
                    model_size_mb=stats.model_size_mb,
                    param_count=stats.param_count,
                )
                results.append(result)

                # Free GPU memory between runs
                del detector
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                console.print(f"[red]  FAILED {model_name}/{precision}: {exc}[/red]")

            progress.advance(task)

    return results


def print_results_table(results: list[BenchmarkResult]) -> None:
    table = Table(
        title="Benchmark Results",
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("Model", style="cyan")
    table.add_column("Precision", justify="center")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_column("FPS", justify="right", style="green")
    table.add_column("Throughput", justify="right")
    table.add_column("Peak Mem (MB)", justify="right")
    table.add_column("Model Size (MB)", justify="right")

    for r in results:
        table.add_row(
            r.model_name,
            r.precision.upper(),
            f"{r.latency.p50:.2f} ± {r.latency.std:.2f}",
            f"{r.latency.p99:.2f}",
            f"{r.latency.fps():.1f}",
            f"{r.throughput.images_per_second:.1f} img/s",
            f"{r.memory.peak_mb:.1f}",
            f"{r.model_size_mb:.1f}",
        )

    console.print(table)


def save_results(results: list[BenchmarkResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = output_dir / f"benchmark_{timestamp}.json"
    data = [r.summary_dict() for r in results]
    json_path.write_text(json.dumps(data, indent=2))
    console.print(f"[dim]Saved JSON → {json_path}[/dim]")

    # CSV
    csv_path = output_dir / f"benchmark_{timestamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    console.print(f"[dim]Saved CSV  → {csv_path}[/dim]")


def main() -> None:
    args = parse_args()
    print_header()

    console.print(f"\n[bold]Models:[/bold] {args.models}")
    console.print(f"[bold]Precisions:[/bold] {args.precisions}")
    console.print(f"[bold]Device:[/bold] {args.device}")
    console.print(f"[bold]Runs:[/bold] {args.n_runs} (warmup: {args.n_warmup})\n")

    results = run_benchmark(args)

    if not results:
        console.print("[red]No results collected. Exiting.[/red]")
        sys.exit(1)

    print_results_table(results)

    output_dir = Path(args.output_dir)
    save_results(results, output_dir)

    if args.save_plots:
        viz = ResultVisualizer(output_dir=str(output_dir / "figures"))
        fig_latency = viz.plot_latency_comparison(results)
        fig_memory = viz.plot_memory_comparison(results)
        console.print(f"[dim]Plots saved → {output_dir / 'figures'}[/dim]")

    speedup_info = []
    fp32_fps = {r.model_name: r.latency.fps() for r in results if r.precision == "fp32"}
    for r in results:
        if r.precision == "fp16" and r.model_name in fp32_fps:
            ratio = r.latency.fps() / fp32_fps[r.model_name]
            speedup_info.append(f"{r.model_name}: FP16 is {ratio:.2f}× faster than FP32")

    if speedup_info:
        console.print("\n[bold green]Speedup Summary:[/bold green]")
        for s in speedup_info:
            console.print(f"  • {s}")


if __name__ == "__main__":
    main()
