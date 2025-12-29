"""
Create robustness visualizations from batch report JSONs.

Outputs:
- Curves per corruption type (metrics vs severity)
- Heatmaps of relative drop vs S0
- Aggregate robustness score bar chart
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


LEVEL_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5"]
LEVEL_TO_IDX = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}
DROP_LEVELS = LEVEL_ORDER[1:]

MODEL_LABELS = {
    "ByteCaption_XE": "ByteCaption",
    "ByteCaption_XE_blip": "BLIP",
    "ByteCaption_XE_git": "GIT",
    "ByteCaption_XE_qwen": "Qwen3-VL-8B",
    "ByteCaption_XE_gpt5.1": "GPT-5.1",
    "ByteCaption_XE_gemini2.5-flash": "Gemini-2.5-flash",
    "ByteCaption_XE_claude-haiku-4.5": "Claude-Haiku-4.5",
    "ByteCaption_XE_internvl": "InternVL3.5-8B",
    "ByteCaption_XE_glm": "GLM",
}

MODEL_ORDER = [
    "ByteCaption_XE",
    "ByteCaption_XE_blip",
    "ByteCaption_XE_git",
    "ByteCaption_XE_qwen",
    "ByteCaption_XE_gpt5.1",
    "ByteCaption_XE_gemini2.5-flash",
    "ByteCaption_XE_claude-haiku-4.5",
    "ByteCaption_XE_internvl",
    "ByteCaption_XE_glm",
]

SCALE_METRICS = {
    "CIDEr": 100.0,
    "SPICE": 100.0,
}


def load_runs(input_dir: Path) -> List[Dict]:
    runs = []
    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "runs" in data:
                return [r for r in data["runs"] if isinstance(r, dict)]
        except Exception:
            pass

    for json_path in sorted(input_dir.glob("*.json")):
        if json_path.name == "summary.json":
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and "metrics" in data:
            runs.append(data)
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize robustness from batch reports")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with per-run JSONs or summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write plots and summaries",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["CIDEr", "SPICE"],
        help="Metrics to visualize",
    )
    return parser.parse_args()


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def iter_models(runs: Iterable[Dict]) -> List[str]:
    present = {r.get("model_name") for r in runs if r.get("model_name")}
    ordered = [m for m in MODEL_ORDER if m in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def iter_corrupt_types(runs: Iterable[Dict]) -> List[str]:
    return sorted({r.get("corrupt_type") for r in runs if r.get("corrupt_type")})


def get_metric_value(run: Dict, metric: str) -> float | None:
    val = (run.get("metrics") or {}).get(metric, None)
    if val is None:
        return None
    scale = SCALE_METRICS.get(metric, 1.0)
    return float(val) * scale


def build_series(
    runs: Iterable[Dict],
    model_name: str,
    corrupt_type: str,
    metric: str,
) -> List[float]:
    subset = [
        r
        for r in runs
        if r.get("model_name") == model_name and r.get("corrupt_type") == corrupt_type
    ]
    by_level = {r.get("corrupt_level"): get_metric_value(r, metric) for r in subset}
    return [by_level.get(level, np.nan) for level in LEVEL_ORDER]


def plot_curves(runs: List[Dict], metrics: List[str], output_dir: Path) -> None:
    models = iter_models(runs)
    corrupt_types = iter_corrupt_types(runs)
    if not models or not corrupt_types:
        return

    base_colors = plt.get_cmap("tab10").colors
    color_map = {}
    color_idx = 0
    for model in models:
        if model == "ByteCaption_XE":
            color_map[model] = "#d1495b"
        else:
            color_map[model] = base_colors[color_idx % len(base_colors)]
            color_idx += 1

    for corrupt_type in corrupt_types:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(8.5, 3.4 * len(metrics)), sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            for model in models:
                ys = build_series(runs, model, corrupt_type, metric)
                if all(np.isnan(y) for y in ys):
                    continue
                is_highlight = model == "ByteCaption_XE"
                ax.plot(
                    list(range(len(LEVEL_ORDER))),
                    ys,
                    marker="o",
                    linewidth=2.4 if is_highlight else 1.4,
                    alpha=1.0 if is_highlight else 0.65,
                    color=color_map[model],
                    label=model_label(model),
                )
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"{metric} vs severity ({corrupt_type.upper()})")
        axes[-1].set_xticks(list(range(len(LEVEL_ORDER))), LEVEL_ORDER)
        axes[-1].set_xlabel("Corruption severity")
        axes[0].legend(ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
        outfile = output_dir / f"curves_{corrupt_type}.png"
        fig.savefig(outfile, dpi=200)
        plt.close(fig)


def compute_drop_matrix(
    runs: Iterable[Dict],
    metric: str,
    corrupt_type: str,
    models: List[str],
) -> np.ndarray:
    matrix = np.full((len(models), len(DROP_LEVELS)), np.nan)
    for i, model in enumerate(models):
        series = build_series(runs, model, corrupt_type, metric)
        s0 = series[0]
        if s0 is None or np.isnan(s0) or s0 == 0:
            continue
        for j, level in enumerate(DROP_LEVELS, start=1):
            sx = series[j]
            if sx is None or np.isnan(sx):
                continue
            matrix[i, j - 1] = (s0 - sx) / s0 * 100.0
    return matrix


def plot_drop_heatmaps(runs: List[Dict], metrics: List[str], output_dir: Path) -> None:
    models = iter_models(runs)
    corrupt_types = iter_corrupt_types(runs)
    if not models or not corrupt_types:
        return

    y_labels = [model_label(m) for m in models]
    x_labels = DROP_LEVELS

    for corrupt_type in corrupt_types:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 5.2), sharey=True)
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            matrix = compute_drop_matrix(runs, metric, corrupt_type, models)
            im = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
            ax.set_title(f"Drop vs S0 ({metric})")
            ax.set_xticks(range(len(x_labels)), x_labels)
            ax.set_yticks(range(len(y_labels)), y_labels)
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="x", labelsize=8)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("% drop")
        fig.suptitle(f"Robustness drop heatmap ({corrupt_type.upper()})")
        fig.tight_layout()
        outfile = output_dir / f"heatmap_drop_{corrupt_type}.png"
        fig.savefig(outfile, dpi=200)
        plt.close(fig)


def compute_robustness_scores(
    runs: Iterable[Dict],
    metrics: List[str],
) -> List[Tuple[str, float]]:
    models = iter_models(runs)
    corrupt_types = iter_corrupt_types(runs)
    results = []
    for model in models:
        ratios = []
        for corrupt_type in corrupt_types:
            for metric in metrics:
                series = build_series(runs, model, corrupt_type, metric)
                s0 = series[0]
                if s0 is None or np.isnan(s0) or s0 == 0:
                    continue
                for sx in series:
                    if sx is None or np.isnan(sx):
                        continue
                    ratios.append(sx / s0)
        score = float(np.mean(ratios)) if ratios else float("nan")
        results.append((model, score))
    return results


def plot_robustness_scores(
    runs: List[Dict],
    metrics: List[str],
    output_dir: Path,
) -> Path:
    results = compute_robustness_scores(runs, metrics)
    results = [r for r in results if not np.isnan(r[1])]
    results.sort(key=lambda x: x[1], reverse=True)
    if not results:
        return output_dir / "robustness_score.png"

    labels = [model_label(m) for m, _ in results]
    scores = [s * 100.0 for _, s in results]
    colors = ["#d1495b" if m == "ByteCaption_XE" else "#6c7a89" for m, _ in results]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(labels, scores, color=colors)
    ax.set_ylabel("Robustness score (avg % of S0)")
    ax.set_ylim(0, max(scores) * 1.15)
    ax.set_title("Aggregate robustness score")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    outfile = output_dir / "robustness_score.png"
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    return outfile


def write_summary_csv(
    runs: List[Dict],
    metrics: List[str],
    output_dir: Path,
) -> Path:
    results = compute_robustness_scores(runs, metrics)
    out_path = output_dir / "robustness_summary.csv"
    lines = ["model,robustness_score"]
    for model, score in results:
        lines.append(f"{model_label(model)},{score:.6f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(input_dir)
    if not runs:
        raise SystemExit(f"No run JSONs found in {input_dir}")

    metrics = args.metrics

    plot_curves(runs, metrics, output_dir)
    plot_drop_heatmaps(runs, metrics, output_dir)
    plot_robustness_scores(runs, metrics, output_dir)
    write_summary_csv(runs, metrics, output_dir)

    print(f"[VIS] Wrote figures and summaries to {output_dir}")


if __name__ == "__main__":
    main()
