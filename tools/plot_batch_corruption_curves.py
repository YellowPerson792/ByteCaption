"""
Plot corruption severity curves from batch evaluation JSONs.

Example:
  python tools/plot_batch_corruption_curves.py \
    --input-dir batch_reports/all_models \
    --models ByteCaption_XE ByteCaption_XE_blip ByteCaption_XE_git ByteCaption_XE_gpt5.1 ByteCaption_XE_gemini2.5-flash ByteCaption_XE_gemini2.5-flash ByteCaption_XE_claude-haiku-4.5 ByteCaption_XE_qwen ByteCaption_XE_internvl ByteCaption_XE_glm\
    --corrupt-types rbbf rbsl \
    --plot-metrics CIDEr Bleu_4 SPICE
    
    python tools/plot_batch_corruption_curves.py \
    --input-dir batch_reports/all_models \
    --models ByteCaption_XE ByteCaption_XE_blip ByteCaption_XE_git ByteCaption_XE_gpt5.1 ByteCaption_XE_gemini2.5-flash ByteCaption_XE_gemini2.5-flash ByteCaption_XE_claude-haiku-4.5 ByteCaption_XE_qwen \
    --corrupt-types rbbf rbsl \
    --plot-metrics CIDEr Bleu_4 SPICE
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


LEVEL_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5"]
LEVEL_TO_IDX = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}


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
    parser = argparse.ArgumentParser(description="Plot corruption curves from batch reports")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Batch report directory (contains per-run JSONs or summary.json)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter by model_name (optional)",
    )
    parser.add_argument(
        "--corrupt-types",
        nargs="+",
        default=None,
        help="Filter by corrupt_type (optional)",
    )
    parser.add_argument(
        "--plot-metrics",
        nargs="+",
        default=["CIDEr", "Bleu_4"],
        help="Metrics to plot against corruption level",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write plots (default: input-dir)",
    )
    return parser.parse_args()


def filter_runs(runs: List[Dict], models: List[str], corrupt_types: List[str]) -> List[Dict]:
    filtered = []
    for r in runs:
        model_name = r.get("model_name", "")
        ctype = r.get("corrupt_type", "")
        if models and model_name not in models:
            continue
        if corrupt_types and ctype not in corrupt_types:
            continue
        filtered.append(r)
    return filtered


def plot_curves(runs: List[Dict], metric_keys: List[str], output_dir: Path) -> None:
    level_order = LEVEL_ORDER
    level_to_idx = LEVEL_TO_IDX

    model_names = sorted({r.get("model_name", "") for r in runs if r.get("model_name")})
    corrupt_types = sorted({r.get("corrupt_type", "") for r in runs if r.get("corrupt_type")})

    for metric in metric_keys:
        plt.figure(figsize=(8, 5))
        for model_name in model_names:
            for ctype in corrupt_types:
                subset = [
                    r
                    for r in runs
                    if r.get("model_name") == model_name and r.get("corrupt_type") == ctype
                ]
                if not subset:
                    continue
                subset = sorted(
                    subset,
                    key=lambda x: level_to_idx.get(x.get("corrupt_level", ""), 999),
                )
                xs = [level_to_idx.get(r.get("corrupt_level", ""), 999) for r in subset]
                ys = [r.get("metrics", {}).get(metric, None) for r in subset]
                if all(v is None for v in ys):
                    continue
                plt.plot(xs, ys, marker="o", label=f"{model_name}-{ctype}")
        plt.xticks(list(level_to_idx.values()), level_order)
        plt.xlabel("Corruption level")
        plt.ylabel(metric)
        plt.title(f"{metric} vs corruption level")
        plt.grid(True, alpha=0.3)
        if model_names and corrupt_types:
            plt.legend()
        outfile = output_dir / f"curve_{metric}.png"
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(input_dir)
    if not runs:
        raise SystemExit(f"No run JSONs found in {input_dir}")

    runs = filter_runs(runs, args.models, args.corrupt_types)
    if not runs:
        raise SystemExit("No runs matched the filters")

    plot_curves(runs, args.plot_metrics, output_dir)
    print(f"[PLOT] Saved metric curves to {output_dir}")


if __name__ == "__main__":
    main()
