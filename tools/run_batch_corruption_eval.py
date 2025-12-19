"""
Batch corruption evaluation helper.

Example:
  python tools/run_batch_corruption_eval.py \
    --models PureT/experiments/ByteCaption_XE PureT/experiments/ByteCaption_XE_blip \
    --corrupt-types rbbf rbsl \
    --corrupt-levels S0 S1 S2 S3 S4 S5 \
    --test-samples 0
"""

import argparse
import json
import logging
import os
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "PureT") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "PureT"))

from lib.config import cfg, cfg_from_file  # noqa: E402
from PureT.main_test import Tester  # noqa: E402
from corenet.data.transforms import jpeg_corruption  # noqa: E402


def reset_logger_handlers():
    """Detach existing handlers to avoid duplicated logs across runs."""
    logger = logging.getLogger(cfg.LOGGER_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def load_config(model_folder: Path):
    """Load the YAML config inside the model folder (COCO assumed)."""
    config_path = model_folder / "config_coco.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg_from_file(str(config_path))
    cfg.ROOT_DIR = str(model_folder)
    return config_path


def run_single_eval(
    model_folder: Path,
    corrupt_type: str,
    level: str,
    test_samples: int,
    dataset: str,
    resume: int,
) -> Dict:
    """Run one evaluation and return metrics dict."""
    reset_logger_handlers()
    load_config(model_folder)
    cfg.LOGGER_NAME = f"log_{model_folder.name}_{corrupt_type}_{level}"

    normalized_level = jpeg_corruption.normalize_level(level)
    cfg.CORRUPTION.BYTE_STREAM_TYPES = [corrupt_type]
    cfg.CORRUPTION.BYTE_STREAM_LEVEL = normalized_level

    args = Namespace(
        folder=str(model_folder),
        resume=resume,
        test_samples=test_samples,
        val_samples=test_samples,  # backward-compat with Tester signature
        corrupt_level=normalized_level,
        corrupt_types=[corrupt_type],
        disable_wandb=True,
        dataset=dataset,
    )
    tester = Tester(args)
    metrics = tester.test_evaler(tester.model, f"{corrupt_type}_{normalized_level}")
    # Attach the corruption params for traceability
    preset_level = jpeg_corruption.normalize_level(normalized_level)
    corruption_params = jpeg_corruption.JPEG_CORRUPTION_PRESETS.get(
        corrupt_type, {}
    ).get(preset_level, {})
    metrics["corruption_params"] = corruption_params
    return metrics


def plot_curves(results: List[Dict], metric_keys: List[str], output_dir: Path):
    """Plot metric vs severity curves grouped by model and corruption type."""
    level_order = ["S0", "S1", "S2", "S3", "S4", "S5"]
    level_to_idx = {lvl: i for i, lvl in enumerate(level_order)}

    for metric in metric_keys:
        plt.figure(figsize=(8, 5))
        for model_name in sorted({r["model_name"] for r in results}):
            for ctype in sorted({r["corrupt_type"] for r in results}):
                subset = [
                    r for r in results if r["model_name"] == model_name and r["corrupt_type"] == ctype
                ]
                if not subset:
                    continue
                subset = sorted(subset, key=lambda x: level_to_idx.get(x["corrupt_level"], 0))
                xs = [level_to_idx.get(r["corrupt_level"], 0) for r in subset]
                ys = [r["metrics"].get(metric, None) for r in subset]
                if all(v is None for v in ys):
                    continue
                plt.plot(xs, ys, marker="o", label=f"{model_name}-{ctype}")
        plt.xticks(list(level_to_idx.values()), level_order)
        plt.xlabel("Corruption level")
        plt.ylabel(metric)
        plt.title(f"{metric} vs corruption level")
        plt.grid(True, alpha=0.3)
        plt.legend()
        outfile = output_dir / f"curve_{metric}.png"
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Batch corruption evaluation runner")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model folders (each containing config_coco.yml)",
    )
    parser.add_argument("--corrupt-types", nargs="+", default=["rbbf", "rbsl", "metadata_loss"])
    parser.add_argument("--corrupt-levels", nargs="+", default=["S0", "S1", "S2", "S3", "S4", "S5"])
    parser.add_argument("--test-samples", type=int, default=80, help="Number of test samples (0 = all)")
    parser.add_argument("--val-samples", type=int, default=None, help="(Alias) Number of test samples (0 = all)")
    parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "flickr8k"])
    parser.add_argument("--resume", type=int, default=-1, help="Checkpoint to load (-1 = best)")
    parser.add_argument(
        "--plot-metrics",
        nargs="+",
        default=["CIDEr", "Bleu_4"],
        help="Metrics to plot against corruption level",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="batch_reports",
        help="Directory to store aggregated results and plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.val_samples is not None:
        args.test_samples = args.val_samples
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_runs = len(args.models) * len(args.corrupt_types) * len(args.corrupt_levels)
    with tqdm(total=total_runs, desc="Batch eval", unit="run") as pbar:
        for model_folder_str in args.models:
            model_folder = Path(model_folder_str)
            model_name = model_folder.name
            for ctype in args.corrupt_types:
                for level in args.corrupt_levels:
                    print(f"\n[RUN] model={model_name}, type={ctype}, level={level}")
                    metrics = run_single_eval(
                        model_folder,
                        ctype,
                        level,
                        test_samples=args.test_samples,
                        dataset=args.dataset,
                        resume=args.resume,
                    )
                    run_record = {
                        "model_folder": str(model_folder),
                        "model_name": model_name,
                        "model_type": str(getattr(cfg.MODEL, "TYPE", "")),
                        "corrupt_type": ctype,
                        "corrupt_level": jpeg_corruption.normalize_level(level),
                        "metrics": metrics,
                    }
                    all_results.append(run_record)
                    # save per-run metrics
                    per_run_path = output_dir / f"{model_name}_{ctype}_{level}.json"
                    with open(per_run_path, "w", encoding="utf-8") as f:
                        json.dump(run_record, f, ensure_ascii=False, indent=2)
                    pbar.update(1)

    # Aggregate summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"runs": all_results}, f, ensure_ascii=False, indent=2)

    # Plot representative metrics
    try:
        plot_curves(all_results, args.plot_metrics, output_dir)
        print(f"[PLOT] Saved metric curves to {output_dir}")
    except Exception as e:
        print(f"[WARN] Failed to plot curves: {e}")

    print(f"\nDone. Aggregated outputs at: {output_dir}")


if __name__ == "__main__":
    main()
