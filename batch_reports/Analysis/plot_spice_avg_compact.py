"""\
竖向小图（上下两个子图）：
- 上：SPICE（RBBF 与 RBSL 的平均）绝对值曲线，BCM vs Avg(w/o BCM)
- 下：相对 SPICE（相对于各自 S0 的百分比），同样对 RBBF 与 RBSL 取平均，BCM vs Avg(w/o BCM)

风格参考：
- batch_reports/Analysis/curves_combined.svg（线型/配色/整体观感）
- batch_reports/Analysis/relative_bar_quadrants.svg（相对指标定义）

输出：batch_reports/Analysis/spice_avg_compact.svg
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


LEVEL_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5"]
HIGHLIGHT_COLOR = "#2E5C8A"  # BCM
AVG_COLOR = "#E67E22"  # Avg

FONT_SIZES = {
    "default": 15,
    "title": 18,
    "axes_label": 15,
    "tick_label": 14,
    "legend": 12,
}


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZES["default"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["axes_label"],
            "legend.fontsize": FONT_SIZES["legend"],
            "xtick.labelsize": FONT_SIZES["tick_label"],
            "ytick.labelsize": FONT_SIZES["tick_label"],
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


def load_runs(input_dir: Path) -> List[Dict]:
    runs: List[Dict] = []
    for json_path in sorted(input_dir.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and "metrics" in data:
            runs.append(data)
    return runs


def iter_models(runs: Iterable[Dict]) -> List[str]:
    model_set = {r.get("model_name") for r in runs if r.get("model_name")}
    models = sorted(model_set)
    if "ByteCaption_XE" in models:
        models.remove("ByteCaption_XE")
        models.insert(0, "ByteCaption_XE")
    return models


def _get_metric_value(run: Dict, metric: str) -> float | None:
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        return None
    v = metrics.get(metric)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def build_series_abs_percent(runs: Sequence[Dict], model: str, corrupt_type: str, metric: str) -> np.ndarray:
    """返回 *100 后的绝对值序列（长度 6），缺失为 NaN。"""
    series: List[float] = []
    for level_key in LEVEL_ORDER:
        value: float | None = None
        for run in runs:
            if run.get("corrupt_type") != corrupt_type:
                continue
            if run.get("model_name") != model:
                continue
            if run.get("corrupt_level") != level_key:
                continue
            raw = _get_metric_value(run, metric)
            if raw is not None:
                value = raw * 100.0
                break
        series.append(np.nan if value is None else value)
    return np.asarray(series, dtype=float)


def build_series_relative(runs: Sequence[Dict], model: str, corrupt_type: str, metric: str) -> np.ndarray:
    """相对序列：S0=100，其它为 (Sx/S0)*100。缺失为 NaN。"""
    abs_series = []
    for level_key in LEVEL_ORDER:
        value: float | None = None
        for run in runs:
            if run.get("corrupt_type") != corrupt_type:
                continue
            if run.get("model_name") != model:
                continue
            if run.get("corrupt_level") != level_key:
                continue
            raw = _get_metric_value(run, metric)
            if raw is not None:
                value = raw
                break
        abs_series.append(np.nan if value is None else float(value))

    abs_arr = np.asarray(abs_series, dtype=float)
    s0 = abs_arr[0]
    if np.isnan(s0) or s0 == 0:
        return np.full_like(abs_arr, np.nan)

    rel = abs_arr / s0 * 100.0
    rel[0] = 100.0
    return rel


def average_two_types(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐点对两个序列取平均；单侧缺失则用另一侧；都缺失则 NaN。"""
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    out = np.empty_like(a, dtype=float)
    for i in range(a.shape[0]):
        av, bv = a[i], b[i]
        if np.isnan(av) and np.isnan(bv):
            out[i] = np.nan
        elif np.isnan(av):
            out[i] = bv
        elif np.isnan(bv):
            out[i] = av
        else:
            out[i] = (av + bv) / 2.0
    return out


def compute_avg_over_models_abs_percent(
    runs: Sequence[Dict],
    corrupt_type: str,
    metric: str,
    exclude_model: str = "ByteCaption_XE",
) -> np.ndarray:
    models = [m for m in iter_models(runs) if m != exclude_model]
    if not models:
        return np.full((len(LEVEL_ORDER),), np.nan)

    stacked: List[np.ndarray] = []
    for m in models:
        stacked.append(build_series_abs_percent(runs, m, corrupt_type, metric))

    arr = np.vstack(stacked)  # (M, 6)
    with np.errstate(all="ignore"):
        return np.nanmean(arr, axis=0)


def compute_avg_over_models_relative(
    runs: Sequence[Dict],
    corrupt_type: str,
    metric: str,
    exclude_model: str = "ByteCaption_XE",
) -> np.ndarray:
    models = [m for m in iter_models(runs) if m != exclude_model]
    if not models:
        return np.full((len(LEVEL_ORDER),), np.nan)

    stacked: List[np.ndarray] = []
    for m in models:
        stacked.append(build_series_relative(runs, m, corrupt_type, metric))

    arr = np.vstack(stacked)
    with np.errstate(all="ignore"):
        return np.nanmean(arr, axis=0)


def plot_spice_avg_compact(runs: List[Dict], output_path: Path) -> None:
    apply_plot_style()

    metric = "SPICE"
    corrupt_types = ("rbbf", "rbsl")

    # 上：绝对 SPICE（百分比）
    bcm_abs_a = build_series_abs_percent(runs, "ByteCaption_XE", corrupt_types[0], metric)
    bcm_abs_b = build_series_abs_percent(runs, "ByteCaption_XE", corrupt_types[1], metric)
    bcm_abs = average_two_types(bcm_abs_a, bcm_abs_b)

    avg_abs_a = compute_avg_over_models_abs_percent(runs, corrupt_types[0], metric)
    avg_abs_b = compute_avg_over_models_abs_percent(runs, corrupt_types[1], metric)
    avg_abs = average_two_types(avg_abs_a, avg_abs_b)

    # 下：相对 SPICE（S0=100）
    bcm_rel_a = build_series_relative(runs, "ByteCaption_XE", corrupt_types[0], metric)
    bcm_rel_b = build_series_relative(runs, "ByteCaption_XE", corrupt_types[1], metric)
    bcm_rel = average_two_types(bcm_rel_a, bcm_rel_b)

    avg_rel_a = compute_avg_over_models_relative(runs, corrupt_types[0], metric)
    avg_rel_b = compute_avg_over_models_relative(runs, corrupt_types[1], metric)
    avg_rel = average_two_types(avg_rel_a, avg_rel_b)

    x = np.arange(len(LEVEL_ORDER), dtype=float)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(6.2, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": [0.75, 0.75]},
    )

    # 背景/网格
    for ax in (ax_top, ax_bot):
        ax.set_facecolor("#FAFAFA")
        ax.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.8, color="gray", zorder=0)
        ax.set_axisbelow(True)

    # --- 顶部曲线 ---
    bcm_valid = ~np.isnan(bcm_abs)
    avg_valid = ~np.isnan(avg_abs)

    if bcm_valid.any():
        ax_top.plot(
            x[bcm_valid],
            bcm_abs[bcm_valid],
            marker="o",
            linewidth=2.4,
            alpha=0.9,
            color=HIGHLIGHT_COLOR,
            label="BCM (Ours)",
            zorder=3,
        )

    if avg_valid.any():
        ax_top.plot(
            x[avg_valid],
            avg_abs[avg_valid],
            marker="s",
            linewidth=2.0,
            alpha=0.8,
            color=AVG_COLOR,
            linestyle="--",
            label="Avg (w/o BCM)",
            zorder=3,
        )

    ax_top.set_ylabel("SPICE", fontweight="bold")
    # 给一点上边界留白
    top_max = np.nanmax([np.nanmax(bcm_abs), np.nanmax(avg_abs)]) if (bcm_valid.any() or avg_valid.any()) else 1.0
    ax_top.set_ylim(0, top_max * 1.12)
    ax_top.legend(loc="upper right", framealpha=0.95)

    # --- 底部柱状 ---
    bar_w = 0.34

    bcm_rel_plot = np.where(np.isnan(bcm_rel), 0.0, bcm_rel)
    avg_rel_plot = np.where(np.isnan(avg_rel), 0.0, avg_rel)

    ax_bot.bar(x - bar_w / 2, bcm_rel_plot, width=bar_w, color=HIGHLIGHT_COLOR, alpha=0.8, label="BCM (Ours)", zorder=2)
    ax_bot.bar(x + bar_w / 2, avg_rel_plot, width=bar_w, color=AVG_COLOR, alpha=0.8, label="Avg (w/o BCM)", zorder=2)

    ax_bot.set_ylabel("Relative SPICE (%)", fontweight="bold")
    bot_max = np.nanmax([np.nanmax(bcm_rel_plot), np.nanmax(avg_rel_plot)]) if (np.any(bcm_rel_plot) or np.any(avg_rel_plot)) else 110.0
    ax_bot.set_ylim(0, max(110.0, bot_max * 1.15))

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(LEVEL_ORDER)
    ax_bot.set_xlabel("Corruption Severity", fontweight="bold")

    # 总标题
    fig.suptitle(
        "SPICE: BCM vs Avg of Pixel-Based Models",
        fontsize=FONT_SIZES["title"],
        y=0.97,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "all_models"
    if not input_dir.exists():
        print(f"[ERROR] Data directory not found: {input_dir}")
        return

    runs = load_runs(input_dir)
    if not runs:
        print("[ERROR] No valid JSON report files found")
        return

    output_path = Path(__file__).parent / "spice_avg_compact.svg"
    plot_spice_avg_compact(runs, output_path)
    print(f"[OK] Saved: {output_path}")


if __name__ == "__main__":
    main()
