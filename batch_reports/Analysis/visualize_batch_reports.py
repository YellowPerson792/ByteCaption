from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ===== 可视化配置常量 =====
# 修改这些常量可以调整图表的外观和行为

# 损坏严重程度级别顺序（S0表示无损坏的基线）
LEVEL_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5"]
LEVEL_TO_IDX = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}
DROP_LEVELS = LEVEL_ORDER[1:]  # 排除S0，用于计算性能下降

# 损坏类型顺序（rbbf=随机位翻转，rbsl=随机位移位）
CORRUPT_ORDER = ["rbbf", "rbsl"]

# 表格中显示的指标列表
TABLE_METRICS = ["CIDEr", "SPICE"]

# 图像输出的DPI（分辨率）- 提高此值可获得更清晰的图像
FIG_DPI = 300

# 高亮颜色（用于突出显示ByteCaption_XE模型）
HIGHLIGHT_COLOR = "#2E5C8A"  # 深蓝色

# 其他模型使用的调色板 - 可以修改这些颜色代码来改变图表配色
PALETTE = [
    "#4E79A7",  # 蓝色
    "#F28E2B",  # 橙色
    "#59A14F",  # 绿色
    "#B07AA1",  # 紫色
    "#9C755F",  # 棕色
    "#EDC948",  # 黄色
    "#76B7B2",  # 青色
    "#FF9DA7",  # 粉色
    "#A0CBE8",  # 浅蓝色
]

# 模型分组配置 - 用于表格中的分组显示
# 格式：("分组名称", [模型列表])
# 修改此配置可以调整表格中模型的分组和顺序
MODEL_GROUPS = [
    ("COCO Pretrained Models", ["ByteCaption_XE_blip", "ByteCaption_XE_git"]),
    (
        "COCO Fine-Tuned Models",
        [
            "ByteCaption_XE_qwen",
            "ByteCaption_XE_internvl",
            "ByteCaption_XE_glm",
            "ByteCaption_XE_ministral",
        ],
    ),
    (
        "Zero-Shot Generative Models",
        [
            "ByteCaption_XE_gpt5.1",
            "ByteCaption_XE_gemini2.5-flash",
            "ByteCaption_XE_claude-haiku-4.5",
        ],
    ),
    ("Ours", ["ByteCaption_XE"]),
]

# 扁平化的模型顺序（按分组顺序排列）
MODEL_ORDER = [m for _, models in MODEL_GROUPS for m in models]

# 模型显示标签映射 - 将内部模型名称映射为图表中显示的友好名称
# 添加新模型时，在这里添加其显示名称
MODEL_LABELS = {
    "ByteCaption_XE": "BCM (Ours)",
    "ByteCaption_XE_blip": "BLIP",
    "ByteCaption_XE_git": "GIT",
    "ByteCaption_XE_qwen": "Qwen3-VL-8B",
    "ByteCaption_XE_gpt5.1": "GPT-5.1",
    "ByteCaption_XE_gemini2.5-flash": "Gemini-2.5-Flash",
    "ByteCaption_XE_claude-haiku-4.5": "Claude-Haiku-4.5",
    "ByteCaption_XE_internvl": "InternVL-3.5-8B",
    "ByteCaption_XE_glm": "GLM-4.6V",
    "ByteCaption_XE_ministral": "Ministral-3-8B",
}

# 占位文本（用于判定无效输入）
PLACEHOLDER_SENTINELS = {
    "the image cannot be identified",
    "a person standing in a room",
    "this is a dummy caption for an undecodable image",
}

# 指标缩放因子 - 将原始指标值乘以此因子（用于转换为百分比）
# 例如：CIDEr从0-1转换为0-100
SCALE_METRICS = {
    "CIDEr": 100.0,
    "SPICE": 100.0,
}

# ===== 统一字体大小配置 =====
# 所有图表的字体大小设置集中在这里
FONT_SIZES = {
    "default": 14,           # 默认字体大小
    "title": 18,             # 大标题（如"Metrics Curves..."）
    "subtitle": 14,          # 列标题（如"RBBF", "RBSL"）
    "axes_label": 14,        # 坐标轴标签字体大小
    "tick_label": 14,        # 刻度标签字体大小
    "legend": 12,            # 图例字体大小
    "robustness_title": 14,  # 鲁棒性得分图标题
}


def load_runs(input_dir: Path) -> List[Dict]:
    """从输入目录加载所有运行的JSON数据
    
    优先尝试加载 summary.json（包含所有运行的汇总文件）
    如果不存在，则逐个加载目录中的所有 *.json 文件
    
    Args:
        input_dir: 包含JSON报告文件的目录
        
    Returns:
        包含所有运行数据的字典列表
    """
    runs = []
    summary_path = input_dir / "summary.json"
    # 首先尝试加载汇总文件
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "runs" in data:
                return [r for r in data["runs"] if isinstance(r, dict)]
        except Exception:
            pass

    # 如果没有汇总文件，逐个加载JSON文件
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
        default="batch_reports/all_models",
        help="Directory with per-run JSONs or summary.json (default: ../all_models)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="batch_reports/Analysis",
        help="Directory to write plots and summaries (default: current directory)",
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
    present = {r.get("corrupt_type") for r in runs if r.get("corrupt_type")}
    ordered = [c for c in CORRUPT_ORDER if c in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def get_metric_value(run: Dict, metric: str) -> float | None:
    val = (run.get("metrics") or {}).get(metric, None)
    if val is None:
        return None
    scale = SCALE_METRICS.get(metric, 1.0)
    return float(val) * scale


def _normalize_caption(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    # 统一大小写并去掉末尾常见标点
    return text.strip().lower().rstrip(".!?:;,\u3002")


def count_placeholder_captions(samples: list | None) -> int:
    """统计占位文本出现的次数，用于计算有效输入率"""
    if not isinstance(samples, list):
        return 0
    count = 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        candidate = sample.get("caption") or sample.get("generated")
        norm = _normalize_caption(candidate)
        if norm in PLACEHOLDER_SENTINELS:
            count += 1
    return count


def _is_pixel_model(model_name: str | None) -> bool:
    """Heuristic: treat all non-ByteCaption models as pixel-domain."""
    if not model_name:
        return False
    return model_name != "ByteCaption_XE"


def compute_valid_input_rate(run: Dict) -> float | None:
    """????????1 - (????) / ???"""
    metrics = run.get("metrics") or {}
    samples = run.get("caption_samples") or []
    model_name = run.get("model_name")

    if _is_pixel_model(model_name):
        total = 500
    else:
        total = metrics.get("Total_Samples")
        if total is None and isinstance(samples, list):
            total = len(samples)

    if not total or total <= 0:
        return None

    placeholder_count = count_placeholder_captions(samples)
    invalid = min(placeholder_count, total)
    valid = total - invalid
    return valid / total


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


def apply_plot_style() -> None:
    """应用全局图表样式配置
    
    修改此函数可以调整所有图表的视觉风格：
    - font.size: 基础字体大小
    - axes.titlesize: 标题字体大小
    - axes.labelsize: 轴标签字体大小
    - legend.fontsize: 图例字体大小
    - xtick/ytick.labelsize: 刻度标签字体大小
    - axes.spines: 控制图表边框的显示（top/right设为False隐藏上和右边框）
    - axes.linewidth: 坐标轴线宽
    """
    plt.rcParams.update(
        {
            "font.size": FONT_SIZES["default"],           # 默认字体大小
            "axes.labelsize": FONT_SIZES["axes_label"],      # 坐标轴标签字体大小
            "legend.fontsize": FONT_SIZES["legend"],      # 图例字体大小
            "xtick.labelsize": FONT_SIZES["tick_label"],      # X轴刻度标签字体大小
            "ytick.labelsize": FONT_SIZES["tick_label"],      # Y轴刻度标签字体大小
            "axes.spines.top": False,  # 隐藏顶部边框
            "axes.spines.right": False,# 隐藏右侧边框
            "axes.linewidth": 0.6,     # 坐标轴线宽
            "xtick.major.width": 0.6,  # X轴刻度线宽
            "ytick.major.width": 0.6,  # Y轴刻度线宽
        }
    )


def build_color_map(models: Iterable[str]) -> Dict[str, str]:
    """为每个模型分配颜色
    
    ByteCaption_XE使用高亮色（红色）
    其他模型依次使用PALETTE中的颜色
    
    Args:
        models: 模型名称列表
        
    Returns:
        模型名称到颜色代码的映射字典
    """
    color_map: Dict[str, str] = {}
    color_idx = 0
    for model in models:
        if model == "ByteCaption_XE":
            # 我们的模型使用特殊的高亮颜色
            color_map[model] = HIGHLIGHT_COLOR
        else:
            # 其他模型循环使用调色板中的颜色
            color_map[model] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1
    return color_map


def build_model_avg(runs: List[Dict], corrupt_type: str, metric: str) -> List[float]:
    """计算除BCM外的模型平均值"""
    models = iter_models(runs)
    models_without_bcm = [m for m in models if m != "ByteCaption_XE"]
    if not models_without_bcm:
        return [np.nan] * len(LEVEL_ORDER)
    
    series_list = [build_series(runs, m, corrupt_type, metric) for m in models_without_bcm]
    avg = np.nanmean(series_list, axis=0)
    return list(avg)


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
    """生成鲁棒性得分条形图，样式与curves_combined和valid_input_rate统一
    
    统一样式包括：
    - 粗体标题（y=0.98）
    - 与其他图表一致的视觉风格
    """
    results = compute_robustness_scores(runs, metrics)
    results = [r for r in results if not np.isnan(r[1])]
    if not results:
        return output_dir / "robustness_score.png"

    models = [m for m, _ in results]
    color_map = build_color_map(models)
    labels = [model_label(m) for m, _ in results]
    scores = [s * 100.0 for _, s in results]
    colors = [color_map[m] for m in models]

    # 调整图表尺寸
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(labels, scores, color=colors, alpha=0.77)
    ax.set_ylabel("Robustness Score (%)", fontweight='bold', fontsize=12)
    ax.set_ylim(0, max(scores) * 1.15)
    
    # 显示四周边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # X轴标签旋转角度
    ax.tick_params(axis="x", labelrotation=27, labelsize=14)
    
    # 添加统一的大标题（粗体，与其他图一致）
    fig.suptitle("Robustness Score", fontsize=FONT_SIZES["title"], y=0.98, fontweight="bold")
    
    fig.tight_layout(rect=[0, 0, 1, 1])
    outfile = output_dir / "robustness_score.svg"
    fig.savefig(outfile, format="svg")
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


def build_run_index(runs: Iterable[Dict]) -> Dict[Tuple[str, str, str], Dict]:
    index: Dict[Tuple[str, str, str], Dict] = {}
    for run in runs:
        valid_rate = compute_valid_input_rate(run)
        if valid_rate is not None:
            run.setdefault("metrics", {})["Valid_Input_Rate"] = valid_rate
        model = run.get("model_name")
        corrupt = run.get("corrupt_type")
        level = run.get("corrupt_level")
        if model and corrupt and level:
            index[(model, corrupt, level)] = run
    return index


def split_models_by_group(present: Iterable[str]) -> List[Tuple[str, List[str]]]:
    present_set = set(present)
    grouped: List[Tuple[str, List[str]]] = []
    used = set()
    for group, models in MODEL_GROUPS:
        filtered = [m for m in models if m in present_set]
        if filtered:
            grouped.append((group, filtered))
            used.update(filtered)
    extras = sorted(present_set - used)
    if extras:
        grouped.append(("Other Models", extras))
    return grouped


def get_index_metric(
    index: Dict[Tuple[str, str, str], Dict],
    model: str,
    corrupt: str,
    level: str,
    metric: str,
    scale: float = 1.0,
) -> float | None:
    run = index.get((model, corrupt, level))
    if not run:
        return None
    metrics = run.get("metrics") or {}
    val = metrics.get(metric)
    if val is None:
        return None
    return float(val) * scale


def format_value(val: float | None, decimals: int = 1) -> str:
    if val is None or np.isnan(val):
        return "-"
    return f"{val:.{decimals}f}"


def build_metrics_table_html(
    index: Dict[Tuple[str, str, str], Dict],
    corrupt_type: str,
    models: List[str],
) -> str:
    max_vals: Dict[str, Dict[str, float | None]] = {
        lvl: {metric: None for metric in TABLE_METRICS} for lvl in LEVEL_ORDER
    }

    models_without_bcm = [m for m in models if m != "ByteCaption_XE"]
    avg_vals: Dict[str, Dict[str, float | None]] = {
        lvl: {metric: None for metric in TABLE_METRICS} for lvl in LEVEL_ORDER
    }
    for lvl in LEVEL_ORDER:
        for metric in TABLE_METRICS:
            values = []
            for model in models:
                val = get_index_metric(
                    index,
                    model,
                    corrupt_type,
                    lvl,
                    metric,
                    SCALE_METRICS.get(metric, 1.0),
                )
                if val is not None:
                    values.append(round(val, 1))
            max_vals[lvl][metric] = max(values) if values else None

            # 计算非BCM模型的平均值（仅用于“Avg (w/o BCM)”行）
            if models_without_bcm:
                avg_candidates = []
                for model in models_without_bcm:
                    v = get_index_metric(
                        index,
                        model,
                        corrupt_type,
                        lvl,
                        metric,
                        SCALE_METRICS.get(metric, 1.0),
                    )
                    if v is not None:
                        avg_candidates.append(round(v, 1))
                avg_vals[lvl][metric] = (
                    float(np.mean(avg_candidates)) if avg_candidates else None
                )

    lines = []
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append("    <tr>")
    lines.append('      <th rowspan="2">Model</th>')
    lines.append(f'      <th colspan="6">{corrupt_type.upper()}</th>')
    lines.append("    </tr>")
    lines.append("    <tr>")
    for lvl in LEVEL_ORDER:
        lines.append(f"      <th>{lvl}</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")
    avg_inserted = False
    for group, group_models in split_models_by_group(models):
        is_bcm_only_group = len(group_models) == 1 and group_models[0] == "ByteCaption_XE"
        if not is_bcm_only_group:
            lines.append(f'  <tr><th colspan="7">{group}</th></tr>')
        for model in group_models:
            if model == "ByteCaption_XE" and not avg_inserted:
                row_avg = ["  <tr><td>Avg (w/o BCM)</td>"]
                for lvl in LEVEL_ORDER:
                    c_val = avg_vals[lvl]["CIDEr"]
                    s_val = avg_vals[lvl]["SPICE"]
                    c_txt = format_value(c_val, 1)
                    s_txt = format_value(s_val, 1)
                    row_avg.append(f"<td>{c_txt} / {s_txt}</td>")
                row_avg.append("</tr>")
                lines.append("".join(row_avg))
                avg_inserted = True
            row = [f"  <tr><td>{model_label(model)}</td>"]
            for lvl in LEVEL_ORDER:
                c_val = get_index_metric(
                    index,
                    model,
                    corrupt_type,
                    lvl,
                    "CIDEr",
                    SCALE_METRICS.get("CIDEr", 1.0),
                )
                s_val = get_index_metric(
                    index,
                    model,
                    corrupt_type,
                    lvl,
                    "SPICE",
                    SCALE_METRICS.get("SPICE", 1.0),
                )
                c_txt = format_value(c_val, 1)
                s_txt = format_value(s_val, 1)
                if c_val is not None and round(c_val, 1) == max_vals[lvl]["CIDEr"]:
                    c_txt = f"<strong>{c_txt}</strong>"
                if s_val is not None and round(s_val, 1) == max_vals[lvl]["SPICE"]:
                    s_txt = f"<strong>{s_txt}</strong>"
                row.append(f"<td>{c_txt} / {s_txt}</td>")
            row.append("</tr>")
            lines.append("".join(row))
    lines.append("  </tbody>")
    lines.append("</table>")
    return "\n".join(lines)


def build_valid_input_table_html(
    index: Dict[Tuple[str, str, str], Dict],
    corrupt_type: str,
    models: List[str],
) -> str:
    models_without_bcm = [m for m in models if m != "ByteCaption_XE"]
    avg_vals: Dict[str, float | None] = {lvl: None for lvl in LEVEL_ORDER}
    for lvl in LEVEL_ORDER:
        candidates = []
        for model in models_without_bcm:
            val = get_index_metric(index, model, corrupt_type, lvl, "Valid_Input_Rate", 100.0)
            if val is not None:
                candidates.append(val)
        avg_vals[lvl] = float(np.mean(candidates)) if candidates else None

    lines = []
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append("    <tr>")
    lines.append('      <th rowspan="2">Model</th>')
    lines.append(f'      <th colspan="6">{corrupt_type.upper()} (Valid Input Rate %)</th>')
    lines.append("    </tr>")
    lines.append("    <tr>")
    for lvl in LEVEL_ORDER:
        lines.append(f"      <th>{lvl}</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")
    avg_inserted = False
    for group, group_models in split_models_by_group(models):
        is_bcm_only_group = len(group_models) == 1 and group_models[0] == "ByteCaption_XE"
        if not is_bcm_only_group:
            lines.append(f'  <tr><th colspan="7">{group}</th></tr>')
        for model in group_models:
            if model == "ByteCaption_XE" and not avg_inserted:
                row_avg = ["  <tr><td>Avg (w/o BCM)</td>"]
                for lvl in LEVEL_ORDER:
                    val = avg_vals[lvl]
                    cell = "-" if val is None else f"{val:.1f}%"
                    row_avg.append(f"<td>{cell}</td>")
                row_avg.append("</tr>")
                lines.append("".join(row_avg))
                avg_inserted = True

            row = [f"  <tr><td>{model_label(model)}</td>"]
            for lvl in LEVEL_ORDER:
                val = get_index_metric(index, model, corrupt_type, lvl, "Valid_Input_Rate", 100.0)
                cell = "-" if val is None else f"{val:.1f}%"
                row.append(f"<td>{cell}</td>")
            row.append("</tr>")
            lines.append("".join(row))
    lines.append("  </tbody>")
    lines.append("</table>")
    return "\n".join(lines)


def write_summary_metrics_tables(
    runs: List[Dict],
    output_dir: Path,
) -> Path:
    models = iter_models(runs)
    index = build_run_index(runs)
    tables = [
        build_metrics_table_html(index, "rbbf", models),
        build_metrics_table_html(index, "rbsl", models),
        build_valid_input_table_html(index, "rbbf", models),
        build_valid_input_table_html(index, "rbsl", models),
    ]
    out_path = output_dir / "summary_metrics_table.md"
    out_path.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    return out_path


def write_tables_md(output_dir: Path) -> Path:
    summary_path = output_dir / "summary_metrics_table.md"
    summary_text = summary_path.read_text(encoding="utf-8").rstrip()
    sections = [
        summary_text,
        "## Visualizations",
        "### Combined Curves (CIDEr/SPICE)",
        "",
        "![Combined Curves](curves_combined.svg)",
        "",
        "### Combined Drop Heatmaps (relative to S0)",
        "",
        "![Combined Drop Heatmap](heatmap_drop_combined.svg)",
        "",
        "### Aggregate robustness score",
        "",
        "![Robustness score](robustness_score.svg)",
        "",
        "Summary CSV: [robustness_summary.csv](robustness_summary.csv)",
        "",
        "## Valid Input Rate across Corruption Levels",
        "",
        "![Combined Valid Input Rate](valid_input_rate_combined.svg)",
    ]
    out_path = output_dir / "tables.md"
    out_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return out_path


def plot_valid_input_curves(runs: List[Dict], output_dir: Path) -> None:
    """生成解码成功/失败的平均堆叠柱状图（左右合并布局）
    
    展现形式：
    - 仅考虑模型平均 (Avg w/o BCM)
    - 下半柱：平均成功率
    - 上半柱：互补失败率 (100 - 成功率)
    二者叠加后总高度恒为100。
    """
    models = iter_models(runs)
    corrupt_types = iter_corrupt_types(runs)
    if not models or not corrupt_types:
        return

    index = build_run_index(runs)
    color_map = build_color_map(models)
    # 为模型平均值添加高亮对比色（与曲线图一致）
    color_map["Model Avg"] = "#157AB5B7"

    # 创建左右合并布局图表
    fig, axes = plt.subplots(1, len(corrupt_types), figsize=(10, 5))
    
    # 确保axes是列表
    if len(corrupt_types) == 1:
        axes = [axes]
    
    # 子图标记
    subplot_labels = [chr(97 + i) for i in range(len(corrupt_types))]
    
    for col, corrupt_type in enumerate(corrupt_types):
        ax = axes[col]

        # 计算模型平均的解码成功率（仅非BCM）
        avg_success: list[float] = []
        for lvl in LEVEL_ORDER:
            values = []
            for model in models:
                if model == "ByteCaption_XE":
                    continue
                val = get_index_metric(index, model, corrupt_type, lvl, "Valid_Input_Rate", 100.0)
                if val is not None:
                    values.append(val)
            avg_success.append(np.mean(values) if values else np.nan)

        # 构建堆叠柱：下半为成功率，上半为失败率（100-成功率）
        x = list(range(len(LEVEL_ORDER)))
        succ = [0 if np.isnan(v) else v for v in avg_success]
        fail = [0 if np.isnan(v) else (100.0 - v) for v in avg_success]

        # 下半：有效输入率（橙金）
        bar1 = ax.bar(x, succ, color=color_map["Model Avg"], alpha=0.80, label="Valid Input (Avg, w/o BCM)")
        # 上半：无效输入率（浅灰）
        bar2 = ax.bar(x, fail, bottom=succ, color="#D9D9D9", alpha=0.90, label="Invalid = 100 - Valid")
        
        ax.set_xticks(list(range(len(LEVEL_ORDER))))
        ax.set_xticklabels(LEVEL_ORDER)
        ax.set_xlabel("Corruption Severity", fontweight='bold', fontsize=12)
        # 只在左列显示Y轴标签
        ax.set_ylabel("Valid Input Rate (%)" if col == 0 else "", fontweight='bold', fontsize=12)
        ax.set_ylim(0, 105)
        
        # 子图标签，采用与曲线图一致的样式
        ax.text(
            0.7, 0.85,
            f"({subplot_labels[col]}) {corrupt_type.upper()}",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5)
        )
    
    # 添加统一的大标题（改为 Valid Input Rate）
    fig.suptitle("Valid Input Rate across Corruption Severity", fontsize=FONT_SIZES["title"], y=0.98, fontweight="bold")
    
    # 在标题下面显示图例，两项即可
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.08), fontsize=FONT_SIZES["legend"])

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    outfile = output_dir / "valid_input_rate_combined.svg"
    fig.savefig(outfile, format="svg")
    plt.close(fig)


def main() -> None:
    apply_plot_style()
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(input_dir)
    if not runs:
        raise SystemExit(f"No run JSONs found in {input_dir}")

    metrics = args.metrics

    plot_robustness_scores(runs, metrics, output_dir)
    write_summary_csv(runs, metrics, output_dir)
    plot_valid_input_curves(runs, output_dir)
    write_summary_metrics_tables(runs, output_dir)
    write_tables_md(output_dir)

    print(f"[VIS] Wrote figures and summaries to {output_dir}")


if __name__ == "__main__":
    main()
