"""
四象限曲线图 - 参考relative_bar_quadrants.svg的布局风格
重新设计curves_combined.svg，关键特性：
1. 保持右上为正方向（不颠倒）
2. CIDER和SPICE虽然纵轴范围不同，但视觉空间相同
3. 横轴为S0-S5，通过坐标变换映射到四个象限
4. 使用坐标变换映射数据点，正确显示刻度值
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


# 配置常量
LEVEL_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5"]
HIGHLIGHT_COLOR = "#2E5C8A"  # BCM深蓝色
AVG_COLOR = "#E67E22"  # Avg橙金色

FONT_SIZES = {
    "default": 14,
    "title": 18,
    "axes_label": 14,
    "tick_label": 14,
    "legend": 10,
}

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


def apply_plot_style() -> None:
    """统一图表样式"""
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
    """从目录加载所有JSON运行数据"""
    runs = []
    for json_path in sorted(input_dir.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and "metrics" in data:
            runs.append(data)
    return runs


def iter_models(runs: List[Dict]) -> List[str]:
    """获取所有模型名称"""
    model_set = {r.get("model_name") for r in runs if r.get("model_name")}
    models = sorted(model_set)
    if "ByteCaption_XE" in models:
        models.remove("ByteCaption_XE")
        models.insert(0, "ByteCaption_XE")
    return models


def build_series(runs: List[Dict], model: str, corrupt_type: str, metric: str) -> List[float]:
    """构建特定模型、损坏类型、指标的数据序列"""
    series = []
    for level_key in LEVEL_ORDER:
        value = None
        for run in runs:
            if run.get("corrupt_type") != corrupt_type:
                continue
            if run.get("model_name") != model:
                continue
            if run.get("corrupt_level") == level_key:
                metrics_data = run.get("metrics", {})
                if metric in metrics_data:
                    value = metrics_data[metric] * 100.0  # 转换为百分比
                    break
        series.append(value if value is not None else np.nan)
    return series


def compute_avg_series(runs: List[Dict], corrupt_type: str, metric: str, exclude_model: str = "ByteCaption_XE") -> List[float]:
    """计算除BCM外所有模型的平均性能"""
    models = iter_models(runs)
    models = [m for m in models if m != exclude_model]
    
    avg_series = []
    for level_idx in range(len(LEVEL_ORDER)):
        values = []
        for model in models:
            series = build_series(runs, model, corrupt_type, metric)
            val = series[level_idx]
            if not np.isnan(val):
                values.append(val)
        
        if values:
            avg_series.append(np.mean(values))
        else:
            avg_series.append(np.nan)
    
    return avg_series


def plot_curves_quadrants(runs: List[Dict], output_path: Path):
    """
    绘制四象限曲线图（单一坐标系 + 坐标变换）
    
    布局（视觉等分四象限）：
    - 左半区：RBBF，横轴为S0→S5（从左往右递增）
    - 右半区：RBSL，横轴为S0→S5（从左往右递增）
    - 上半区：CIDEr，纵轴从下往上递增
    - 下半区：SPICE，纵轴从下往上递增
    
    关键技术：
    1. CIDEr/SPICE使用各自真实数值范围，但映射到相同的视觉高度
    2. x轴显示为S0-S5，然后回到S0-S5（两段都从左往右递增）
    3. 黑线位于画布正中央，将图面等分为视觉相同大小的4个象限
    """
    metrics = ["CIDEr", "SPICE"]
    corrupt_types = ["rbbf", "rbsl"]
    
    apply_plot_style()
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 收集所有数据并计算范围
    data = {}
    y_ranges = {}
    
    for metric in metrics:
        all_values = []
        for corrupt_type in corrupt_types:
            bcm_data = build_series(runs, "ByteCaption_XE", corrupt_type, metric)
            avg_data = compute_avg_series(runs, corrupt_type, metric)
            data[(corrupt_type, metric)] = {
                'bcm': np.array(bcm_data),
                'avg': np.array(avg_data)
            }
            all_values.extend([v for v in bcm_data if not np.isnan(v)])
            all_values.extend([v for v in avg_data if not np.isnan(v)])
        
        if all_values:
            # 指标均为百分比，期望从0起始，避免出现负刻度
            y_max = max(all_values)
            y_min = 0.0
            # 给上边界留少量空隙，但不改变下边界
            y_ranges[metric] = (y_min, y_max * 1.08 if y_max > 0 else 1.0)
        else:
            y_ranges[metric] = (0.0, 1.0)
    
    # 统一的视觉空间高度（上下各占相同高度）
    # 约定：
    # - CIDEr映射到 [0, +visual_y_range]
    # - SPICE映射到 [-visual_y_range, 0]
    visual_y_range = 100.0

    # x轴采用两段等距点：S0..S5（左半区），S0..S5（右半区）
    # 左半区: x=0..5    (RBBF)
    # 右半区: x=5.5..10.5 (RBSL)，让右侧S0贴近黑线x=5.5
    x_levels = np.arange(len(LEVEL_ORDER))
    x_base = {"rbbf": 0.0, "rbsl": 5.5}

    # y轴基线（用于将两个指标放入上下半区，但保持“向上递增”）
    y_base = {"CIDEr": 0.0, "SPICE": -visual_y_range}

    def map_metric_y(metric: str, y_val: float) -> float:
        """将真实metric值线性映射到统一视觉坐标。"""
        if np.isnan(y_val):
            return np.nan
        y_min, y_max = y_ranges[metric]
        if y_max <= y_min:
            return y_base[metric]
        t = (y_val - y_min) / (y_max - y_min)  # 0..1
        return y_base[metric] + t * visual_y_range

    # 四象限标签（仅用于展示）
    quadrant_labels = {
        ("CIDEr", "rbbf"): "(a) RBBF × CIDEr",
        ("CIDEr", "rbsl"): "(b) RBSL × CIDEr",
        ("SPICE", "rbbf"): "(c) RBBF × SPICE",
        ("SPICE", "rbsl"): "(d) RBSL × SPICE",
    }
    
    # 存储所有绘制的线条用于图例
    legend_handles = []
    legend_labels = []
    bcm_plotted = False
    avg_plotted = False
    
    for corrupt_type in corrupt_types:
        for metric in metrics:
            label = quadrant_labels[(metric, corrupt_type)]

            bcm_vals = data[(corrupt_type, metric)]["bcm"]
            avg_vals = data[(corrupt_type, metric)]["avg"]

            x_plot = x_base[corrupt_type] + x_levels

            # 变换y值
            bcm_y_visual = np.array([map_metric_y(metric, v) for v in bcm_vals])
            avg_y_visual = np.array([map_metric_y(metric, v) for v in avg_vals])

            # 过滤掉nan值用于绘图
            bcm_valid = ~np.isnan(bcm_y_visual)
            avg_valid = ~np.isnan(avg_y_visual)

            # BCM
            if bcm_valid.any():
                h_bcm = ax.plot(
                    x_plot[bcm_valid],
                    bcm_y_visual[bcm_valid],
                    marker="o",
                    linewidth=2.4,
                    alpha=0.85,
                    color=HIGHLIGHT_COLOR,
                    label="BCM (Ours)" if not bcm_plotted else "",
                    zorder=3,
                    clip_on=False,
                )[0]
                if not bcm_plotted:
                    legend_handles.append(h_bcm)
                    legend_labels.append("BCM (Ours)")
                    bcm_plotted = True

                ax.fill_between(
                    x_plot[bcm_valid],
                    y_base[metric],
                    bcm_y_visual[bcm_valid],
                    alpha=0.16,
                    color=HIGHLIGHT_COLOR,
                    zorder=1,
                )

            # Avg
            if avg_valid.any():
                h_avg = ax.plot(
                    x_plot[avg_valid],
                    avg_y_visual[avg_valid],
                    marker="s",
                    linewidth=2.0,
                    alpha=0.7,
                    color=AVG_COLOR,
                    label="Avg (w/o BCM)" if not avg_plotted else "",
                    linestyle="--",
                    zorder=3,
                    clip_on=False,
                )[0]
                if not avg_plotted:
                    legend_handles.append(h_avg)
                    legend_labels.append("Avg (w/o BCM)")
                    avg_plotted = True

                ax.fill_between(
                    x_plot[avg_valid],
                    y_base[metric],
                    avg_y_visual[avg_valid],
                    alpha=0.20,
                    color=AVG_COLOR,
                    zorder=1,
                )

            # 象限标签（放在每个象限的中上部，稍微往右移动）
            label_x = x_base[corrupt_type] + 3.2
            # 上半部分(CIDEr)标签往下调，下半部分(SPICE)保持原位
            label_y_offset = 0.68 if y_base[metric] >= 0 else 0.72
            label_y = y_base[metric] + visual_y_range * label_y_offset
            ax.text(
                label_x,
                label_y,
                label,
                fontsize=FONT_SIZES["tick_label"],
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="gray",
                    linewidth=1.5,
                ),
                ha="center",
                va="center",
                zorder=5,
            )
    
    # 设置坐标轴范围（保证黑线等分四象限 & 色块贴边）
    ax.set_xlim(0.0, 10.5)
    ax.set_ylim(-visual_y_range, visual_y_range)

    # X轴刻度：左段S0-S5（0..5），右段S0-S5（5.5..10.5）
    x_ticks = [float(i) for i in range(len(LEVEL_ORDER))] + [5.5 + float(i) for i in range(len(LEVEL_ORDER))]
    x_labels = LEVEL_ORDER + LEVEL_ORDER
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # Y轴刻度：同一条y轴上分段显示（避免y=0同时属于两种指标导致冲突）
    y_tick_positions: List[float] = []
    y_tick_labels: List[str] = []

    # SPICE（下半区）：[-100, 0)，从下往上递增
    spice_min, spice_max = y_ranges["SPICE"]
    spice_pos = np.linspace(-visual_y_range, 0.0, 5)[:-1]
    spice_vals = np.linspace(spice_min, spice_max, 5)[:-1]
    for pos, val in zip(spice_pos, spice_vals):
        y_tick_positions.append(float(pos))
        y_tick_labels.append(f"{val:.1f}")

    # y=0处添加"0.0"刻度标签（分割线）
    y_tick_positions.append(0.0)
    y_tick_labels.append("0.0")

    # CIDEr（上半区）：(0, +100]，从下往上递增
    cider_min, cider_max = y_ranges["CIDEr"]
    cider_pos = np.linspace(0.0, visual_y_range, 5)[1:]
    cider_vals = np.linspace(cider_min, cider_max, 5)[1:]
    for pos, val in zip(cider_pos, cider_vals):
        y_tick_positions.append(float(pos))
        y_tick_labels.append(f"{val:.1f}")

    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)

    # 禁用自动外边距，确保填充区域贴近坐标轴
    ax.margins(x=0, y=0)
    ax.autoscale(enable=False)
    
    # 网格
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, color='gray', zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor('#FAFAFA')
    
    # 添加象限分割参考线（细灰线）：正中等分
    ax.axhline(y=0.0, color="black", linewidth=1.3, alpha=0.7, zorder=2)
    ax.axvline(x=5.5, color="black", linewidth=1.3, alpha=0.7, zorder=2)
    
    # 标签和标题
    ax.set_xlabel("Corruption Severity", fontweight="bold", fontsize=FONT_SIZES["axes_label"])
    ax.set_title(
        "Captioning Metrics across Corruption Severity",
        fontweight="bold",
        fontsize=FONT_SIZES["title"],
        pad=20,
    )
    
    # 图例
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper right',
        framealpha=0.95,
        fontsize=FONT_SIZES["legend"],
    )
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Curves quadrants chart saved: {output_path}")
    plt.close()


def main():
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "all_models"
    
    if not input_dir.exists():
        print(f"[ERROR] Data directory not found: {input_dir}")
        return
    
    print(f"[INFO] Loading data from: {input_dir}")
    runs = load_runs(input_dir)
    
    if not runs:
        print("[ERROR] No valid JSON report files found")
        return
    
    print(f"[OK] Loaded {len(runs)} run data")
    
    output_path = Path(__file__).parent / "curves_combined.svg"
    plot_curves_quadrants(runs, output_path)


if __name__ == "__main__":
    main()
