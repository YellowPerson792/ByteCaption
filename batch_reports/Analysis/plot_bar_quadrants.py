"""
四象限柱状图 - 中心对称展现BCM vs Avg的相对性能
BCM和Avg为紧贴着的两列柱子
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Iterable


# 从原始脚本借用的辅助函数
LEVEL_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5"]


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


def iter_models(runs: Iterable[Dict]) -> List[str]:
    """获取所有模型名称"""
    model_set = {r.get("model_name") for r in runs if r.get("model_name")}
    models = sorted(model_set)
    # 确保BCM在前
    if "ByteCaption_XE" in models:
        models.remove("ByteCaption_XE")
        models.insert(0, "ByteCaption_XE")
    return models


def iter_corrupt_types(runs: Iterable[Dict]) -> List[str]:
    """获取所有损坏类型"""
    ct_set = {r.get("corrupt_type") for r in runs if r.get("corrupt_type")}
    return sorted(ct_set)


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
                    value = metrics_data[metric]
                    break
        series.append(value)
    return series


def compute_relative_series(runs: List[Dict], model: str, corrupt_type: str, metric: str) -> List[float]:
    """计算相对于S0的百分比"""
    series = build_series(runs, model, corrupt_type, metric)
    s0 = series[0]
    if s0 is None or np.isnan(s0) or s0 == 0:
        return [np.nan] * len(series)
    
    relative = [100.0]  # S0为100%
    for sx in series[1:]:
        if sx is None or np.isnan(sx):
            relative.append(np.nan)
        else:
            relative.append((sx / s0) * 100.0)
    return relative


def compute_avg_relative_series(runs: List[Dict], corrupt_type: str, metric: str, exclude_model: str = "ByteCaption_XE") -> List[float]:
    """计算除BCM外所有模型的平均相对性能"""
    models = iter_models(runs)
    models = [m for m in models if m != exclude_model]
    
    avg_relative = [100.0]  # S0为100%
    for level_idx in range(1, len(LEVEL_ORDER)):
        values = []
        for model in models:
            series = build_series(runs, model, corrupt_type, metric)
            s0 = series[0]
            if s0 is None or np.isnan(s0) or s0 == 0:
                continue
            sx = series[level_idx]
            if sx is None or np.isnan(sx):
                continue
            values.append((sx / s0) * 100.0)
        
        if values:
            avg_relative.append(np.mean(values))
        else:
            avg_relative.append(np.nan)
    
    return avg_relative


def plot_bar_quadrants(runs: List[Dict], output_path: Path):
    """
    绘制四象限柱状图 - 真正的中心对称坐标系
    
    布局：
    - 左上象限：RBBF × CIDEr
    - 右上象限：RBSL × CIDEr  
    - 左下象限：RBBF × SPICE
    - 右下象限：RBSL × SPICE
    
    x轴：-3到+3，从S5到S0，中心0为S0
    y轴：-105到+105，正值向上，负值向下
    """
    metrics = ["CIDEr", "SPICE"]
    corrupt_types = ["rbbf", "rbsl"]
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # 色彩方案
    color_bcm = "#2E5C8A"  # 深蓝
    color_avg = "#E67E22"  # 橙金
    
    # 获取数据
    data = {}
    for corrupt_type in corrupt_types:
        for metric in metrics:
            bcm_data = compute_relative_series(runs, "ByteCaption_XE", corrupt_type, metric)
            avg_data = compute_avg_relative_series(runs, corrupt_type, metric)
            data[(corrupt_type, metric)] = {
                'bcm': np.array([v if not np.isnan(v) else 0 for v in bcm_data]),
                'avg': np.array([v if not np.isnan(v) else 0 for v in avg_data])
            }
    
    # 柱子配置：(metric, corrupt_type, y_sign, quadrant_label)
    # y_sign决定正值向上还是向下
    configs = [
        ("CIDEr", "rbbf", 1, "(a) RBBF × Rel-CIDEr"),    # 左上
        ("CIDEr", "rbsl", 1, "(b) RBSL × Rel-CIDEr"),     # 右上
        ("SPICE", "rbbf", -1, "(d) RBBF × Rel-SPICE"),    # 左下
        ("SPICE", "rbsl", -1, "(c) RBSL × Rel-SPICE"),    # 右下
    ]
    
    # 柱子宽度
    bar_width = 0.3
    
    # 绘制四个象限
    for metric, corrupt_type, y_sign, label in configs:
        bcm_vals = data[(corrupt_type, metric)]['bcm']
        avg_vals = data[(corrupt_type, metric)]['avg']
        
        # 确定x方向：rbbf左边（负），rbsl右边（正）
        x_sign = -1 if corrupt_type == "rbbf" else 1
        
        # X轴位置 (S0到S5，映射到0-5)
        x_orig = np.arange(len(LEVEL_ORDER))
        x_orig_plot = x_orig * x_sign
        
        for i in range(len(LEVEL_ORDER)):
            # x坐标：应用象限变换
            x_center = x_orig_plot[i]
            
            # BCM柱子（左侧）
            bcm_val = bcm_vals[i] * y_sign
            x_bcm = x_center - bar_width/2
            ax.bar(x_bcm, bcm_val, width=bar_width, color=color_bcm,
                  alpha=0.8, edgecolor='darkblue', linewidth=0.5, zorder=2)
            
            # Avg柱子（右侧）
            avg_val = avg_vals[i] * y_sign
            x_avg = x_center + bar_width/2
            ax.bar(x_avg, avg_val, width=bar_width, color=color_avg,
                  alpha=0.8, edgecolor='darkorange', linewidth=0.5, zorder=2)
        
        # 添加象限标签
        if corrupt_type == "rbbf":
            label_x = -2.5
        else:
            label_x = 2.5
        
        if y_sign > 0:
            label_y = 75
        else:
            label_y = -75
        
        ax.text(label_x, label_y, label, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9),
               ha='center', va='center', zorder=5)
    
    # 设置统一的坐标轴范围
    axis_limit = len(LEVEL_ORDER) + 0.5
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-110, 110)
    
    # X轴：在两侧显示S0-S5的刻度（参考waterfall）
    x_orig = np.arange(len(LEVEL_ORDER))
    x_ticks = list(-x_orig) + list(x_orig)
    ax.set_xticks(sorted(set(x_ticks)))
    ax.set_xticklabels([LEVEL_ORDER[abs(int(x))] for x in sorted(set(x_ticks))], fontsize=11)
    
    # Y轴刻度
    y_ticks = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(abs(y)) for y in y_ticks], fontsize=10)
    
    # 网格
    ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8, color='gray', zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor('#FAFAFA')
    
    # 添加坐标轴参考线（粗黑线）
    ax.axhline(y=0, color='black', linewidth=2, alpha=0.8, zorder=2)
    ax.axvline(x=0, color='black', linewidth=2, alpha=0.8, zorder=2)
    
    # 标签和标题
    ax.set_xlabel("Corruption Severity", fontsize=13, fontweight='semibold')
    ax.set_ylabel("Relative Metric (%)", fontsize=13, fontweight='semibold')
    ax.set_title("Relative Metrics: BCM vs Avg over Four Quadrants",
                fontsize=16, fontweight='bold', pad=20)
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_bcm, alpha=0.8, edgecolor='darkblue', label='BCM (Ours)'),
        Patch(facecolor=color_avg, alpha=0.8, edgecolor='darkorange', label='Avg (w/o BCM)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Bar chart saved: {output_path}")
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
    
    output_path = Path(__file__).parent / "relative_bar_quadrants.svg"
    plot_bar_quadrants(runs, output_path)


if __name__ == "__main__":
    main()
