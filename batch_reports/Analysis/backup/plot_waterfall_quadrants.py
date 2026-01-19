"""
四象限色块趋势图 - 瀑布式展现BCM vs Avg的鲁棒性
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
from typing import Dict, List, Iterable


# 从原始脚本借用的辅助函数
LEVEL_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5"]


def create_step_data(x_orig: np.ndarray, y_orig: np.ndarray, x_smooth: np.ndarray) -> np.ndarray:
    """
    创建严格单调递减的平坦台阶曲线，每个等级内平坦，过渡处平滑下降
    """
    if len(x_orig) < 2:
        return np.interp(x_smooth, x_orig, y_orig)
    
    # 为每个平坦段和过渡段创建控制点
    x_points = []
    y_points = []
    
    for i in range(len(x_orig)):
        x_curr = x_orig[i]
        y_curr = y_orig[i]
        
        if i == 0:
            # 第一个点的起始位置
            x_points.append(x_curr - 0.35)
            y_points.append(y_curr)
        
        # 当前点左侧边界（平坦）
        x_points.append(x_curr - 0.25)
        y_points.append(y_curr)
        
        # 当前点
        x_points.append(x_curr)
        y_points.append(y_curr)
        
        # 当前点右侧边界（平坦）
        x_points.append(x_curr + 0.25)
        y_points.append(y_curr)
        
        if i < len(x_orig) - 1:
            # 下降过渡区域 - 从当前值线性下降到下一个值
            x_next = x_orig[i + 1]
            x_transition_mid = (x_curr + x_next) / 2
            # 添加过渡中点（一定要在两个值之间）
            y_mid = (y_curr + y_orig[i + 1]) / 2
            x_points.append(x_transition_mid)
            y_points.append(y_mid)
    
    # 最后一个点的结束位置
    x_points.append(x_orig[-1] + 0.35)
    y_points.append(y_orig[-1])
    
    # 确保严格单调递减
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    
    # 确保y值严格递减（对于有重复x的部分）
    for i in range(1, len(y_points)):
        if y_points[i] > y_points[i-1]:
            y_points[i] = y_points[i-1]
    
    # 使用线性插值保证单调性
    result = np.interp(x_smooth, x_points, y_points)
    
    # 二次检查确保严格递减
    result = np.clip(result, y_orig[-1], y_orig[0])
    
    return result


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


def plot_waterfall_quadrants(runs: List[Dict], output_path: Path):
    """
    绘制四象限瀑布式色块图 - 统一坐标系
    - 第一象限（右上）：RBSL + CIDEr
    - 第二象限（左上）：RBBF + CIDEr  
    - 第三象限（左下）：RBBF + SPICE
    - 第四象限（右下）：RBSL + SPICE
    - 中心原点为(0,0)，统一坐标轴
    """
    metrics = ["CIDEr", "SPICE"]
    corrupt_types = ["rbbf", "rbsl"]
    
    # 创建单个统一坐标系
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # 色彩方案：BCM用深蓝色系，Avg用橙金色系
    color_bcm = "#2E5C8A"  # 深蓝
    color_avg = "#E67E22"  # 橙金
    alpha_bcm = 0.65
    alpha_avg = 0.50
    
    # 四个象限的配置：(metric_idx, corrupt_type_idx, x_sign, y_sign, label)
    # 左上角开始逆时针：a(左上) -> b(右上) -> c(右下) -> d(左下)
    quadrants = [
        (0, 0, -1, 1, "(a) RBBF × CIDEr"),   # (a) 第二象限：左上
        (0, 1, 1, 1, "(b) RBSL × CIDEr"),    # (b) 第一象限：右上
        (1, 1, 1, -1, "(c) RBSL × SPICE"),   # (c) 第四象限：右下
        (1, 0, -1, -1, "(d) RBBF × SPICE"),  # (d) 第三象限：左下
    ]
    
    legend_added = False
    
    for metric_idx, corrupt_idx, x_sign, y_sign, label in quadrants:
        metric = metrics[metric_idx]
        corrupt_type = corrupt_types[corrupt_idx]
        
        # 获取BCM和Avg的数据
        bcm_values = compute_relative_series(runs, "ByteCaption_XE", corrupt_type, metric)
        avg_values = compute_avg_relative_series(runs, corrupt_type, metric)
        
        # X轴位置 (S0到S5，映射到0-5)
        x_orig = np.arange(len(LEVEL_ORDER))
        x_smooth = np.linspace(0, len(LEVEL_ORDER)-1, 100)
        
        # 应用象限变换
        x_orig_plot = x_orig * x_sign
        x_smooth_plot = x_smooth * x_sign
        
        # 处理Avg数据（底层）- 从0填充到当前值
        if not all(np.isnan(v) for v in avg_values):
            y_avg = np.array([max(0, v) if not np.isnan(v) else 0 for v in avg_values])
            # 使用阶梯插值 - 在每个点保持平坦，然后像台阶一样下到下一个值
            y_avg_smooth = create_step_data(x_orig, y_avg, x_smooth)
            y_avg_smooth = np.clip(y_avg_smooth, 0, 100)
            
            y_avg_smooth_plot = y_avg_smooth * y_sign
            y_bottom = np.zeros_like(x_smooth)
            y_avg_plot = y_avg * y_sign
            
            # 先绘制竖直条形表示离散性（半透明，宽度为0.3）
            for i, (x_pos, y_val) in enumerate(zip(x_orig, y_avg)):
                x_pos_plot = x_pos * x_sign
                y_val_plot = y_val * y_sign
                ax.bar(x_pos_plot, y_val_plot, width=0.3, bottom=0,
                      color=color_avg, alpha=0.25, edgecolor='none', zorder=1)
            
            # 绘制瀑布（从0到当前值）
            ax.fill_between(x_smooth_plot, y_bottom, y_avg_smooth_plot, 
                           color=color_avg, alpha=alpha_avg, 
                           label="Avg (w/o BCM)" if not legend_added else "", 
                           linewidth=0,
                           edgecolor='none')
            
            # 添加边缘轮廓线
            ax.plot(x_smooth_plot, y_avg_smooth_plot, color=color_avg, linewidth=2.5, alpha=0.9, zorder=3)
        
        # 处理BCM数据（上层）- 从0填充到当前值
        if not all(np.isnan(v) for v in bcm_values):
            y_bcm = np.array([max(0, v) if not np.isnan(v) else 0 for v in bcm_values])
            # 使用阶梯插值 - 在每个点保持平坦，然后像台阶一样下到下一个值
            y_bcm_smooth = create_step_data(x_orig, y_bcm, x_smooth)
            y_bcm_smooth = np.clip(y_bcm_smooth, 0, 100)
            
            y_bcm_smooth_plot = y_bcm_smooth * y_sign
            y_bottom = np.zeros_like(x_smooth)
            y_bcm_plot = y_bcm * y_sign
            
            # 先绘制竖直条形表示离散性（半透明，宽度为0.3）
            for i, (x_pos, y_val) in enumerate(zip(x_orig, y_bcm)):
                x_pos_plot = x_pos * x_sign
                y_val_plot = y_val * y_sign
                ax.bar(x_pos_plot, y_val_plot, width=0.3, bottom=0,
                      color=color_bcm, alpha=0.2, edgecolor='none', zorder=1)
            
            # 绘制瀑布
            ax.fill_between(x_smooth_plot, y_bottom, y_bcm_smooth_plot, 
                           color=color_bcm, alpha=alpha_bcm, 
                           label="BCM (Ours)" if not legend_added else "", 
                           linewidth=0,
                           edgecolor='none')
            
            # 添加边缘轮廓线
            ax.plot(x_smooth_plot, y_bcm_smooth_plot, color=color_bcm, linewidth=3, alpha=1.0, zorder=4)
        
        legend_added = True
        
        # 添加象限标签 - 靠近坐标轴，不超出色块范围
        label_x = x_sign * 2.5  # 更靠近坐标轴
        label_y = y_sign * 75   # 更靠近坐标轴
        ax.text(label_x, label_y, label,
               fontsize=13, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5))
    
    # 设置统一的坐标轴范围
    axis_limit = len(LEVEL_ORDER) + 0.5
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-105, 105)
    
    # 设置主刻度和网格
    # X轴：在两侧显示S0-S5的刻度
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
    
    # 坐标轴标签
    ax.set_xlabel('Corruption Severity', fontsize=13, fontweight='semibold')
    ax.set_ylabel('Relative Metric (%)', fontsize=13, fontweight='semibold')
    
    # 总标题
    ax.set_title('Relative Metrics: BCM vs Avg over Four Quadrants', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 图例
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, 
                 loc='upper right', 
                 frameon=True, framealpha=0.95,
                 fontsize=11,
                 edgecolor='gray',
                 shadow=True)
    
    plt.tight_layout()
    fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 四象限瀑布图已保存: {output_path}")


def main():
    # 定位数据文件
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
    
    output_path = Path(__file__).parent / "relative_waterfall_quadrants.svg"
    plot_waterfall_quadrants(runs, output_path)


if __name__ == "__main__":
    main()
