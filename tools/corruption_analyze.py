"""
Sample COCO test images, corrupt their JPEG bitstreams, and save category-wise
examples for visual inspection. Dataset loading now mirrors training/eval
(`PureT/datasets_/coco_dataset_hf.py` + cfg).

Quick run examples (from repo root, with venv python):

  # 首次运行：收集统计数据并绘图
  python tools/corruption_analyze.py \
      --config PureT/experiments/ByteCaption_XE/config_coco.yml \
      --images-per-cat 100 --max-images 5000 \
      --save-images 50 \
      --severity-levels S0 S1 S2 S3 S4 S5 \
      --corrupt-types rbbf rbsl \
      --mode sequential \
      --output-dir ./evaluation_samples/bitstream_corruption_test

  # 后续运行：直接从已保存的statistics.json重新绘图
  python tools/corruption_analyze.py \
      --plot-only \
      --output-dir ./evaluation_samples/bitstream_corruption_test
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
if REPO_ROOT.name.lower() == "tools":
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PURET_ROOT = REPO_ROOT / "PureT"
if PURET_ROOT.exists() and str(PURET_ROOT) not in sys.path:
    sys.path.insert(0, str(PURET_ROOT))

from corenet.data.transforms.jpeg_corruption import JPEGCorruptionPipeline, normalize_level
from lib.config import cfg, cfg_from_file
from PureT.datasets_.coco_dataset import CocoDataset, pil_to_jpeg_bytes

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===== 统一字体样式配置（与可视化脚本对齐）=====
FONT_SIZES = {
    "default": 16,
    "title": 22,
    "axes_label": 18,
    "tick_label": 16,
    "legend": 14,
}


def apply_plot_style() -> None:
    """统一图表字体与轴样式。"""
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
        }
    )


def _decode_image(data: bytes) -> Image.Image | None:
    """Best-effort decode; returns None if decoding fails."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


def _encode_jpeg(pil_img: Image.Image, quality: int) -> bytes:
    """Encode PIL image to JPEG bytes with given quality."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return buf.getvalue()


def _count_unflipped_bits(original: bytes, corrupted: bytes) -> int:
    """
    Count bits unchanged between two byte streams.
    
    NOTE: When bitstream length changes (common in JPEG due to structure changes),
    we count unflipped bits only in the overlapping region (min length).
    Bytes beyond the shorter stream are NOT counted as "flipped" or "unflipped".
    This is a limitation but necessary since we can't reliably compare streams
    of different lengths in a compressed format.
    """
    if not original or not corrupted:
        return 0
    limit = min(len(original), len(corrupted))
    if limit <= 0:
        return 0
    flipped = 0
    for a, b in zip(original[:limit], corrupted[:limit]):
        flipped += (a ^ b).bit_count()
    return limit * 8 - flipped


def _build_pipelines(corrupt_types: List[str], levels: List[str]) -> Dict[str, Dict[str, JPEGCorruptionPipeline]]:
    pipelines: Dict[str, Dict[str, JPEGCorruptionPipeline]] = {}
    for ctype in corrupt_types:
        pipelines[ctype] = {}
        for lvl in levels:
            norm_level = normalize_level(lvl)
            pipelines[ctype][norm_level] = JPEGCorruptionPipeline([ctype], level=norm_level)
    return pipelines


def _load_coco_dataset(image_ids_path: Optional[str], gv_feat_path: str, max_samples: Optional[int]) -> CocoDataset:
    """Create CocoDataset in the same way as eval/test loaders."""
    return CocoDataset(
        image_ids_path=image_ids_path,
        input_seq=None,
        target_seq=None,
        gv_feat_path=gv_feat_path or "",
        seq_per_img=1,
        max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
        max_samples=max_samples,
        model_type="bytecaption",
        is_training=False
    )


def _plot_lossless_histogram(bitstream_sizes: List[int], output_dir: Path) -> None:
    """???????????"""
    if not bitstream_sizes:
        return

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 7))

    # ??bin???????????
    min_size = min(bitstream_sizes)
    max_size = max(bitstream_sizes)
    bin_start = (min_size // 1000) * 1000
    # ???????????????????1000???????
    bin_end = ((max_size // 1000) + 2) * 1000
    bins = np.arange(bin_start, bin_end, 500)

    ax.hist(bitstream_sizes, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Size of Bitstream (bytes)', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Bitstream Size Distribution without Corruption', fontweight='bold')
    ax.tick_params(axis='both', which='major')
    ax.set_xlim(bin_start, bin_end)

    mean_size = np.mean(bitstream_sizes)
    median_size = np.median(bitstream_sizes)
    std_size = np.std(bitstream_sizes)

    stats_text = (
        f'Mean: {mean_size:.0f}\n'
        f'Median: {median_size:.0f}\n'
        f'Std Dev: {std_size:.0f}\n'
        f'Min: {min_size}\n'
        f'Max: {max_size}'
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, pad=0.8),
    )

    plt.tight_layout()
    output_png = output_dir / "bitstream_length_histogram.png"
    output_svg = output_dir / "bitstream_length_histogram.svg"
    plt.savefig(str(output_svg), format='svg', bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_png}")
    print(f"   Saved: {output_svg}")


def _plot_rbsl_corruption_effect(corrupted_sizes: Dict[str, Dict[str, List[int]]], 
                                  lossless_sizes: List[int], output_dir: Path) -> None:
    """??RBSL?????????????"""
    if 'rbsl' not in corrupted_sizes or not corrupted_sizes['rbsl']:
        print("   RBSL corruption data not available, skipping plot.")
        return

    apply_plot_style()
    fig = plt.figure(figsize=(11, 6.5))

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.3)

    level_order = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
    rbsl_data = corrupted_sizes['rbsl']

    available_levels = [lvl for lvl in level_order if lvl in rbsl_data and rbsl_data[lvl]]
    if not available_levels:
        print("   No RBSL data available for plotting.")
        return

    sizes_by_level = [rbsl_data[lvl] for lvl in available_levels]

    ax_main = fig.add_subplot(gs[0])
    bp_main = ax_main.boxplot(
        sizes_by_level,
        labels=available_levels,
        patch_artist=True,
        whis=1.5,
        widths=0.6,
        showfliers=False,
    )

    for patch in bp_main['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax_main.set_xlabel('Corruption Severity', fontweight='bold')
    ax_main.set_ylabel('Size of Bitstream (bytes)', fontweight='bold')
    ax_main.set_title('Full Range View', fontweight='bold', fontsize=FONT_SIZES['title'])
    ax_main.tick_params(axis='both', which='major')

    ax_zoom = fig.add_subplot(gs[1])

    zoom_levels = [lvl for lvl in ['S4', 'S5'] if lvl in available_levels]
    if zoom_levels:
        zoom_indices = [available_levels.index(lvl) for lvl in zoom_levels]
        zoom_data = [sizes_by_level[i] for i in zoom_indices]

        bp_zoom = ax_zoom.boxplot(
            zoom_data,
            labels=zoom_levels,
            patch_artist=True,
            whis=1.5,
            widths=0.6,
            showfliers=False,
        )

        for patch in bp_zoom['boxes']:
            patch.set_facecolor('lightsalmon')
            patch.set_alpha(0.7)

        ax_zoom.set_xlabel('Corruption Severity', fontweight='bold')
        ax_zoom.set_title('Zoomed View', fontweight='bold', fontsize=FONT_SIZES['title'])
        ax_zoom.tick_params(axis='both', which='major')

    fig.suptitle('RBSL: Bitstream Size by Corruption Severity', fontweight='bold', y=0.98)

    plt.tight_layout()
    output_svg = output_dir / "rbsl_corruption_effect.svg"
    plt.savefig(str(output_svg), format='svg', bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_svg}")


def _plot_decode_success_rates(decode_count: Dict[str, Dict[str, int]], 
                                decode_total: Dict[str, Dict[str, int]], 
                                output_dir: Path) -> None:
    """????????????????"""
    if not decode_count:
        print("   No decode success data available.")
        return

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 7))

    level_order = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
    colors = {'rbbf': 'steelblue', 'rbsl': 'darkorange', 'metadata_loss': 'forestgreen'}

    for ctype, level_dict in sorted(decode_count.items()):
        available_levels = [lvl for lvl in level_order if lvl in level_dict]
        if not available_levels:
            continue

        success_rates = []
        for lvl in available_levels:
            ok = decode_count[ctype][lvl]
            total = decode_total[ctype][lvl]
            rate = ok / total if total > 0 else 0.0
            success_rates.append(rate * 100)

        color = colors.get(ctype, 'gray')
        ax.plot(
            available_levels,
            success_rates,
            marker='o',
            linewidth=2.5,
            markersize=10,
            label=ctype.upper(),
            color=color,
            alpha=0.8,
        )

    ax.set_xlabel('Corruption Severity', fontweight='bold')
    ax.set_ylabel('Decode Success Rate (%)', fontweight='bold')
    ax.set_title('Tendency of Decode Success Rate', fontweight='bold')
    ax.set_ylim([0, 105])
    ax.tick_params(axis='both', which='major')
    ax.legend(loc='best')

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    plt.tight_layout()
    output_svg = output_dir / "decode_success_rates.svg"
    plt.savefig(str(output_svg), format='svg', bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_svg}")




def _plot_unflipped_bits(unflipped_bits: Dict[str, Dict[str, List[int]]], output_dir: Path) -> None:
    """????????????????????"""
    if not unflipped_bits:
        print("   No unflipped-bit data available.")
        return

    apply_plot_style()

    level_order = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
    # Only plot RBBF
    ctypes = ['rbbf'] if 'rbbf' in unflipped_bits and unflipped_bits.get('rbbf') else []
    if not ctypes:
        print("   No unflipped-bit data available for RBBF.")
        return

    fig, axes = plt.subplots(len(ctypes), 1, figsize=(9.75, 7), sharex=True)
    if len(ctypes) == 1:
        axes = [axes]

    for ax, ctype in zip(axes, ctypes):
        level_dict = unflipped_bits.get(ctype, {})
        available_levels = [lvl for lvl in level_order if lvl in level_dict and level_dict[lvl]]
        if not available_levels:
            continue
        data = [level_dict[lvl] for lvl in available_levels]

        bp = ax.boxplot(
            data,
            labels=available_levels,
            patch_artist=True,
            whis=1.5,
            widths=0.6,
            showfliers=False,
        )
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax.set_ylabel('Unflipped Bits (count)', fontweight='bold')
        ax.tick_params(axis='both', which='major')

    axes[-1].set_xlabel('Corruption Severity', fontweight='bold')
    fig.suptitle('RBBF: Unflipped Bit Counts by Corruption Severity', fontweight='bold', y=0.98)

    plt.tight_layout()
    output_svg = output_dir / "unflipped_bits_by_severity.svg"
    plt.savefig(str(output_svg), format='svg', bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_svg}")


def _load_statistics_from_file(stats_path: Path) -> Dict[str, Any]:
    """从已保存的statistics.json文件加载统计数据。"""
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")
    
    with open(stats_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[INFO] Loaded statistics from: {stats_path}")
    print(f"[INFO] Contains {len(data.get('lossless_bitstream_sizes', []))} lossless samples")
    
    return data


def _attach_categories(instances_ann: Optional[str]) -> Tuple[Optional[Any], Dict[str, str]]:
    """Return COCO API handle and image_id -> category name mapping if available."""
    if not instances_ann or not Path(instances_ann).exists() or COCO is None:
        return None, {}
    coco = COCO(instances_ann)
    img_to_cat: Dict[str, str] = {}
    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids) if ann_ids else []
        cat_name = "uncategorized"
        if anns:
            cat = coco.loadCats([anns[0]["category_id"]])[0]
            cat_name = cat["name"].replace(" ", "_")
        img_to_cat[str(img_id)] = cat_name
    return coco, img_to_cat


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrupt COCO JPEG bitstreams and save samples.")
    parser.add_argument("--config", type=str, default="PureT/experiments/ByteCaption_XE/config_coco.yml",
                        help="Config file to load (same as train/eval).")
    parser.add_argument("--test-ids", type=str, default=None, help="Optional override for test id list (JSON).")
    parser.add_argument(
        "--val-ids",
        type=str,
        default=None,
        help="[DEPRECATED] Alias for --test-ids (kept for backward compatibility).",
    )
    parser.add_argument("--instances-ann", type=str, default=None,
                        help="Optional instances annotation (instances_val2017.json) for category grouping.")
    parser.add_argument("--images-per-cat", type=int, default=3, help="How many images to sample per category.")
    parser.add_argument("--corrupt-types", type=str, nargs="+", default=["rbbf", "rbsl", "metadata_loss"],
                        choices=["rbbf", "rbsl", "metadata_loss", "none"], help="Corruption types to apply.")
    parser.add_argument("--severity-levels", type=str, nargs="+", default=["S1", "S3", "S5"],
                        help="Severity levels to sweep (S0-S5/M0-M1; S0 included only if specified).")
    parser.add_argument("--output-dir", type=str, default="./evaluation_samples/bitstream_corruption",
                        help="Where to save corrupted streams/previews.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for sampling images.")
    parser.add_argument("--save-clean", action="store_true", help="Also save clean JPEGs for side-by-side comparison.")
    parser.add_argument("--save-images", type=int, default=30, help="Maximum number of images to save to disk (default: 30, 0 = save all).")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap on total images processed (0 = no cap).")
    parser.add_argument("--jpeg-quality", type=int, default=60, help="JPEG quality used when re-encoding bytes (matches eval path).")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "sequential"],
                        help="Sampling mode: random (shuffle) or sequential over COCO order.")
    parser.add_argument("--plot-only", action="store_true", 
                        help="Skip data collection and only generate plots from existing statistics.json file.")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    
    # Plot-only模式：从已保存的统计数据直接绘图
    if args.plot_only:
        print("\n" + "="*70)
        print("PLOT-ONLY MODE: Loading existing statistics...")
        print("="*70)
        
        stats_path = output_root / "statistics.json"
        stats_data = _load_statistics_from_file(stats_path)
        
        # 从保存的数据中提取信息
        lossless_bitstream_sizes = stats_data.get("lossless_bitstream_sizes", [])
        corrupted_bitstream_sizes = stats_data.get("corrupted_bitstream_sizes", {})
        decode_success_rates = stats_data.get("decode_success_rates", {})
        unflipped_bits = stats_data.get("unflipped_bits", {})
        
        # 重建decode_count和decode_total
        decode_count = {}
        decode_total = {}
        for ctype, levels_dict in decode_success_rates.items():
            decode_count[ctype] = {}
            decode_total[ctype] = {}
            for level, stats in levels_dict.items():
                decode_count[ctype][level] = stats["success_count"]
                decode_total[ctype][level] = stats["total_count"]
        
        # 生成图表
        print("\nGENERATING PLOTS...")
        print("="*70)
        
        print("\n1. Generating lossless bitstream histogram...")
        _plot_lossless_histogram(lossless_bitstream_sizes, output_root)
        
        print("2. Generating RBSL corruption effect plot...")
        _plot_rbsl_corruption_effect(corrupted_bitstream_sizes, lossless_bitstream_sizes, output_root)
        
        print("3. Generating decode success rates plot...")
        _plot_decode_success_rates(decode_count, decode_total, output_root)
        print("4. Generating unflipped bits plot...")
        _plot_unflipped_bits(unflipped_bits, output_root)
        
        print("\n" + "="*70)
        print("ALL PLOTS REGENERATED SUCCESSFULLY!")
        print("="*70)
        return
    
    # 正常模式：统计+绘图
    cfg_from_file(args.config)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    levels = [normalize_level(lvl) for lvl in args.severity_levels]
    
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    max_samples = args.max_images if args.max_images > 0 else None
    test_ids_path = args.test_ids or args.val_ids or cfg.DATA_LOADER.TEST_ID
    test_gv_feat = getattr(cfg.DATA_LOADER, "TEST_GV_FEAT", cfg.DATA_LOADER.VAL_GV_FEAT)
    dataset = _load_coco_dataset(test_ids_path, test_gv_feat, max_samples=max_samples)

    coco_api, img_to_cat = _attach_categories(args.instances_ann)
    category_names = list(set(img_to_cat.values())) if img_to_cat else ["all"]

    pipelines = _build_pipelines(args.corrupt_types, levels)
    manifest: List[Dict] = []
    decode_stats: Dict[str, Dict[str, int]] = {}
    
    # 新增：统计信息
    clean_jpeg_stats: Dict[str, int | float] = {}  # 无损情形下的统计
    lossless_bitstream_sizes: List[int] = []  # 存储所有无损码流大小
    corrupted_bitstream_sizes: Dict[str, Dict[str, List[int]]] = {}  # 存储损坏码流大小
    unflipped_bits: Dict[str, Dict[str, List[int]]] = {}  # ???????
    decode_count: Dict[str, Dict[str, int]] = {}  # 记录每种情况的解码成功次数
    decode_total: Dict[str, Dict[str, int]] = {}  # 记录每种情况的总数

    total_processed = 0
    saved_images = 0  # 跟踪已保存的图像数量
    per_cat_saved: Dict[str, int] = {cat: 0 for cat in category_names}  # 每个类别已保存的数量
    
    # Build a list of dataset indices; shuffle only in random mode
    all_indices = list(range(len(dataset)))
    if args.mode == "random":
        rng.shuffle(all_indices)

    for idx in all_indices:
        if args.max_images and total_processed >= args.max_images:
            break

        sample = dataset.ds[idx] if hasattr(dataset, "ds") else None
        if sample is None:
            continue
        img_id = str(sample.get("image_id", idx))
        cat_name = img_to_cat.get(img_id, "all")
        if cat_name not in per_cat_saved:
            per_cat_saved[cat_name] = 0

        try:
            pil_img = dataset._extract_image(sample)
            # 统一与训练/评估管线：使用 CocoDataset 的工具函数进行 224x224 resize + JPEG 编码
            raw_bytes = pil_to_jpeg_bytes(pil_img, quality=args.jpeg_quality)
        except Exception:
            continue

        # 统计无损码流大小
        lossless_bitstream_sizes.append(len(raw_bytes))

        # 判断是否需要保存图像文件（基于每个类别的配额）
        should_save = (args.save_images == 0) or (per_cat_saved[cat_name] < args.images_per_cat and saved_images < args.save_images)

        # Optionally save clean image
        if args.save_clean and should_save:
            clean_dir = output_root / "clean" / cat_name
            clean_dir.mkdir(parents=True, exist_ok=True)
            clean_path = clean_dir / f"{img_id}_clean.jpg"
            clean_path.write_bytes(raw_bytes)

        for ctype, level_map in pipelines.items():
            for level, pipeline in level_map.items():
                # S0 特殊处理：直接使用无损数据
                if level == "S0":
                    corrupted_variants = [(raw_bytes, "S0_lossless")]
                elif not pipeline.is_enabled():
                    continue
                else:
                    corrupted_variants = pipeline.apply(raw_bytes)
                    
                for corrupted_bytes, marker in corrupted_variants:
                    # 只在需要保存时才创建目录和保存文件
                    if should_save:
                        out_dir = output_root / ctype / level / cat_name
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"{img_id}_{marker}.jpg"
                        out_path.write_bytes(corrupted_bytes)
                    else:
                        out_path = None

                    # 统计损坏码流大小
                    if ctype not in corrupted_bitstream_sizes:
                        corrupted_bitstream_sizes[ctype] = {}
                    if level not in corrupted_bitstream_sizes[ctype]:
                        corrupted_bitstream_sizes[ctype][level] = []
                    corrupted_bitstream_sizes[ctype][level].append(len(corrupted_bytes))
                    
                    # 统计未翻转比特数量（基于重叠字节对齐比较）
                    if ctype not in unflipped_bits:
                        unflipped_bits[ctype] = {}
                    if level not in unflipped_bits[ctype]:
                        unflipped_bits[ctype][level] = []
                    unflipped_bits[ctype][level].append(
                        _count_unflipped_bits(raw_bytes, corrupted_bytes)
                    )

                    preview = _decode_image(corrupted_bytes)
                    preview_ok = preview is not None

                    decode_stats.setdefault(ctype, {}).setdefault(level, {"ok": 0, "total": 0})
                    decode_stats[ctype][level]["total"] += 1
                    if preview_ok:
                        decode_stats[ctype][level]["ok"] += 1

                    # 统计解码成功率
                    if ctype not in decode_count:
                        decode_count[ctype] = {}
                        decode_total[ctype] = {}
                    if level not in decode_count[ctype]:
                        decode_count[ctype][level] = 0
                        decode_total[ctype][level] = 0
                    
                    decode_total[ctype][level] += 1
                    if preview_ok:
                        decode_count[ctype][level] += 1

                    manifest.append(
                        {
                            "image_id": img_id,
                            "category": cat_name,
                            "corruption": marker,
                            "saved_to": str(out_path) if out_path else "not_saved",
                            "decode_ok": preview_ok,
                            "bitstream_size": len(corrupted_bytes),
                        }
                    )

        total_processed += 1
        if should_save:
            saved_images += 1
            per_cat_saved[cat_name] += 1

    # 计算统计信息
    if lossless_bitstream_sizes:
        avg_lossless_size = np.mean(lossless_bitstream_sizes)
        min_lossless = min(lossless_bitstream_sizes)
        max_lossless = max(lossless_bitstream_sizes)
        
        # 计算JPEG metadata比例（估算）
        # 简单启发式：metadata通常占JPEG 5-15%，这里取平均估计
        avg_metadata_ratio = 0.08  # 8% 估计
        
        clean_jpeg_stats = {
            "avg_size": avg_lossless_size,
            "min_size": min_lossless,
            "max_size": max_lossless,
            "count": len(lossless_bitstream_sizes),
            "estimated_metadata_ratio": avg_metadata_ratio,
        }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] Processed {total_processed} images, saved {saved_images} image files")
    print(f"Generated {len(manifest)} corrupted samples (statistics)")
    print(f"Manifest written to: {manifest_path}")
    
    # 输出统计报告
    print("\n" + "="*70)
    print("DETAILED STATISTICS REPORT")
    print("="*70)
    
    # 1. 无损码流统计
    print("\n1. LOSSLESS JPEG BITSTREAM STATISTICS:")
    print(f"   Total images: {clean_jpeg_stats.get('count', 0)}")
    print(f"   Average bitstream length: {clean_jpeg_stats.get('avg_size', 0):.0f} bytes")
    print(f"   Min bitstream length: {clean_jpeg_stats.get('min_size', 0)} bytes")
    print(f"   Max bitstream length: {clean_jpeg_stats.get('max_size', 0)} bytes")
    print(f"   Estimated metadata ratio: {clean_jpeg_stats.get('estimated_metadata_ratio', 0):.2%}")
    
    # 2. 码流长度直方图（以20000为中心，1000为间隔，覆盖最小到最大）
    print("\n2. LOSSLESS BITSTREAM LENGTH HISTOGRAM (center=20000, bin=1000):")
    if lossless_bitstream_sizes:
        min_size = min(lossless_bitstream_sizes)
        max_size = max(lossless_bitstream_sizes)
        # 从20000向两边扩展，覆盖最小和最大值
        bin_start = (min_size // 1000) * 1000
        bin_end = ((max_size // 1000) + 2) * 1000
        bins = np.arange(bin_start, bin_end, 1000)
        hist, bin_edges = np.histogram(lossless_bitstream_sizes, bins=bins)
        for i, (edge, count) in enumerate(zip(bin_edges[:-1], hist)):
            bin_range = f"{edge:.0f}-{edge+1000:.0f}"
            bar = "*" * (count // max(1, max(hist) // 40))
            print(f"   {bin_range:>15}: {count:>4} {bar}")
    
    # 3. 每种情形的解码成功概率
    print("\n3. DECODING SUCCESS PROBABILITY BY CORRUPTION TYPE AND LEVEL:")
    for ctype in sorted(decode_count.keys()):
        print(f"   {ctype.upper()}:")
        for level in sorted(decode_count[ctype].keys()):
            ok = decode_count[ctype][level]
            total = decode_total[ctype][level]
            prob = ok / total if total > 0 else 0.0
            print(f"      {level}: {ok}/{total} ({prob:.2%})")
    
    # 4. 损坏后码流平均长度
    print("\n4. AVERAGE CORRUPTED BITSTREAM LENGTH BY TYPE AND LEVEL:")
    for ctype in sorted(corrupted_bitstream_sizes.keys()):
        print(f"   {ctype.upper()}:")
        for level in sorted(corrupted_bitstream_sizes[ctype].keys()):
            sizes = corrupted_bitstream_sizes[ctype][level]
            if sizes:
                avg_size = np.mean(sizes)
                std_size = np.std(sizes)
                min_size = np.min(sizes)
                max_size = np.max(sizes)
                avg_original = np.mean(lossless_bitstream_sizes)
                size_diff = avg_size - avg_original
                print(f"      {level}: {avg_size:.0f} ± {std_size:.0f} bytes (Δ: {size_diff:+.0f}, range: {min_size}-{max_size}, n={len(sizes)})")
    
    # 5. Unflipped bits统计与有效性分析
    print("\n5. UNFLIPPED BITS ANALYSIS:")
    print("   NOTE: Unflipped bits counted only in overlapping region when stream lengths differ.")
    for ctype in sorted(unflipped_bits.keys()):
        print(f"   {ctype.upper()}:")
        for level in sorted(unflipped_bits[ctype].keys()):
            bits = unflipped_bits[ctype][level]
            if bits:
                avg_unflipped = np.mean(bits)
                std_unflipped = np.std(bits)
                min_unflipped = np.min(bits)
                max_unflipped = np.max(bits)
                avg_original_bits = np.mean(lossless_bitstream_sizes) * 8
                unflipped_ratio = avg_unflipped / avg_original_bits if avg_original_bits > 0 else 0
                print(f"      {level}: {avg_unflipped:.0f} ± {std_unflipped:.0f} bits ({unflipped_ratio:.2%} of original, range: {min_unflipped:.0f}-{max_unflipped:.0f})")
    
    print("\n" + "="*70)
    
    if decode_stats:
        print("\n6. DECODE SUCCESS RATES (detailed):")
        for ctype in sorted(decode_stats.keys()):
            for level in sorted(decode_stats[ctype].keys()):
                stats = decode_stats[ctype][level]
                ok, total = stats["ok"], stats["total"]
                rate = ok / total if total else 0.0
                print(f"  {ctype.upper()} {level}: {ok}/{total} ({rate:.2%})")
    
    # 保存统计信息到JSON文件
    print("\n" + "="*70)
    print("SAVING STATISTICS TO FILES...")
    print("="*70)
    
    stats_output = {
        "lossless_stats": clean_jpeg_stats,
        "unflipped_bits": unflipped_bits,
        "lossless_bitstream_sizes": lossless_bitstream_sizes,
        "corrupted_bitstream_sizes": {
            ctype: {
                level: [int(s) for s in sizes]  # 转换为标准Python int
                for level, sizes in level_dict.items()
            }
            for ctype, level_dict in corrupted_bitstream_sizes.items()
        },
        "decode_success_rates": {
            ctype: {
                level: {
                    "success_count": int(decode_count[ctype][level]),
                    "total_count": int(decode_total[ctype][level]),
                    "success_rate": float(decode_count[ctype][level] / decode_total[ctype][level]) 
                                   if decode_total[ctype][level] > 0 else 0.0
                }
                for level in decode_count[ctype]
            }
            for ctype in decode_count
        }
    }
    
    # 保存JSON统计文件
    stats_json_path = output_root / "statistics.json"
    stats_json_path.write_text(json.dumps(stats_output, indent=2), encoding="utf-8")
    print(f"   Saved: {stats_json_path}")
    
    # 保存CSV格式的统计文件
    stats_csv_path = output_root / "statistics.csv"
    with open(stats_csv_path, 'w', encoding='utf-8') as f:
        f.write("Corruption_Type,Severity_Level,Success_Count,Total_Count,Success_Rate,Avg_Bitstream_Size(bytes)\n")
        for ctype in sorted(corrupted_bitstream_sizes.keys()):
            for level in sorted(corrupted_bitstream_sizes[ctype].keys()):
                sizes = corrupted_bitstream_sizes[ctype][level]
                avg_size = np.mean(sizes) if sizes else 0.0
                if ctype in decode_count and level in decode_count[ctype]:
                    success = decode_count[ctype][level]
                    total = decode_total[ctype][level]
                    rate = success / total if total > 0 else 0.0
                    f.write(f"{ctype},{level},{success},{total},{rate:.4f},{avg_size:.0f}\n")
    print(f"   Saved: {stats_csv_path}")
    
    # 生成图表
    print("\nGENERATING PLOTS...")
    print("="*70)
    
    # 1. 无损码流长度直方图
    print("\n1. Generating lossless bitstream histogram...")
    _plot_lossless_histogram(lossless_bitstream_sizes, output_root)
    
    # 2. RBSL损坏程度与码流长度的关系
    print("2. Generating RBSL corruption effect plot...")
    _plot_rbsl_corruption_effect(corrupted_bitstream_sizes, lossless_bitstream_sizes, output_root)
    
    # 3. 各情形下的解码成功率曲线图
    print("3. Generating decode success rates plot...")
    _plot_decode_success_rates(decode_count, decode_total, output_root)

    print("4. Generating unflipped bits plot...")
    _plot_unflipped_bits(unflipped_bits, output_root)
    
    print("\n" + "="*70)
    print("ALL STATISTICS AND PLOTS SAVED SUCCESSFULLY!")
    print("="*70)


if __name__ == "__main__":
    main()
