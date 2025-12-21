"""
Sample COCO test images, corrupt their JPEG bitstreams, and save category-wise
examples for visual inspection. Dataset loading now mirrors training/eval
(`PureT/datasets_/coco_dataset_hf.py` + cfg).

Quick run examples (from repo root, with venv python):
  python tools/corrupt_coco_bitstreams.py \
      --config PureT/experiments/ByteCaption_XE/config_coco.yml \
      --images-per-cat 10 --max-images 200 \
      --severity-levels S1 S2 S3 S4 S5 \
      --corrupt-types rbbf rbsl \
      --mode sequential \
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
from PureT.datasets_.coco_dataset_hf import CocoDataset

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    )


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
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap on total images processed (0 = no cap).")
    parser.add_argument("--jpeg-quality", type=int, default=60, help="JPEG quality used when re-encoding bytes (matches eval path).")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "sequential"],
                        help="Sampling mode: random (shuffle) or sequential over COCO order.")
    args = parser.parse_args()

    cfg_from_file(args.config)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    levels = [normalize_level(lvl) for lvl in args.severity_levels]
    output_root = Path(args.output_dir)
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

    total_processed = 0
    # Build a list of dataset indices; shuffle only in random mode
    all_indices = list(range(len(dataset)))
    if args.mode == "random":
        rng.shuffle(all_indices)

    # Helper to map to category and select quota
    per_cat_count: Dict[str, int] = {cat: 0 for cat in category_names}

    for idx in all_indices:
        if args.max_images and total_processed >= args.max_images:
            break

        sample = dataset.ds[idx] if hasattr(dataset, "ds") else None
        if sample is None:
            continue
        img_id = str(sample.get("image_id", idx))
        cat_name = img_to_cat.get(img_id, "all")
        if cat_name not in per_cat_count:
            per_cat_count[cat_name] = 0
        if per_cat_count[cat_name] >= args.images_per_cat:
            continue

        try:
            pil_img = dataset._extract_image(sample)
            raw_bytes = _encode_jpeg(pil_img, args.jpeg_quality)
        except Exception:
            continue

        # Optionally save clean image
        if args.save_clean:
            clean_dir = output_root / "clean" / cat_name
            clean_dir.mkdir(parents=True, exist_ok=True)
            clean_path = clean_dir / f"{img_id}_clean.jpg"
            clean_path.write_bytes(raw_bytes)
            decoded = _decode_image(raw_bytes)
            if decoded is not None:
                decoded.save(clean_dir / f"{img_id}_clean_preview.png")

        for ctype, level_map in pipelines.items():
            for level, pipeline in level_map.items():
                if not pipeline.is_enabled():
                    continue
                corrupted_variants = pipeline.apply(raw_bytes)
                for corrupted_bytes, marker in corrupted_variants:
                    out_dir = output_root / ctype / level / cat_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{img_id}_{marker}.jpg"
                    out_path.write_bytes(corrupted_bytes)

                    preview = _decode_image(corrupted_bytes)
                    preview_ok = preview is not None
                    if preview_ok:
                        preview.save(out_dir / f"{img_id}_{marker}_preview.png")

                    decode_stats.setdefault(ctype, {}).setdefault(level, {"ok": 0, "total": 0})
                    decode_stats[ctype][level]["total"] += 1
                    if preview_ok:
                        decode_stats[ctype][level]["ok"] += 1

                    manifest.append(
                        {
                            "image_id": img_id,
                            "category": cat_name,
                            "corruption": marker,
                            "saved_to": str(out_path),
                            "decode_ok": preview_ok,
                        }
                    )

        per_cat_count[cat_name] += 1
        total_processed += 1

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] Saved {len(manifest)} corrupted samples to {output_root}")
    print(f"Manifest written to: {manifest_path}")
    if decode_stats:
        print("\nDecode success rates:")
        for ctype in sorted(decode_stats.keys()):
            for level in sorted(decode_stats[ctype].keys()):
                stats = decode_stats[ctype][level]
                ok, total = stats["ok"], stats["total"]
                rate = ok / total if total else 0.0
                print(f"  {ctype.upper()} {level}: {ok}/{total} ({rate:.2%})")


if __name__ == "__main__":
    main()
