#!/usr/bin/env python3
"""Calculate decoding success rates under JPEG bitstream corruptions.

Goal: fast, minimal, and comparable to the eval pipeline.

Speed notes:
- Prefer raw JPEG bytes from the HF dataset when available (no re-encode).
- Parallelize across CPU processes; each worker loads the HF split once.

Example:
  python tools/calc_decoding_rates.py 
    --config PureT/experiments/ByteCaption_XE/config_coco.yml 
    --corrupt-types rbbf rbsl 
    --severity-levels S1 S2 S3 S4 S5 
    --max-images 0
    --workers 8
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PURET_ROOT = REPO_ROOT / "PureT"
if PURET_ROOT.exists() and str(PURET_ROOT) not in sys.path:
    sys.path.insert(0, str(PURET_ROOT))

from corenet.data.transforms.jpeg_corruption import JPEGCorruptionPipeline, normalize_level  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from lib.config import cfg, cfg_from_file  # noqa: E402


@dataclass(frozen=True)
class JobConfig:
    hf_builder: str
    split: str
    num_items: int
    corrupt_types: Tuple[str, ...]
    levels: Tuple[str, ...]
    jpeg_quality: int
    allow_truncated: bool
    seed: int


def _detect_split(image_ids_path: Optional[str]) -> str:
    """Best-effort split detection consistent with CocoDataset."""
    if not image_ids_path:
        return "train"
    base = os.path.basename(str(image_ids_path)).lower()
    if "val" in base or "valid" in base:
        return "validation"
    if "test" in base:
        return "test"
    return "train"


def _decode_check(data: bytes) -> bool:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return True
    except Exception:
        return False


def _try_get_jpeg_bytes(sample: Dict, jpeg_quality: int) -> Optional[bytes]:
    """Get JPEG bytes from HF sample with minimal overhead.

Priority:
1) dict with 'bytes'
2) dict with 'path'
3) PIL.Image -> re-encode (slower)
"""
    img = sample.get("image", None)
    if img is None:
        return None

    # HF datasets.Image sometimes yields a dict payload.
    if isinstance(img, dict):
        b = img.get("bytes")
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)
        p = img.get("path")
        if isinstance(p, str) and p and os.path.exists(p):
            try:
                return Path(p).read_bytes()
            except Exception:
                return None
        return None

    # PIL.Image.Image
    if isinstance(img, Image.Image):
        try:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            return buf.getvalue()
        except Exception:
            return None

    return None


def _iter_chunks(n: int, chunks: int) -> List[Tuple[int, int]]:
    if chunks <= 1:
        return [(0, n)]
    chunks = min(chunks, n)
    base = n // chunks
    rem = n % chunks
    out = []
    start = 0
    for i in range(chunks):
        size = base + (1 if i < rem else 0)
        end = start + size
        out.append((start, end))
        start = end
    return out


_W_JOB: Optional[JobConfig] = None
_W_DS = None
_W_PIPELINES: Dict[Tuple[str, str], JPEGCorruptionPipeline] = {}


def _worker_init(job: JobConfig) -> None:
    """Initializer for each worker process.

Loads HF split once and builds corruption pipelines once.
"""
    global _W_JOB, _W_DS, _W_PIPELINES
    _W_JOB = job
    ImageFile.LOAD_TRUNCATED_IMAGES = bool(job.allow_truncated)

    # Seed per-process RNG to keep runs reasonably stable.
    # (Per-chunk reseeding happens inside _worker_process_range.)
    np.random.seed(int(job.seed) + (os.getpid() % 100000))

    _W_DS = load_from_disk(f"{job.hf_builder}/{job.split}")
    _W_PIPELINES = {}
    for ctype in job.corrupt_types:
        for lvl in job.levels:
            _W_PIPELINES[(ctype, lvl)] = JPEGCorruptionPipeline([ctype], level=lvl)


def _worker_process_range(start: int, end: int) -> Dict[str, Tuple[int, int]]:
    """Process a slice of indices [start, end) using worker globals."""
    if _W_JOB is None or _W_DS is None:
        raise RuntimeError("Worker not initialized")

    # Make corruption randomness reproducible per-range (independent of scheduling).
    np.random.seed(int(_W_JOB.seed) + int(start))

    stats: Dict[str, Tuple[int, int]] = {}
    for ctype in _W_JOB.corrupt_types:
        for lvl in _W_JOB.levels:
            stats[f"{ctype}:{lvl}"] = (0, 0)

    for idx in range(start, end):
        sample = _W_DS[idx]
        raw_bytes = _try_get_jpeg_bytes(sample, jpeg_quality=_W_JOB.jpeg_quality)
        if not raw_bytes:
            continue

        for ctype in _W_JOB.corrupt_types:
            for lvl in _W_JOB.levels:
                pipeline = _W_PIPELINES[(ctype, lvl)]
                if not pipeline.is_enabled():
                    continue
                corrupted_variants = pipeline.apply(raw_bytes)
                for corrupted_bytes, _marker in corrupted_variants:
                    ok = _decode_check(corrupted_bytes)
                    key = f"{ctype}:{lvl}"
                    prev_ok, prev_total = stats[key]
                    stats[key] = (prev_ok + (1 if ok else 0), prev_total + 1)

    return stats


def _parse_image_ids_len(image_ids_path: Optional[str], ds_len: int) -> int:
    """Mirror CocoDataset image-id list truncation logic.

If the JSON is a dict, only its keys are used; dataset length is capped by ds_len.
"""
    if not image_ids_path:
        return ds_len
    p = Path(image_ids_path)
    if not p.exists():
        return ds_len
    try:
        txt = p.read_text(encoding="utf-8").strip()
    except Exception:
        return ds_len
    if not txt.startswith("{"):
        return ds_len
    try:
        obj = json.loads(txt)
    except Exception:
        return ds_len
    if not isinstance(obj, dict) or not obj:
        return ds_len
    return min(len(obj.keys()), ds_len)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate JPEG decoding success rates.")
    parser.add_argument("--config", type=str, default="PureT/experiments/ByteCaption_XE/config_coco.yml")
    parser.add_argument("--corrupt-types", type=str, nargs="+", default=["rbbf", "rbsl", "metadata_loss"])
    parser.add_argument("--severity-levels", type=str, nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    parser.add_argument("--max-images", type=int, default=0, help="0 for all")
    parser.add_argument("--jpeg-quality", type=int, default=60, help="Only used if raw bytes unavailable")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=0, help="0 = use os.cpu_count()")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Images per task chunk (smaller = more progress updates).",
    )
    parser.add_argument(
        "--no-allow-truncated",
        action="store_true",
        help="Disable Pillow LOAD_TRUNCATED_IMAGES (stricter decode)",
    )
    args = parser.parse_args()

    cfg_from_file(args.config)

    # Determine HF split and dataset length once in main
    image_ids_path = getattr(cfg.DATA_LOADER, "TEST_ID", None)
    split = _detect_split(image_ids_path)
    hf_builder = "./PureT/data/coco_karpathy/AbdoTW___coco_2014_karpathy"
    ds = load_from_disk(f"{hf_builder}/{split}")
    n = _parse_image_ids_len(image_ids_path, len(ds))
    if args.max_images and args.max_images > 0:
        n = min(n, int(args.max_images))

    corrupt_types = tuple(c.lower() for c in args.corrupt_types)
    levels = tuple(normalize_level(l) for l in args.severity_levels)
    workers = int(args.workers) if int(args.workers) > 0 else (os.cpu_count() or 1)
    workers = max(1, workers)

    job = JobConfig(
        hf_builder=hf_builder,
        split=split,
        num_items=n,
        corrupt_types=corrupt_types,
        levels=levels,
        jpeg_quality=int(args.jpeg_quality),
        allow_truncated=not bool(args.no_allow_truncated),
        seed=int(args.seed),
    )

    print(f"Config: {args.config}")
    print(f"HF split: {split}")
    print(f"Images: {n}")
    print(f"Corruptions: {list(corrupt_types)}")
    print(f"Levels: {list(levels)}")
    print(f"Workers: {workers}")
    print(f"Pillow allow truncated: {job.allow_truncated}")

    chunk_size = max(1, int(args.chunk_size))
    chunks: List[Tuple[int, int]] = []
    for s in range(0, n, chunk_size):
        chunks.append((s, min(n, s + chunk_size)))

    # Aggregate stats: key -> ok,total
    agg_ok: DefaultDict[str, int] = defaultdict(int)
    agg_total: DefaultDict[str, int] = defaultdict(int)

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(job,)) as ex:
        futures = [ex.submit(_worker_process_range, s, e) for (s, e) in chunks]
        pbar = tqdm(total=n, desc="images")
        try:
            for fut in as_completed(futures):
                part = fut.result()
                for key, (ok, total) in part.items():
                    agg_ok[key] += int(ok)
                    agg_total[key] += int(total)
                # Update by any one key's total (all keys share same count increments)
                any_total = 0
                if part:
                    any_total = next(iter(part.values()))[1]
                pbar.update(any_total)
        finally:
            pbar.close()

    # Print report
    print("\n" + "=" * 72)
    print(f"{'Corruption':<15} {'Level':<10} {'Success Rate':<14} {'OK/Total':<12}")
    print("-" * 72)
    for ctype in sorted(corrupt_types):
        for lvl in levels:
            key = f"{ctype}:{lvl}"
            ok = agg_ok.get(key, 0)
            total = agg_total.get(key, 0)
            rate = (ok / total * 100.0) if total else 0.0
            print(f"{ctype:<15} {lvl:<10} {rate:>8.2f}% {str(ok) + '/' + str(total):>12}")
    print("=" * 72)


if __name__ == "__main__":
    main()
