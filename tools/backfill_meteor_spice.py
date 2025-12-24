import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "PureT") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "PureT"))

from PureT.coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from PureT.coco_caption.pycocoevalcap.meteor.meteor import Meteor
from PureT.coco_caption.pycocoevalcap.spice.spice import Spice

def build_reference_map(ann_path: Path) -> Dict[int, List[str]]:
    with ann_path.open("r", encoding="utf-8") as f:
        ann_data = json.load(f)
    ref_map: Dict[int, List[str]] = {}
    for ann in ann_data.get("annotations", []):
        image_id = ann.get("image_id")
        caption = ann.get("caption")
        if image_id is None or caption is None:
            continue
        try:
            image_id = int(image_id)
        except Exception:
            pass
        ref_map.setdefault(image_id, []).append(caption)
    return ref_map


def build_gts(img_ids: List[int], ref_map: Dict[int, List[str]]):
    gts = {}
    for image_id in img_ids:
        refs = ref_map.get(image_id, [])
        if not refs:
            refs = ["."]
        gts[image_id] = [{"caption": c} for c in refs]
    return gts


def build_res(results: List[Dict]):
    res = {}
    for item in results:
        image_id = item.get("image_id")
        if image_id is None:
            continue
        try:
            image_id = int(image_id)
        except Exception:
            pass
        res[image_id] = [{"caption": item.get("caption", "")}]
    return res


def find_result_file(model_folder: Path, corrupt_type: str, corrupt_level: str) -> Path:
    result_dir = model_folder / "result"
    direct = result_dir / f"result_{corrupt_type}_{corrupt_level}.json"
    if direct.exists():
        return direct
    pattern = f"result_{corrupt_type}*{corrupt_level}.json"
    matches = sorted(result_dir.glob(pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Result file not found for {corrupt_type} {corrupt_level} in {result_dir}")


def main():
    parser = argparse.ArgumentParser(description="Backfill METEOR/SPICE in batch report JSONs")
    parser.add_argument("--reports-dir", default="batch_reports/openrouter_claude", help="Directory of batch report JSONs")
    parser.add_argument("--model-folder", default=None, help="Optional override for model folder")
    parser.add_argument(
        "--annfile",
        default="PureT/data/coco_karpathy/captions_test.json",
        help="Reference annotations JSON (COCO-style)",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports dir not found: {reports_dir}")

    ann_path = Path(args.annfile)
    if not ann_path.is_absolute():
        ann_path = (PROJECT_ROOT / ann_path).resolve()
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    tok = PTBTokenizer()
    meteor = Meteor()
    spice = Spice()

    ref_cache: Dict[Path, Dict[int, List[str]]] = {}
    gts_cache: Dict[Tuple[Path, Tuple[int, ...]], Dict] = {}

    report_paths = sorted(reports_dir.glob("*.json"))
    if not report_paths:
        print(f"No report JSONs found in {reports_dir}")
        return

    for report_path in report_paths:
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        model_folder = Path(args.model_folder) if args.model_folder else Path(report.get("model_folder", ""))
        if not model_folder:
            raise ValueError(f"Missing model_folder for {report_path}")

        ref_map = ref_cache.get(ann_path)
        if ref_map is None:
            ref_map = build_reference_map(ann_path)
            ref_cache[ann_path] = ref_map

        corrupt_type = report.get("corrupt_type")
        corrupt_level = report.get("corrupt_level")
        if not corrupt_type or not corrupt_level:
            raise ValueError(f"Missing corrupt_type/level in {report_path}")

        result_path = find_result_file(model_folder, corrupt_type, corrupt_level)
        with result_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        if not isinstance(results, list):
            raise ValueError(f"Unexpected result format in {result_path}")

        img_ids = sorted({int(item["image_id"]) for item in results if "image_id" in item})
        gts_key = (ann_path, tuple(img_ids))
        gts_tok = gts_cache.get(gts_key)
        if gts_tok is None:
            gts = build_gts(img_ids, ref_map)
            gts_tok = tok.tokenize(gts)
            gts_cache[gts_key] = gts_tok

        res = build_res(results)
        res_tok = tok.tokenize(res)

        meteor_score, _ = meteor.compute_score(gts_tok, res_tok)
        spice_score, _ = spice.compute_score(gts_tok, res_tok)

        metrics = report.get("metrics") or {}
        metrics["METEOR"] = float(meteor_score)
        metrics["SPICE"] = float(spice_score)
        report["metrics"] = metrics

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Updated {report_path.name}: METEOR={meteor_score:.4f}, SPICE={spice_score:.4f}")


if __name__ == "__main__":
    main()
