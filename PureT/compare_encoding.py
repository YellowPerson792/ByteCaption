import os
import sys
import io
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
from tqdm import tqdm

# project root -> 加入导入路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 从项目里导入数据集与转换函数
from PureT.datasets_.coco_dataset_hf import CocoDataset
# 导入 corenet 的编码器（用于比较）
from corenet.data.transforms.image_bytes import _image_to_bytes

OUTDIR = Path("./encoding_compare_samples")
OUTDIR.mkdir(parents=True, exist_ok=True)

def compare(n_samples=20, batch_size=4, quality=95):
    ds = CocoDataset(
        image_ids_path='./PureT/data/coco_karpathy/validation_ids.json',
        input_seq=None, target_seq=None, gv_feat_path='', seq_per_img=1,
        max_feat_num=-1, max_samples=n_samples
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    cnt = 0
    for batch in tqdm(loader, desc="compare batches"):
        _, _, att_feats = batch  # (indices, gv_feat, att_feats)
        for t in att_feats:
            if cnt >= n_samples:
                return
            # ensure tensor on CPU and proper dtype
            tensor = t.clone().cpu()
            try:
                # Method A: torchvision -> PIL -> save (your script)
                pil = to_pil_image(tensor)
                b_io = io.BytesIO()
                pil.save(b_io, format="JPEG", quality=quality)
                bytes_a = b_io.getvalue()
            except Exception as e:
                bytes_a = None
                print(f"[A] encode error idx {cnt}: {e}")

            try:
                # Method B: project's _image_to_bytes implementation
                bs = _image_to_bytes(tensor, format="jpeg", quality=quality)
                bytes_b = bs.getvalue() if bs is not None else None
            except Exception as e:
                bytes_b = None
                print(f"[B] _image_to_bytes error idx {cnt}: {e}")

            # 保存示例对比
            info = {
                "idx": cnt,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "len_a": len(bytes_a) if bytes_a else None,
                "len_b": len(bytes_b) if bytes_b else None
            }
            print(info)

            # 保存 bytes 到文件以便打开查看
            if bytes_a:
                Path(OUTDIR / f"{cnt:04d}_A_len{len(bytes_a)}.jpg").write_bytes(bytes_a)
            if bytes_b:
                Path(OUTDIR / f"{cnt:04d}_B_len{len(bytes_b)}.jpg").write_bytes(bytes_b)

            # 保存 PIL 的 size/format
            if bytes_a or bytes_b:
                try:
                    if bytes_a:
                        im = Image.open(io.BytesIO(bytes_a))
                        print("  A ->", im.format, im.size, im.mode)
                    if bytes_b:
                        im = Image.open(io.BytesIO(bytes_b))
                        print("  B ->", im.format, im.size, im.mode)
                except Exception as e:
                    print("  open saved bytes error:", e)

            cnt += 1

if __name__ == "__main__":
    compare(n_samples=20, batch_size=4, quality=95)