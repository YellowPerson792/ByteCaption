"""Flickr8k Dataset (HuggingFace) 兼容封装。
"""

from __future__ import annotations

import os
import random
import json
import pickle
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from datasets import load_dataset
from torchvision import transforms

from PureT.lib.config import cfg
import PureT.lib.utils as utils
from .feature_extractor import get_feature_extractor  # noqa: F401 (预留未来使用)


# timm interp compatibility
try:
    from timm.data.transforms import _pil_interp
except ImportError:
    def _pil_interp(method):
        if method == 'bicubic':
            return Image.BICUBIC
        if method == 'bilinear':
            return Image.BILINEAR
        if method == 'nearest':
            return Image.NEAREST
        return Image.BICUBIC

def pil_to_tensor_transform(img: Image.Image) -> torch.Tensor:
    """基础 PIL -> Tensor 转换。
    当前固定 Resize(224,224)+ToTensor，可按需扩展加入归一化或数据增强。"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img)

class Flickr8kDataset(data.Dataset):
    """Flickr8k HF 数据集封装。"""

    def __init__(
        self,
        image_ids_path,
        input_seq,
        target_seq,
        gv_feat_path,
        seq_per_img,
        max_feat_num,
        max_samples=None,  # Add parameter to limit dataset size
    ):
        self.max_feat_num: int = max_feat_num
        self.seq_per_img: int = seq_per_img
        self.max_samples: Optional[int] = max_samples 

        # Optional global feature dict
        self.gv_feat = (
            pickle.load(open(gv_feat_path, 'rb'), encoding='bytes')
            if (isinstance(gv_feat_path, str) and len(gv_feat_path) > 0)
            else None
        ) 

        # Determine HF split from the image_ids_path name (train/val/test); default to train
        if image_ids_path and os.path.exists(image_ids_path):
            basename = os.path.basename(str(image_ids_path)).lower()
            if 'val' in basename or 'valid' in basename:
                split = 'validation'
            elif 'test' in basename:
                split = 'test'
            else:
                split = 'train'
        else:
            # Default to train when no image_ids_path is provided
            split = 'train'
        self.hf_split = split

        # Load and store only a lightweight handle; avoid pickling-heavy state
        self._hf_builder = 'jxie/flickr8k'
        self.ds = load_dataset(self._hf_builder, split=split)

        # Build image_ids list for compatibility
        # - If a JSON mapping is provided, use its keys order
        # - Else, fall back to 0..len(ds)-1 as ids
        ids_from_json = None
        # support both dict-json and plain-lines file
        if image_ids_path and os.path.exists(image_ids_path):
            with open(image_ids_path, 'r', encoding='utf-8') as f:
                txt = f.read().strip()
                if txt.startswith('{') and len(txt) > 2:  # Check for non-empty JSON
                    obj = json.loads(txt)
                    if isinstance(obj, dict) and len(obj) > 0:
                        ids_from_json = list(obj.keys())

        if ids_from_json is None:
            # Default to index-based ids as strings for compatibility with pkl dicts
            self.image_ids = [str(i) for i in range(len(self.ds))]
            print(f"Using sequential image IDs: 0 to {len(self.ds)-1}")
        else:
            # Align length with HF dataset to avoid index overflow
            max_n = min(len(ids_from_json), len(self.ds))
            self.image_ids = ids_from_json[:max_n]
            print(f"Loaded {len(self.image_ids)} image IDs from JSON file")
            # If provided ids exceed dataset size, they will be truncated silently

        # Optional sequence pkls; if unavailable, auto-build sequences from HF captions
        self.auto_seq: bool = False
        use_pkls = False
        if isinstance(input_seq, str) and isinstance(target_seq, str):
            if os.path.exists(input_seq) and os.path.exists(target_seq):
                use_pkls = True

        if use_pkls:
            self.input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')
            self.target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')
            # Determine sequence length from stored arrays
            first_key = None
            if len(self.image_ids) > 0 and self.image_ids[0] in self.input_seq:
                first_key = self.image_ids[0]
            elif len(self.input_seq) > 0:
                first_key = next(iter(self.input_seq.keys()))
            if first_key is None:
                # fallback to config value
                self.seq_len = int(getattr(cfg.MODEL, 'SEQ_LEN', 17))
            else:
                self.seq_len = int(self.input_seq[first_key].shape[1])
        else:
            # Check if we're in validation mode (input_seq and target_seq are None)
            self.is_validation = (input_seq is None and target_seq is None)
            
            if not self.is_validation:
                # Build sequences on the fly from HF captions using the configured vocabulary
                self.auto_seq = True
                self.input_seq = None  # 动态生成
                self.target_seq = None
                self.seq_len = int(getattr(cfg.MODEL, 'SEQ_LEN', 17))
                self.vocab_path = cfg.INFERENCE.VOCAB
                # If no vocab file exists, build one from Flickr8k captions (top-K by freq)
                if not os.path.exists(self.vocab_path):
                    print(f"Building vocabulary file at {self.vocab_path}")
                    self._build_vocab_file(self.vocab_path, vocab_size=int(getattr(cfg.MODEL, 'VOCAB_SIZE', 9487)))
                self.vocab = utils.load_vocab(self.vocab_path)
                # word -> index mapping (EOS reserved at index 0 as '.')
                self.w2i = {w: i for i, w in enumerate(self.vocab)}
                print(f"Loaded vocabulary with {len(self.vocab)} words")
            else:
                # Validation mode: still need vocab for evaluation, but no sequences needed
                self.auto_seq = False
                self.input_seq = None
                self.target_seq = None
                self.vocab_path = cfg.INFERENCE.VOCAB
                # Ensure vocab file exists even in validation mode
                if not os.path.exists(self.vocab_path):
                    print(f"Building vocabulary file for validation at {self.vocab_path}")
                    self._build_vocab_file(self.vocab_path, vocab_size=int(getattr(cfg.MODEL, 'VOCAB_SIZE', 9487)))
                # Load vocab for evaluation purposes
                self.vocab = utils.load_vocab(self.vocab_path)
                self.w2i = {w: i for i, w in enumerate(self.vocab)}
                print(f"Loaded vocabulary for validation with {len(self.vocab)} words")

    def set_seq_per_img(self, seq_per_img: int) -> None:
        self.seq_per_img = seq_per_img

    def __len__(self) -> int:
        base_length = min(len(self.image_ids), len(self.ds))
        return min(base_length, self.max_samples) if self.max_samples is not None else base_length

    def __getitem__(self, index: int):
        # index within HF split
        indices = np.array([index]).astype('int')

        # Select a corresponding id for gv/seq lookup when available
        image_id = self.image_ids[index] if index < len(self.image_ids) else str(index)

        # Load image from HF first
        sample = self.ds[index]
        img = self._extract_image(sample)
        
        gv_feat = np.zeros((1,), dtype=np.float32)
        att_feats = pil_to_tensor_transform(img)  # [1, 224, 224]
        # gv_feat is a placeholder, att_feats is preprocessed image for Swin backbone

        if self.max_feat_num > 0 and hasattr(att_feats, 'shape') and len(att_feats.shape) > 0:
            # For image tensors this generally does nothing; kept for API parity
            pass

        # Check if we're in validation mode
        if hasattr(self, 'is_validation') and self.is_validation:
            # print("[DEBUG] Validation mode - returning indices, gv_feat, att_feats only")
            # print("[DEBUG] indices:", indices)
            # print("[DEBUG] gv_feat shape:", gv_feat.shape if gv_feat is not None else "N/A")
            # print("[DEBUG] att_feats shape:", att_feats.shape if att_feats is not None else "N/A")
            return indices, gv_feat, att_feats

        # If auto_seq is enabled, build sequences from HF captions on the fly
        if self.auto_seq:
            input_seq, target_seq = self._build_seqs_from_captions(sample)
            # print("[DEBUG] indices:", indices)
            # print("[DEBUG] input_seq shape:", input_seq.shape)
            # print("[DEBUG] target_seq shape:", target_seq.shape)
            # print("[DEBUG] input_seq sample:", input_seq[0] if input_seq.shape[0] > 0 else "N/A")
            # print("[DEBUG] target_seq sample:", target_seq[0] if target_seq.shape[0] > 0 else "N/A")
            # print("[DEBUG] gv_feat shape:", gv_feat.shape if gv_feat is not None else "N/A")
            # print("[DEBUG] att_feats shape:", att_feats.shape if att_feats is not None else "N/A")
            return indices, input_seq, target_seq, gv_feat, att_feats

        # Training path with sequences
        if image_id not in self.input_seq:
            # If ids don't match, fall back to an arbitrary key to avoid KeyError
            # This keeps pipeline running but indicates a mismatch in upstream mappings
            key = next(iter(self.input_seq.keys()))
        else:
            key = image_id

        input_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        target_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')

        n = len(self.input_seq[key])
        if n >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_img)
        else:
            sid = n
            ixs = random.sample(range(n), self.seq_per_img - n)
            input_seq[0:n, :] = self.input_seq[key]
            target_seq[0:n, :] = self.target_seq[key]

        for i, ix in enumerate(ixs):
            input_seq[sid + i] = self.input_seq[key][ix, :]
            target_seq[sid + i] = self.target_seq[key][ix, :]

        return indices, input_seq, target_seq, gv_feat, att_feats
    
    # ====================
    # Internal helpers
    # ====================
    def _extract_image(self, sample: Dict[str, Any]) -> Image.Image:
        img = sample.get('image', None)
        if img is None:
            raise KeyError('Flickr8k sample missing `image` field')
        if not isinstance(img, Image.Image):
            # datasets.Image can return dict with 'bytes' or similar; try to convert
            # Fallback: use PIL to open if a path is available
            if isinstance(img, dict) and 'path' in img and os.path.exists(img['path']):
                img = Image.open(img['path']).convert('RGB')
            else:
                # Last resort: try to build PIL from raw bytes
                from io import BytesIO

                if isinstance(img, dict) and 'bytes' in img:
                    img = Image.open(BytesIO(img['bytes'])).convert('RGB')
                else:
                    raise TypeError('Unsupported image payload type')
        else:
            # ensure RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
        return img

    def __getstate__(self):
        # Avoid pickling the HF dataset object into workers; reload on demand
        state = self.__dict__.copy()
        state['ds'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.ds is None:
            # reload HF dataset in worker process
            self.ds = load_dataset(self._hf_builder, split=self.hf_split)

    def _basic_tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"[a-z0-9']+", str(text).lower())

    def _tokenize(self, text: str) -> List[int]:
        return [self.w2i[t] for t in self._basic_tokenize(text) if t in self.w2i]

    def _build_single_seq(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        # Build one pair of (input_seq, target_seq) arrays with BOS=0 at start, EOS=0 at end, ignore_index=-1
        ids = self._tokenize(text)
        max_len = max(0, min(len(ids), self.seq_len - 1))  # Reserve space for BOS
        
        in_arr = np.zeros((self.seq_len,), dtype='int')
        tgt_arr = np.full((self.seq_len,), -1, dtype='int')
        
        # BOS token (0) at position 0 in input_seq
        in_arr[0] = 0
        
        if max_len > 0:
            # Place actual tokens starting from position 1
            in_arr[1:max_len + 1] = ids[:max_len]
            # Target sequence: predict the actual tokens, then EOS
            tgt_arr[:max_len] = ids[:max_len]
            tgt_arr[max_len] = 0  # EOS at the end
        else:
            # no valid tokens: train to output EOS at first step after BOS
            tgt_arr[0] = 0
        return in_arr, tgt_arr

    def _extract_captions_from_sample(self, sample: Dict[str, Any]) -> List[str]:
        caps: List[str] = []
        for j in range(5):
            key = f"caption_{j}"
            if key in sample and sample[key]:
                caps.append(sample[key])
        if not caps:
            for alt in ["captions", "caption", "text"]:
                if alt in sample:
                    alt_caps = sample[alt]
                    if isinstance(alt_caps, str):
                        caps = [alt_caps]
                    elif isinstance(alt_caps, list):
                        caps = alt_caps
                    break
        if not caps:
            caps = ['.']
        return caps

    def _build_seqs_from_captions(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        caps = self._extract_captions_from_sample(sample)
        if len(caps) >= self.seq_per_img:
            chosen = caps[: self.seq_per_img]
        else:
            repeat_times = self.seq_per_img // len(caps)
            remainder = self.seq_per_img % len(caps)
            chosen = caps * repeat_times + caps[:remainder]
        input_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        target_seq = np.full((self.seq_per_img, self.seq_len), -1, dtype='int')
        for i, cap in enumerate(chosen):
            in_arr, tgt_arr = self._build_single_seq(cap)
            input_seq[i] = in_arr
            target_seq[i] = tgt_arr
        return input_seq, target_seq

    def _build_vocab_file(self, path: str, vocab_size: int) -> None:
        counter: Counter = Counter()
        dataset_length = len(self) if (hasattr(self, 'max_samples') and self.max_samples) else len(self.ds)
        for i in range(dataset_length):
            sample = self.ds[i]
            for cap in self._extract_captions_from_sample(sample):
                for tok in self._basic_tokenize(cap):
                    counter[tok] += 1
        most_common = [w for w, _ in counter.most_common(vocab_size)]
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for w in most_common:
                f.write(f"{w}\n")
        print(f"Built vocabulary file with {len(most_common)} words at {path}")
