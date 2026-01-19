"""COCO Dataset (HuggingFace 2014) 兼容实现。

重构后职责：
1. 加载图像并标准化为224x224的JPEG字节流
2. 根据配置应用损坏
3. 根据模型类型返回：
   - ByteCaption模型：损坏后的字节流（bytes）
   - 其他视觉模型：损坏后解码的PIL图像
"""

from __future__ import annotations

# ====================
# Standard Library
# ====================
import os
import io
import random
import json
import pickle
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional, Union

# ====================
# Third-party Libraries
# ====================
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from datasets import load_from_disk
from torchvision import transforms

# ====================
# Project Local Imports
# ====================
from lib.config import cfg
import lib.utils as utils
from .feature_extractor import get_feature_extractor  # 保留，后续可能需要
from corenet.data.transforms.jpeg_corruption import JPEGCorruptionPipeline


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
    """基础图像 -> Tensor 变换。

    说明：目前固定 Resize(224,224) + ToTensor；如需与主干网络保持一致，可在此扩展。
    放在函数而非全局常量，便于未来根据 cfg 修改。
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img)

def pil_to_jpeg_bytes(img: Image.Image, quality: int = 60) -> bytes:
    """将PIL图像转换为JPEG字节流。
    
    Args:
        img: PIL.Image 对象（已转为RGB）
        quality: JPEG压缩质量（默认60）
    
    Returns:
        JPEG字节流
    """
    import io
    # 确保尺寸为224x224
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.BICUBIC)
    
    byte_buffer = io.BytesIO()
    img.save(byte_buffer, format='JPEG', quality=quality)
    byte_buffer.seek(0)
    return byte_buffer.getvalue()

class CocoDataset(data.Dataset):
    """COCO数据集 - 智能处理损坏和返回格式。
    
    职责：
    1. 从HuggingFace数据集加载图像
    2. 标准化为224x224的JPEG字节流
    3. 根据配置应用损坏
    4. 根据模型类型返回合适的格式：
       - ByteCaption: 返回损坏后的字节流
       - 其他模型: 返回损坏后解码的PIL图像
    """
    def __init__(
        self,
        image_ids_path,
        input_seq,
        target_seq,
        gv_feat_path,
        seq_per_img,
        max_feat_num,
        max_samples=None,
        return_captions: bool = False,
        jpeg_quality: int = 60,
        # 损坏相关参数
        corruption_types: Optional[List[str]] = None,
        corruption_level: str = "S0",
        corruption_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        # 模型类型参数
        model_type: str = "bytecaption",  # "bytecaption" 或 "visual"
        is_training: bool = True,  # 训练模式不损坏
        # 评估图像保存
        save_eval_images_dir: Optional[str] = None,
        save_eval_images_max: int = 0,
    ):
        # 基础配置保存
        self.max_feat_num: int = max_feat_num
        self.seq_per_img: int = seq_per_img
        self.max_samples: Optional[int] = max_samples 
        self.return_captions = bool(return_captions)
        self.jpeg_quality = int(jpeg_quality)
        
        # 损坏配置
        self.corruption_types = corruption_types or []
        self.corruption_level = corruption_level
        self.corruption_overrides = corruption_overrides or {}
        self.model_type = model_type.lower()
        self.is_training = is_training

        # 评估图像保存配置
        self.save_eval_images_dir = save_eval_images_dir
        self.save_eval_images_max = max(0, int(save_eval_images_max))
        self._saved_images = 0
        if self.save_eval_images_dir:
            os.makedirs(self.save_eval_images_dir, exist_ok=True)
            print(f"[CocoDataset] Image save dir created: {self.save_eval_images_dir}")
            print(f"[CocoDataset] Max images to save: {self.save_eval_images_max}")
        
        # 创建损坏管线
        pipeline_seed = int(getattr(cfg, "SEED", 0)) if getattr(cfg, "SEED", 0) > 0 else None
        self.corruption_pipeline = JPEGCorruptionPipeline(
            corruption_types=self.corruption_types,
            level=self.corruption_level,
            overrides=self.corruption_overrides,
            seed=pipeline_seed,
        ) if self.corruption_types else None

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
        self._hf_builder = './PureT/data/coco_karpathy/AbdoTW___coco_2014_karpathy'  # Path to the dataset on disk
        # 只在主进程 / 初始化时加载一次；worker 进程通过 __setstate__ 重新加载
        self.ds = load_from_disk(f"{self._hf_builder}/{self.hf_split}")

        # Build image_ids list for compatibility
        ids_from_json = None
        print(f"[DEBUG CocoDataset] image_ids_path={image_ids_path}")
        print(f"[DEBUG CocoDataset] Path exists: {os.path.exists(image_ids_path) if image_ids_path else 'Path is None'}")
        if image_ids_path and os.path.exists(image_ids_path):
            with open(image_ids_path, 'r', encoding='utf-8') as f:
                txt = f.read().strip()
                if txt.startswith('{') and len(txt) > 2:
                    obj = json.loads(txt)
                    if isinstance(obj, dict) and len(obj) > 0:
                        ids_from_json = list(obj.keys())
                        print(f"[DEBUG CocoDataset] Successfully loaded {len(ids_from_json)} IDs from JSON")

        if ids_from_json is None:
            # 使用顺序 ID：转换为整数以便与评估器匹配
            self.image_ids = [i for i in range(len(self.ds))]
            print(f"Using sequential image IDs: 0 to {len(self.ds)-1}")
        else:
            max_n = min(len(ids_from_json), len(self.ds))
            # 尝试将 IDs 转换为整数，以便与评估器中的 id_to_captions 键匹配
            converted_ids = []
            for id_str in ids_from_json[:max_n]:
                try:
                    converted_ids.append(int(id_str))
                except (ValueError, TypeError):
                    # 如果转换失败，保持原始形式
                    converted_ids.append(id_str)
            self.image_ids = converted_ids
            print(f"Loaded {len(self.image_ids)} image IDs from JSON file")
            if len(self.image_ids) > 0:
                print(f"First 5 image IDs: {self.image_ids[:5]}")

        # Optional sequence pkls; if unavailable, auto-build sequences from HF captions
        self.auto_seq: bool = False
        if self.return_captions:
            # HF caption training path does not need seq/vocab construction
            self.input_seq = None
            self.target_seq = None
            self.seq_len = int(getattr(cfg.MODEL, 'SEQ_LEN', 17))
            self.auto_seq = False
            self.is_validation = False
        else:
            use_pkls = False
            if isinstance(input_seq, str) and isinstance(target_seq, str):
                if os.path.exists(input_seq) and os.path.exists(target_seq):
                    use_pkls = True

            if use_pkls:
                self.input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')
                self.target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')
                first_key = None
                if len(self.image_ids) > 0 and self.image_ids[0] in self.input_seq:
                    first_key = self.image_ids[0]
                elif len(self.input_seq) > 0:
                    first_key = next(iter(self.input_seq.keys()))
                self.seq_len = int(getattr(cfg.MODEL, 'SEQ_LEN', 17)) if first_key is None else int(self.input_seq[first_key].shape[1])
            else:
                self.is_validation = (input_seq is None and target_seq is None)
                if not self.is_validation:
                    self.auto_seq = True
                    self.input_seq = None
                    self.target_seq = None
                    self.seq_len = int(getattr(cfg.MODEL, 'SEQ_LEN', 17))
                    self.vocab_path = "./PureT/data/coco/coco_vocabulary.txt"
                    if not os.path.exists(self.vocab_path):
                        print(f"Building vocabulary file at {self.vocab_path}")
                        self._build_vocab_file(self.vocab_path, vocab_size=int(getattr(cfg.MODEL, 'VOCAB_SIZE', 9487)))
                    self.vocab = utils.load_vocab(self.vocab_path)
                    self.w2i = {w: i for i, w in enumerate(self.vocab)}
                    print(f"Loaded vocabulary with {len(self.vocab)} words")
                else:
                    self.auto_seq = False
                    self.input_seq = None
                    self.target_seq = None
                    self.vocab_path = cfg.INFERENCE.VOCAB
                    if not os.path.exists(self.vocab_path):
                        print(f"Building vocabulary file for validation at {self.vocab_path}")
                        self._build_vocab_file(self.vocab_path, vocab_size=int(getattr(cfg.MODEL, 'VOCAB_SIZE', 9487)))
                    self.vocab = utils.load_vocab(self.vocab_path)
                    self.w2i = {w: i for i, w in enumerate(self.vocab)}
                    print(f"Loaded vocabulary for validation with {len(self.vocab)} words")

    def set_seq_per_img(self, seq_per_img: int) -> None:
        """动态调整每图序列数量（兼容旧接口）。"""
        self.seq_per_img = seq_per_img

    def __len__(self) -> int:
        """数据集长度（受 image_ids / ds / max_samples 共同限制）。"""
        base_length = min(len(self.image_ids), len(self.ds))
        return min(base_length, self.max_samples) if self.max_samples is not None else base_length

    def _maybe_save_corrupted(self, image_id, corrupted_list: List[Tuple[bytes, str]]):
        """Optionally save corrupted JPEG bytes used for evaluation (bounded by max)."""
        if not self.save_eval_images_dir or self.save_eval_images_max <= 0:
            return
        if self._saved_images >= self.save_eval_images_max:
            return
        for corrupted_bytes, marker in corrupted_list:
            if self._saved_images >= self.save_eval_images_max:
                break
            fname = f"{image_id}_{marker}.jpg"
            out_path = os.path.join(self.save_eval_images_dir, fname)
            try:
                with open(out_path, "wb") as f:
                    f.write(corrupted_bytes)
                self._saved_images += 1
                if self._saved_images <= 3 or self._saved_images == self.save_eval_images_max:
                    print(f"[SAVED] Image {self._saved_images}/{self.save_eval_images_max}: {out_path}")
            except Exception as e:
                print(f"[WARN] Failed to save eval image {out_path}: {e}")

    def __getitem__(self, index: int) -> Union[Tuple, ...]:
        """获取数据项，智能处理损坏和返回格式。"""
        indices = np.array([index]).astype('int')
        image_id = self.image_ids[index] if index < len(self.image_ids) else str(index)
        
        # 加载图像
        sample = self.ds[index]
        img = self._extract_image(sample)
        gv_feat = np.zeros((1,), dtype=np.float32)
        
        # 1. 转换为JPEG字节流
        jpeg_bytes = pil_to_jpeg_bytes(img, quality=self.jpeg_quality)
        
        # 2. 应用损坏（如果需要）
        if not self.is_training and self.corruption_pipeline and self.corruption_pipeline.is_enabled():
            # 评估模式且配置了损坏
            corrupted_variants = self.corruption_pipeline.apply(jpeg_bytes, image_id=image_id)
            corrupted_list = corrupted_variants
        else:
            # 训练模式或无损坏配置
            corrupted_list = [(jpeg_bytes, "none")]

        # 可选保存用于评估的损坏图像
        self._maybe_save_corrupted(image_id, corrupted_list)
        
        # 3. 根据模型类型决定返回格式
        processed_results = []
        for corrupted_bytes, marker in corrupted_list:
            if self.model_type == "bytecaption":
                # ByteCaption: 返回字节流
                processed_results.append(corrupted_bytes)
            else:
                # 其他视觉模型: 解码为PIL图像
                try:
                    decoded_img = Image.open(io.BytesIO(corrupted_bytes)).convert("RGB")
                    processed_results.append(decoded_img)
                except Exception as e:
                    processed_results.append(None)
        
        # 4. 单个结果还是多个结果
        if len(processed_results) == 1:
            att_feats = processed_results[0]
        else:
            # 多个损坏版本，返回列表
            att_feats = processed_results

        if self.return_captions:
            captions = self._extract_captions_from_sample(sample)
            return indices, captions, gv_feat, att_feats

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
        """从 HF sample 中解析出 PIL.Image (确保 RGB)。"""
        img = sample.get('image', None)
        if img is None:
            raise KeyError('COCO sample missing `image` field')
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
            self.ds = load_from_disk(f"{self._hf_builder}/{self.hf_split}")

    def _basic_tokenize(self, text: str) -> List[str]:
        """基础分词：小写 + 正则提取。"""
        import re
        return re.findall(r"[a-z0-9']+", str(text).lower())

    def _tokenize(self, text: str) -> List[int]:
        """分词并映射到词表索引（忽略 OOV）。"""
        tokens = self._basic_tokenize(text)
        return [self.w2i[t] for t in tokens if t in self.w2i]

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
        """统一的 caption 提取逻辑，支持不同字段名与兜底。"""
        caps = sample.get("caption", [])
        if isinstance(caps, str):
            caps = [caps]
        elif not isinstance(caps, list):
            caps = []
        if not caps:
            for alt_key in ["captions", "text"]:
                if alt_key in sample:
                    alt_caps = sample[alt_key]
                    if isinstance(alt_caps, str):
                        caps = [alt_caps]
                    elif isinstance(alt_caps, list):
                        caps = alt_caps
                    break
        if not caps:
            caps = ['.']  # 兜底保证至少一个 token
        return caps

    def _build_seqs_from_captions(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        caps = self._extract_captions_from_sample(sample)
        # 顺序使用，不随机；不足则重复补齐
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
        """从当前 split captions 构建基于频率的词表文件。"""
        counter: Counter = Counter()
        dataset_length = len(self) if (hasattr(self, 'max_samples') and self.max_samples) else len(self.ds)
        for i in range(dataset_length):
            s = self.ds[i]
            for cap in self._extract_captions_from_sample(s):
                for tok in self._basic_tokenize(cap):
                    counter[tok] += 1
        most_common = [w for w, _ in counter.most_common(vocab_size)]
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for w in most_common:
                # Each line corresponds to vocab index i (starting from 1 because 0 is EOS '.')
                f.write(f"{w}\n")
        print(f"Built vocabulary file with {len(most_common)} words at {path}")
