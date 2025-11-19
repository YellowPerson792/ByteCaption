#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import io
from typing import Dict, Union, List

import numpy as np
import torch
from PIL import Image

from corenet.data.transforms import TRANSFORMATIONS_REGISTRY, BaseTransformation


def _image_to_bytes(x: torch.Tensor, **kwargs) -> io.BytesIO:
    """
    将一个范围在 [0, 1] 的图像张量通过 PIL 保存为文件字节。
    （已移除损坏逻辑，仅做转换）
    """
    # 移除 corrupt_level 参数，它现在由专门的类处理
    # kwargs.pop("corrupt_level", "none")

    assert x.min() >= 0
    assert x.max() <= 1
    x = (x * 255).byte().permute(1, 2, 0).cpu().numpy()  # 转换为 H, W, C 顺序的字节

    img = Image.fromarray(x)
    byte_array = io.BytesIO()

    img.save(byte_array, **kwargs)
    byte_array.seek(0)
    return byte_array


def _bytes_to_int32(byte_array: io.BytesIO) -> torch.Tensor:
    """
    Convert a byte array to int32 values.

    Args:
        byte_array: The input byte array.
    Returns:
        The int32 tensor.
    """
    buf = np.frombuffer(byte_array.getvalue(), dtype=np.uint8)
    # The copy operation is required to avoid a warning about non-writable
    # tensors.
    buf = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
    return buf


@TRANSFORMATIONS_REGISTRY.register(name="byte_stream_corrupter", type="image_torch")
class ByteStreamCorrupter(BaseTransformation):
    """
    根据配置对输入的字节流应用一种或多种损坏。
    新逻辑：将一个样本增强为多个损坏的样本（堆叠模式）。
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.corruption_types = getattr(opts, "image_augmentation.byte_stream_corrupter.types", [])
        self.level = getattr(opts, "image_augmentation.byte_stream_corrupter.level", "none")
        
        self.cfg_map = {
            "light": {"bit_flip": 0.001, "drop": 0.60, "head": 256, "tail": 0.6},
            "medium": {"bit_flip": 0.05, "drop": 0.20, "head": 64, "tail": 512},
            "heavy": {"bit_flip": 0.1, "drop": 0.30, "head": 256, "tail": 1024},
        }
        self.params = self.cfg_map.get(self.level, {})

    # -- 将原有的损坏函数变成类的静态方法，方便管理 --
    @staticmethod
    def _random_bit_flip(data: bytes, flip_prob: float) -> bytes:
        ba = bytearray(data)
        total_bits = max(1, len(ba) * 8)
        num_flips = max(1, int(total_bits * flip_prob))
        for _ in range(num_flips):
            bit_index = np.random.randint(0, total_bits)
            byte_index, bit_pos = divmod(bit_index, 8)
            ba[byte_index] ^= (1 << bit_pos)
        return bytes(ba)

    @staticmethod
    def _segment_dropout(data: bytes, drop_ratio: float) -> bytes:
        """
        从字节流中随机丢弃一个连续的片段。
        优化后的逻辑确保每个字节被丢弃的概率是均匀的。
        """
        data_len = len(data)
        if data_len <= 1:
            return data

        # 1. 计算要丢弃的片段长度
        seg_len = max(1, int(data_len * drop_ratio))
        # 确保不会丢弃所有数据，至少保留一个字节
        if seg_len >= data_len:
            seg_len = data_len - 1
        
        if seg_len == 0:
            return data

        # 2. 扩展随机选择的范围
        # 旧范围: [0, data_len - seg_len]
        # 新范围: [-seg_len + 1, data_len - 1]
        # 这允许丢弃的片段可以“部分悬挂”在数据流的开始或结束位置之外，
        # 从而确保了头部和尾部字节有相同的机会被丢弃。
        start = np.random.randint(-seg_len + 1, data_len)

        # 3. 计算实际要操作的切片索引
        # 如果 start 是负数，实际的切片从 0 开始
        clip_start = max(0, start)
        # 如果 start + seg_len 超出范围，实际的切片到末尾结束
        clip_end = min(data_len, start + seg_len)

        # 如果计算出的切片无效（例如，整个片段都在数据流外部），则不进行任何操作
        if clip_start >= clip_end:
            return data

        # 4. 拼接字节流，跳过 [clip_start, clip_end) 这个区间
        return data[:clip_start] + data[clip_end:]
    
    @staticmethod
    def _header_truncation(data: bytes, trunc_size: int) -> bytes:
        if len(data) <= 1: return data
        trunc_size = min(max(0, trunc_size), len(data) - 1)
        return data[trunc_size:] if trunc_size > 0 else data

    # @staticmethod
    # def _tail_truncation(data: bytes, trunc_size: int) -> bytes:
    #     if len(data) <= 1: return data
    #     trunc_size = min(max(0, trunc_size), len(data) - 1)
    #     print(f"The length :", len(data))
    #     print(f"The trunc_size :", trunc_size)
    #     print(f"After truncation length :", len(data[:-trunc_size]))
    #     return data[:-trunc_size] if trunc_size > 0 else data
    @staticmethod
    def _tail_truncation(data: bytes, trunc_ratio: float) -> bytes:
        """根据给定的比例截断字节流的尾部。"""
        if len(data) <= 1 or not (0 < trunc_ratio < 1):
            return data
        
        # 1. 根据比例计算要截断的字节数
        trunc_size = int(len(data) * trunc_ratio)
        
        # 2. 确保截断大小在安全范围内（至少保留一个字节）
        trunc_size = min(max(0, trunc_size), len(data) - 1)
        
        # 3. 执行截断
        return data[:-trunc_size] if trunc_size > 0 else data

    def __call__(self, data: Dict[str, Union[torch.Tensor, int]]) -> List[Dict[str, Union[torch.Tensor, int]]]:
        """
        将单个样本的 'samples' 字段增强为多个损坏版本。
        返回一个字典列表，每个字典只包含 'samples' 和 'corruption_marker'。
        """
        if self.level == "none" or not self.corruption_types or not self.params:
            return [{"samples": data["samples"], "corruption_marker": "none"}]

        int_tensor = data["samples"]
        byte_values = (int_tensor.numpy() & 0xFF).astype(np.uint8)
        original_bytes = byte_values.tobytes()

        # (这里的调试语句可以保留或移除)
        print(f"[DEBUG Corrupter] Received original byte stream. len={len(original_bytes)}, startswith={original_bytes[:16].hex(' ')}")

        augmented_samples = []

        for corruption_type in self.corruption_types:
            corrupted_bytes = None
            # (这里的调试语句可以保留或移除)
            print(f"[DEBUG Corrupter] Applying '{corruption_type}' with level '{self.level}'...")

            if corruption_type == "bit_flip":
                corrupted_bytes = self._random_bit_flip(original_bytes, self.params["bit_flip"])
            elif corruption_type == "segment_dropout":
                corrupted_bytes = self._segment_dropout(original_bytes, self.params["drop"])
            elif corruption_type == "header_truncation":
                corrupted_bytes = self._header_truncation(original_bytes, self.params["head"])
            elif corruption_type == "tail_truncation":
                corrupted_bytes = self._tail_truncation(original_bytes, self.params["tail"])
            
            if corrupted_bytes is not None:
                # (这里的调试语句可以保留或移除)
                print(f"    -> Original len: {len(original_bytes)}, Corrupted len: {len(corrupted_bytes)}")

                # 只创建包含 'samples' 和标记的新字典
                new_sample_dict = {}
                buf = np.frombuffer(corrupted_bytes, dtype=np.uint8)
                new_sample_dict["samples"] = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
                new_sample_dict["corruption_marker"] = f"{corruption_type}_{self.level}"
                
                augmented_samples.append(new_sample_dict)

        return augmented_samples if augmented_samples else [{"samples": data["samples"], "corruption_marker": "none"}]

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.level",
            type=str, default="light", choices=["none", "light", "medium", "heavy"],
            help="强度级别，控制所有损坏的参数。"
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.types",
            type=str, nargs="+", default=["tail_truncation"],
            choices=["bit_flip", "segment_dropout", "header_truncation", "tail_truncation"],
            help="要应用的损坏类型列表。每种类型都会生成一个独立的样本。"
        )
        return parser


@TRANSFORMATIONS_REGISTRY.register(name="pil_save", type="image_torch")
class PILSave(BaseTransformation):
    """
    使用支持的文件编码对图像进行编码。
    （现在不再在这里处理损坏逻辑,损坏逻辑我专门封装一个类实现，不然太过于臃肿了）
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.file_encoding = getattr(opts, "image_augmentation.pil_save.file_encoding")
        self.quality = getattr(opts, "image_augmentation.pil_save.quality")
        self.opts = opts

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        x = data["samples"]
        
        # --- 调试语句 5：确认 PILSave 被调用 ---
        print(f"[DEBUG PILSave] Encoding image to '{self.file_encoding}'...")
        # ---------------------------------------

        byte_stream = None # 用于调试

        if self.file_encoding == "fCHW":
            x = (x * 255).byte().to(dtype=torch.int32).reshape(-1)
        elif self.file_encoding == "fHWC":
            x = (x * 255).byte().to(dtype=torch.int32).permute(1, 2, 0).reshape(-1)
        elif self.file_encoding == "TIFF":
            byte_stream = _image_to_bytes(x, format="tiff")
            x = _bytes_to_int32(byte_stream)
        elif self.file_encoding == "PNG":
            byte_stream = _image_to_bytes(x, format="png", compress_level=0)
            x = _bytes_to_int32(byte_stream)
        elif self.file_encoding == "JPEG":
            quality = getattr(self.opts, "image_augmentation.pil_save.quality")
            byte_stream = _image_to_bytes(x, format="jpeg", quality=quality)
            x = _bytes_to_int32(byte_stream)
        else:
            raise NotImplementedError(
                f"Invalid file encoding {self.file_encoding}. Expected one of 'fCHW, fHWC, TIFF, PNG, JPEG'."
            )
        
        # --- 调试语句 6：打印生成的原始字节流信息，以便与 Corrupter 收到的进行比对 ---
        if byte_stream:
            raw_bytes = byte_stream.getvalue()
            print(f"[DEBUG PILSave] Generated byte stream. len={len(raw_bytes)}, startswith={raw_bytes[:16].hex(' ')}")
        # -------------------------------------------------------------------------

        data["samples"] = x
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(file_encoding={self.file_encoding}, quality={self.quality})"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.pil-save.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.pil-save.file-encoding",
            choices=("fCHW", "fHWC", "TIFF", "PNG", "JPEG"),
            help="The type of file encoding to use. Defaults to TIFF.",
            default="TIFF",
        )
        group.add_argument(
            "--image-augmentation.pil-save.quality",
            help="JPEG quality if using JPEG encoding. Defaults to 100.",
            type=int,
            default=100,
        )
        return parser


@TRANSFORMATIONS_REGISTRY.register(name="shuffle_bytes", type="image_torch")
class ShuffleBytes(BaseTransformation):
    """
    Reorder the bytes in a 1-dimensional buffer.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.mode = getattr(opts, "image_augmentation.shuffle_bytes.mode")
        self.stride = getattr(opts, "image_augmentation.shuffle_bytes.stride")
        window_size = getattr(opts, "image_augmentation.shuffle_bytes.window_size")
        self.window_shuffle = torch.randperm(window_size)

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Reorder the bytes of a 1-dimensional buffer.

        Args:
            data: A dictionary containing a key called "samples", which contains
                a tensor of shape [N], where N is the number of bytes.

        Returns:
            The transformed data.
        """
        x = data["samples"]
        if not x.dim() == 1:
            raise ValueError(f"Expected 1d input, got {x.shape}")

        if self.mode == "reverse":
            x = torch.fliplr(x.view(1, -1))[0]
        elif self.mode == "random_shuffle":
            x = x[torch.randperm(x.shape[0])]
        elif self.mode == "cyclic_half_length":
            x = torch.roll(x, x.shape[0] // 2)
        elif self.mode == "stride":
            l = len(x)
            values = []
            for i in range(self.stride):
                values.append(x[i :: self.stride])
            x = torch.cat(values, dim=0)
            assert len(x) == l
        elif self.mode == "window_shuffle":
            l = len(x)
            window_size = self.window_shuffle.shape[0]
            num_windows = l // window_size
            values = []
            for i in range(num_windows):
                chunk = x[i * window_size : (i + 1) * window_size]
                values.append(chunk[self.window_shuffle])

            # Add the last bits that fall outside the shuffling window.
            values.append(x[num_windows * window_size :])
            x = torch.cat(values, dim=0)
            assert len(x) == l
        else:
            raise NotImplementedError(
                f"mode={self.mode} not implemented. Expected one of 'reverse, random_shuffle, cyclic_half_length, stride, window_shuffle'."
            )
        data["samples"] = x
        return data

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.shuffle-bytes.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.shuffle-bytes.mode",
            default="reverse",
            help="The mode to use when shuffling bytes. Defaults to 'reverse'.",
            choices=(
                "reverse",
                "random_shuffle",
                "cyclic_half_length",
                "stride",
                "window_shuffle",
            ),
        )
        group.add_argument(
            "--image-augmentation.shuffle-bytes.stride",
            type=int,
            default=1024,
            help="The stride of the window used in shuffling operations that are windowed. Defaults to 1024.",
        )
        group.add_argument(
            "--image-augmentation.shuffle-bytes.window-size",
            type=int,
            default=1024,
            help="The size of the window used in shuffling operations that are windowed. Defaults to 1024.",
        )
        return parser


@TRANSFORMATIONS_REGISTRY.register(name="mask_positions", type="image_torch")
class MaskPositions(BaseTransformation):
    """
    Mask out values in a 1-dimensional buffer using a fixed masking pattern.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.keep_frac = getattr(opts, "image_augmentation.mask_positions.keep_frac")
        self._cached_masks = None

    def _generate_masks(self, N: int) -> torch.Tensor:
        if self._cached_masks is None:
            g = torch.Generator()
            # We want to fix the mask across all inputs, so we fix the seed.
            # Choose a seed with a good balance of 0 and 1 bits. See:
            # https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator.manual_seed
            g.manual_seed(2147483647)
            random_mask = torch.zeros([N], requires_grad=False, dtype=torch.bool)
            random_mask[torch.randperm(N, generator=g)[: int(self.keep_frac * N)]] = 1
            self._cached_masks = random_mask
        return self._cached_masks

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Mask values in a 1-dimensional buffer with a fixed masking pattern.

        Args:
            data: A dictionary containing a key called "samples", which contains
                a tensor of shape [N], where N is the number of bytes.

        Returns:
            The transformed data.
        """
        x = data["samples"]
        mask = self._generate_masks(x.shape[0])
        x = x[mask]
        data["samples"] = x
        return data

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.mask-positions.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.mask-positions.keep-frac",
            type=float,
            default=0.5,
            help="The fraction of bytes to keep. Defaults to 0.5.",
        )
        return parser


@TRANSFORMATIONS_REGISTRY.register(name="byte_permutation", type="image_torch")
class BytePermutation(BaseTransformation):
    """
    Remap byte values in [0, 255] to new values in [0, 255] using a permutation.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)

        g = torch.Generator()
        g.manual_seed(2147483647)
        self.mask = torch.randperm(256, generator=g)

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Remap byte values in [0, 255] to new values in [0, 255] using a permutation.

        Args:
            data: A dictionary containing a key called "samples", which contains
                a tensor of shape [N], where N is the number of bytes.

        Returns:
            The transformed data.
        """
        x = data["samples"]

        if x.dim() != 1:
            raise ValueError(f"Expected 1d tensor. Got {x.shape}.")
        x = torch.index_select(self.mask, dim=0, index=x)
        data["samples"] = x
        return data

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.byte-permutation.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        return parser


@TRANSFORMATIONS_REGISTRY.register(name="random_uniform", type="image_torch")
class RandomUniformNoise(BaseTransformation):
    """
    Add random uniform noise to integer values.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.opts = opts

        self.width_range = getattr(
            opts, "image_augmentation.random_uniform.width_range"
        )

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, int]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Add random uniform noise to byte values.

        Args:
            data: A dict containing a tensor in its "samples" key. The tensor
                contains integers representing byte values. Integers are used
                because negative padding values may be added later. The shape
                of the tenor is [N], where N is the number of bytes.

        Returns:
            The transformed data.
        """
        x = data["samples"]
        noise = torch.randint_like(x, self.width_range[0], self.width_range[1] + 1)
        dtype = x.dtype
        x = x.int()
        x = x + noise
        x = x % 256
        x = x.to(dtype)
        data["samples"] = x
        return data

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--image-augmentation.random-uniform.enable",
            action="store_true",
            help="Use {}. This flag is useful when you want to study the effect of different "
            "transforms.".format(cls.__name__),
        )
        group.add_argument(
            "--image-augmentation.random-uniform.width-range",
            type=int,
            nargs=2,
            default=[-5, 5],
            help="The range of values from which to add noise. It is specified"
            " as [low, high] (inclusive). Defaults to [-5, 5].",
        )
        return parser
