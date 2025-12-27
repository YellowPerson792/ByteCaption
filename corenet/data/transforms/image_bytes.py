#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import io
from typing import Any, Dict, Union, List

import numpy as np
import torch
from PIL import Image

from corenet.data.transforms import TRANSFORMATIONS_REGISTRY, BaseTransformation
from corenet.data.transforms import jpeg_corruption


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
        raw_level = getattr(opts, "image_augmentation.byte_stream_corrupter.level", "S0")
        self.level = jpeg_corruption.normalize_level(raw_level)
        self.param_overrides = self._read_overrides(opts)
        self.pipeline = jpeg_corruption.JPEGCorruptionPipeline(
            corruption_types=self.corruption_types,
            level=self.level,
            overrides=self.param_overrides,
        )

    def _read_overrides(self, opts: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
        overrides: Dict[str, Dict[str, Any]] = {
            "rbbf": {},
            "rbsl": {},
            "metadata_loss": {},
        }
        overrides["rbbf"]["trigger_prob"] = getattr(
            opts, "image_augmentation.byte_stream_corrupter.rbbf.trigger_prob", None
        )
        overrides["rbbf"]["burst_lambda"] = getattr(
            opts, "image_augmentation.byte_stream_corrupter.rbbf.burst_lambda", None
        )
        overrides["rbbf"]["bit_error_rate"] = getattr(
            opts, "image_augmentation.byte_stream_corrupter.rbbf.bit_error_rate", None
        )
        overrides["rbsl"]["trigger_prob"] = getattr(
            opts, "image_augmentation.byte_stream_corrupter.rbsl.trigger_prob", None
        )
        overrides["rbsl"]["burst_lambda"] = getattr(
            opts, "image_augmentation.byte_stream_corrupter.rbsl.burst_lambda", None
        )
        overrides["rbsl"]["max_drop_ratio"] = getattr(
            opts, "image_augmentation.byte_stream_corrupter.rbsl.max_drop_ratio", None
        )
        overrides["metadata_loss"]["strip_app_segments"] = getattr(
            opts, "image_augmentation.byte_stream_corrupter.metadata_loss.strip_app_segments", None
        )
        overrides["metadata_loss"]["zero_prefix_bytes"] = getattr(
            opts, "image_augmentation.byte_stream_corrupter.metadata_loss.zero_prefix_bytes", None
        )
        overrides["metadata_loss"]["body_trim_ratio"] = getattr(
            opts, "image_augmentation.byte_stream_corrupter.metadata_loss.body_trim_ratio", None
        )
        cleaned: Dict[str, Dict[str, Any]] = {}
        for ctype, params in overrides.items():
            valid = {k: v for k, v in params.items() if v is not None}
            if valid:
                cleaned[ctype] = valid
        return cleaned

    def __call__(self, data: Dict[str, Union[torch.Tensor, int]]) -> List[Dict[str, Union[torch.Tensor, int]]]:
        """
        将单个样本的 'samples' 字段增强为多个损坏版本。
        返回一个字典列表，每个字典只包含 'samples' 和 'corruption_marker'。
        """
        if not self.pipeline.is_enabled():
            return [{"samples": data["samples"], "corruption_marker": "none"}]

        int_tensor = data["samples"]
        byte_values = (int_tensor.numpy() & 0xFF).astype(np.uint8)
        original_bytes = byte_values.tobytes()

        augmented_samples = []
        for corrupted_bytes, marker in self.pipeline.apply(original_bytes):
            new_sample_dict: Dict[str, Union[torch.Tensor, str]] = {}
            buf = np.frombuffer(corrupted_bytes, dtype=np.uint8)
            new_sample_dict["samples"] = torch.from_numpy(buf.copy()).to(dtype=torch.int32)
            new_sample_dict["corruption_marker"] = marker
            augmented_samples.append(new_sample_dict)

        return augmented_samples if augmented_samples else [{"samples": data["samples"], "corruption_marker": "none"}]

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        level_choices = sorted(
            set(jpeg_corruption.available_levels() + ["none", "light", "medium", "heavy"])
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.level",
            type=str,
            default="S0",
            choices=level_choices,
            help="强度级别（S0-S5/M0-M1），兼容 none/light/medium/heavy 别名。"
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.types",
            type=str,
            nargs="+",
            default=["rbbf"],
            choices=["rbbf", "rbsl", "metadata_loss", "none"],
            help="要应用的损坏类型列表。每种类型都会生成一个独立的样本。"
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.rbbf.trigger-prob",
            type=float,
            default=None,
            help="RBBF：每个位触发突发的概率（覆盖 preset）。"
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.rbbf.burst-lambda",
            type=float,
            default=None,
            help="RBBF：突发长度 Poisson 均值（覆盖 preset）。"
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.rbbf.bit-error-rate",
            type=float,
            default=None,
            help="RBBF：突发内每个位翻转概率（覆盖 preset）。"
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.rbsl.trigger-prob",
            type=float,
            default=None,
            help="RBSL：每个字节触发丢失片段的概率（覆盖 preset）。"
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.rbsl.burst-lambda",
            type=float,
            default=None,
            help="RBSL：丢失片段长度 Poisson 均值（覆盖 preset）。"
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.rbsl.max-drop-ratio",
            type=float,
            default=None,
            help="RBSL：最多允许丢失的比例（覆盖 preset）。"
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.metadata-loss.strip-app-segments",
            type=int,
            default=None,
            help="ML：删除的 APP 段数量（覆盖 preset）。"
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.metadata-loss.zero-prefix-bytes",
            type=int,
            default=None,
            help="ML：将头部前若干字节置零（覆盖 preset）。"
        )
        group.add_argument(
            "--image-augmentation.byte-stream-corrupter.metadata-loss.body-trim-ratio",
            type=float,
            default=None,
            help="ML：对 SOS 之后熵编码区的截断比例（覆盖 preset）。"
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
            help="JPEG quality if using JPEG encoding. Defaults to 60.",
            type=int,
            default=60,
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
