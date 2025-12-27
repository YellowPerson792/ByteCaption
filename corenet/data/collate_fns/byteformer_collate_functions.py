#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

"""This file contains collate functions used by ByteFormer.

Since the model operates on a variety of input types, these collate functions
are not associated with a particular dataset.

These transforms are applied before the model (rather than inside the model) to
take advantage of parallelism, and to avoid the need to move tensors from the
GPU, back to the CPU, then back to GPU (since these transforms cannot be done
on GPU).
"""

import argparse
from typing import Dict, List, Mapping, Optional, Union

import torch
from torch import Tensor
from torch.nn import functional

from corenet.data.collate_fns import COLLATE_FN_REGISTRY, collate_functions
from corenet.data.transforms import audio_bytes, image_bytes


@COLLATE_FN_REGISTRY.register(name="byteformer_image_collate_fn")
def byteformer_image_collate_fn(
    batch: List[Mapping[str, Tensor]], opts: argparse.Namespace
) -> Mapping[str, Tensor]:
    """
    Apply augmentations specific to `ByteFormer <https://arxiv.org/abs/2306.00238>`_
    image training, then perform padded collation.

    Args:
        batch: The batch of data.
        opts: The global arguments.

    Returns:
        The modified batch.
    """
    # 如果样本已经是字节流（int32 1D），则跳过 PILSave，避免重复编码
    first = batch[0]["samples"] if isinstance(batch, list) and batch else None
    needs_pil_save = True
    try:
        if isinstance(first, torch.Tensor) and first.dtype == torch.int32 and first.dim() == 1:
            needs_pil_save = False
    except Exception:
        needs_pil_save = True
    if needs_pil_save:
        batch = apply_pil_save(batch, opts)
    # 在 PILSave 之后，应用新的字节流损坏
    batch = apply_byte_stream_corrupter(batch, opts)
    batch = apply_shuffle_bytes(batch, opts)
    batch = apply_mask_positions(batch, opts)
    batch = apply_random_uniform_noise(batch, opts)
    batch = apply_byte_permutation(batch, opts)
    batch = apply_padding(batch, opts)
    batch = collate_functions.pytorch_default_collate_fn(batch, opts)
    return batch


def apply_padding(
    batch: List[Mapping[str, Union[Dict[str, Tensor], Tensor]]],
    opts: argparse.Namespace,
    key: Optional[str] = None,
) -> List[Mapping[str, Tensor]]:
    """
    Apply padding to make samples the same length.

    The input is a list of dictionaries of the form:
        [{"samples": @entry, ...}, ...].
    If @key is specified, @entry has the form {@key: @value}, where @value
    corresponds to the entry that should be padded. Otherwise, @entry is assumed
    to be a tensor.

    The tensor mentioned in the above paragraph will have shape [batch_size,
        sequence_length, ...].

    Args:
        batch: The batch of data.
        opts: The global arguments.
        key: The key of the sample element to pad. If @key is None, the entry
            is assumed to be a tensor.

    Returns:
        The modified batch of size [batch_size, padded_sequence_length, ...].
    """

    def get_entry(
        entry: Union[Dict[str, Tensor], Tensor], key: Optional[str]
    ) -> Tensor:
        """
        Helper function to deal with the cases where entries in the samples.

        Args:
            entry: Either a tensor of shape [batch_size, sequence_length, ...],
                or a dictionary containing {@key: tensor of shape
                [batch_size, sequence_length, ...]}.
        Returns:
            A tensor of shape [batch_size, ...].
        """
        if isinstance(entry, dict):
            return entry[key]
        if key is not None:
            raise ValueError(f"Key should not be specified if entries are not dicts.")
        return entry

    if get_entry(batch[0]["samples"], key).dim() != 1:
        # Padding only applies to 1d tensors.
        return batch
    padding_idx = getattr(opts, "model.classification.byteformer.padding_index")
    # Tensors have shape [batch_size, sequence_length, ...]. Get the maximum
    # sequence length.
    padded_seq_len = max(get_entry(be["samples"], key).shape[0] for be in batch)
    for elem in batch:
        sample = get_entry(elem["samples"], key)  # [batch_size, sequence_length, ...].
        sample = functional.pad(
            sample, (0, padded_seq_len - sample.shape[0]), value=padding_idx
        )  # [batch_size, padded_sequence_length, ...].
        if isinstance(elem["samples"], dict):
            elem["samples"][key] = sample
        else:
            elem["samples"] = sample
    return batch


def apply_pil_save(
    batch: List[Mapping[str, Tensor]],
    opts: argparse.Namespace,
) -> List[Mapping[str, Tensor]]:
    """
    Apply the PILSave transform to each batch element.

    Args:
        batch: The batch of data.
        opts: The global arguments.

    Returns:
        The modified batch.
    """
    if getattr(opts, "image_augmentation.pil_save.enable"):
        transform = image_bytes.PILSave(opts)
        for i, elem in enumerate(batch):
            sample = elem.get("samples") if isinstance(elem, dict) else None
            # 仅当样本看起来是规范化图像张量（[C,H,W] 浮点且范围[0,1]）时才应用
            if isinstance(sample, torch.Tensor) and sample.dim() == 3 and sample.dtype in (torch.float32, torch.float64):
                batch[i] = transform(elem)
    return batch


# --- 损坏函数 ---
def apply_byte_stream_corrupter(
    batch: List[Mapping[str, Tensor]],
    opts: argparse.Namespace,
) -> List[Mapping[str, Tensor]]:
    """
    Apply the ByteStreamCorrupter transform, which may augment one sample into many.
    """
    transform = image_bytes.ByteStreamCorrupter(opts)
    if not transform.pipeline.is_enabled():
        return batch

    new_batch = []
    for i, elem in enumerate(batch):
        augmented_samples = transform(elem)
        new_batch.extend(augmented_samples)

    return new_batch


def apply_shuffle_bytes(
    batch: List[Mapping[str, Tensor]],
    opts: argparse.Namespace,
) -> List[Mapping[str, Tensor]]:
    """
    Apply the ShuffleBytes transform to each batch element.

    Args:
        batch: The batch of data.
        opts: The global arguments.

    Returns:
        The modified batch.
    """
    if getattr(opts, "image_augmentation.shuffle_bytes.enable"):
        transform = image_bytes.ShuffleBytes(opts)
        for i, elem in enumerate(batch):
            batch[i] = transform(elem)
    return batch


def apply_mask_positions(
    batch: List[Mapping[str, Tensor]], opts: argparse.Namespace
) -> List[Mapping[str, Tensor]]:
    """
    Apply the MaskPositions transform to each batch element.

    Args:
        batch: The batch of data.
        opts: The global arguments.

    Returns:
        The modified batch.
    """
    if getattr(opts, "image_augmentation.mask_positions.enable"):
        transform = image_bytes.MaskPositions(opts)
        for i, elem in enumerate(batch):
            batch[i] = transform(elem)
    return batch


def apply_random_uniform_noise(
    batch: List[Mapping[str, Tensor]], opts: argparse.Namespace
) -> List[Mapping[str, Tensor]]:
    """
    Apply the RandomUniformNoise transform to each batch element.

    Args:
        batch: The batch of data.
        opts: The global arguments.

    Returns:
        The modified batch.
    """
    if getattr(opts, "image_augmentation.random_uniform.enable"):
        transform = image_bytes.RandomUniformNoise(opts)
        for i, elem in enumerate(batch):
            batch[i] = transform(elem)
    return batch


def apply_byte_permutation(
    batch: List[Mapping[str, Tensor]], opts: argparse.Namespace
) -> List[Mapping[str, Tensor]]:
    """
    Apply the BytePermutation transform to each batch element.

    Args:
        batch: The batch of data.
        opts: The global arguments.

    Returns:
        The modified batch.
    """
    if getattr(opts, "image_augmentation.byte_permutation.enable"):
        transform = image_bytes.BytePermutation(opts)
        for i, elem in enumerate(batch):
            batch[i] = transform(elem)
    return batch


def apply_torchaudio_save(
    batch: List[Mapping[str, Tensor]], opts: argparse.Namespace
) -> List[Mapping[str, Tensor]]:
    """
    Apply the TorchaudioSave transform to each batch element.

    Args:
        batch: The batch of data.
        opts: The global arguments.

    Returns:
        The modified batch.
    """
    if getattr(opts, "audio_augmentation.torchaudio_save.enable"):
        transform = audio_bytes.TorchaudioSave(opts)
        for i, elem in enumerate(batch):
            batch[i] = transform(elem)
    return batch


@COLLATE_FN_REGISTRY.register(name="byteformer_audio_collate_fn")
def byteformer_audio_collate_fn(
    batch: List[Mapping[str, Tensor]], opts: argparse.Namespace
) -> Mapping[str, Tensor]:
    """
    Apply augmentations specific to ByteFormer audio training, then perform
    padded collation.

    Args:
        batch: The batch of data.
        opts: The global arguments.

    Returns:
        The modified batch.
    """
    batch = apply_torchaudio_save(batch, opts)
    batch = apply_padding(batch, opts, "audio")
    # Remove the metadata, which is no longer needed.
    for elem in batch:
        del elem["metadata"]
    batch = collate_functions.pytorch_default_collate_fn(batch, opts)
    return batch
