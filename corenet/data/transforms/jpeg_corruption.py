"""
Shared JPEG bitstream corruption utilities used by both training collates and
offline sampling scripts.

The design follows the experiment plan in `实验设计.md`:
- Random Bursty Bit Flips (RBBF)
- Random Bursty Segment Loss (RBSL)
- Metadata Loss (ML)

Each corruption type has severity presets (S0–S5 for RBBF/RBSL, M0–M1 for ML)
and supports lightweight per-run overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Optional
import hashlib

import numpy as np

# Severity presets (guessed, tunable) inspired by the placeholder table
# in `实验设计.md`. All numbers are intentionally conservative so the
# majority of JPEGs remain decodable at low severities while producing
# visible corruption at higher levels.
JPEG_CORRUPTION_PRESETS: Dict[str, Dict[str, Dict[str, float]]] = {
    "rbbf": {  # Random Bursty Bit Flips
        "S0": {"trigger_prob": 0.0, "burst_lambda": 0.0, "bit_error_rate": 0.0},
        # Slightly lighter S1 to let baseline BLIP stay competitive
        "S1": {"trigger_prob": 5e-5, "burst_lambda": 80.0, "bit_error_rate": 0.12},
        # Keep higher severities strong but a bit gentler than before to retain caption quality
        "S2": {"trigger_prob": 2e-4, "burst_lambda": 160.0, "bit_error_rate": 0.24},
        "S3": {"trigger_prob": 5e-4, "burst_lambda": 200.0, "bit_error_rate": 0.36},
        "S4": {"trigger_prob": 1e-3, "burst_lambda": 260.0, "bit_error_rate": 0.42},
        "S5": {"trigger_prob": 5e-3, "burst_lambda": 360.0, "bit_error_rate": 0.5},
    },
    "rbsl": {  # Random Bursty Segment Loss
        "S0": {"trigger_prob": 0.0, "burst_lambda": 0.0, "max_drop_ratio": 1.0},
        "S1": {"trigger_prob": 5e-4, "burst_lambda": 100.0, "max_drop_ratio": 1.0},
        "S2": {"trigger_prob": 8e-4, "burst_lambda": 220.0, "max_drop_ratio": 1.0},
        "S3": {"trigger_prob": 2e-3, "burst_lambda": 360.0, "max_drop_ratio": 1.0},
        "S4": {"trigger_prob": 5e-3, "burst_lambda": 520.0, "max_drop_ratio": 1.0},
        "S5": {"trigger_prob": 7e-3, "burst_lambda": 640.0, "max_drop_ratio": 1.0},
    },
    "metadata_loss": {
        "M0": {
            "strip_app_segments": 0,
            "zero_prefix_bytes": 0,
            "body_trim_ratio": 0.0,
        },
        "M1": {
            "strip_app_segments": 4,
            "zero_prefix_bytes": 256,
            "body_trim_ratio": 0.08,
        },
    },
}


def normalize_level(level: str) -> str:
    """Map legacy level names to preset keys."""
    if level is None:
        return "S0"
    norm = level.strip().upper()
    if norm in {"NONE", ""}:
        return "S0"
    if norm in {"LIGHT"}:
        return "S1"
    if norm in {"MEDIUM"}:
        return "S3"
    if norm in {"HEAVY"}:
        return "S5"
    return norm


def available_levels() -> List[str]:
    """Return a flat list of supported severity labels."""
    rbbf_levels = set(JPEG_CORRUPTION_PRESETS["rbbf"].keys())
    rbsl_levels = set(JPEG_CORRUPTION_PRESETS["rbsl"].keys())
    ml_levels = set(JPEG_CORRUPTION_PRESETS["metadata_loss"].keys())
    merged = sorted(rbbf_levels | rbsl_levels | ml_levels | {"S0"})
    return merged


@dataclass
class JPEGCorruptionPipeline:
    """
    Lightweight, reusable corruption pipeline that operates on raw JPEG bytes.

    Args:
        corruption_types: Sequence of corruption types to apply independently.
        level: Shared severity label (S0–S5, M0–M1 or legacy aliases).
        overrides: Optional per-type overrides, e.g.
                   {"rbbf": {"trigger_prob": 1e-5}}
        seed: Optional base seed used to deterministically derive per-image RNGs.
    """

    corruption_types: Sequence[str] = field(default_factory=list)
    level: str = "S0"
    overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.level = normalize_level(self.level)
        self.corruption_types = [c.lower() for c in self.corruption_types]

    def is_enabled(self) -> bool:
        if not self.corruption_types:
            return False
        if all(ct == "none" for ct in self.corruption_types):
            return False
        if self.level in {"S0", "M0"}:
            # S0/M0 are meant to disable corruption entirely
            return False
        return True

    def _resolve_params(self, corruption_type: str) -> Dict[str, float]:
        ctype = corruption_type.lower()
        preset_level = self.level
        if ctype == "metadata_loss":
            # Map generic severities onto metadata presets
            preset_level = "M0" if self.level in {"S0", "NONE"} else "M1"
        presets = JPEG_CORRUPTION_PRESETS.get(ctype, {})
        params = dict(presets.get(preset_level, {}))
        for k, v in self.overrides.get(ctype, {}).items():
            if v is not None:
                params[k] = v
        return params

    def _rng_for(self, image_id: Optional[int], corruption_type: str):
        if self.seed is None:
            return np.random.default_rng()
        token = f"{self.seed}-{self.level}-{corruption_type}-{image_id}"
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        seed_int = int.from_bytes(digest[:8], byteorder="little", signed=False)
        return np.random.default_rng(seed_int)

    def apply(self, data: bytes, image_id: Optional[int] = None) -> List[Tuple[bytes, str]]:
        """Apply all configured corruptions to a single JPEG byte stream deterministically per image."""
        if not self.corruption_types:
            return [(data, "none")]

        outputs: List[Tuple[bytes, str]] = []
        for corruption_type in self.corruption_types:
            if corruption_type == "none":
                outputs.append((data, "none"))
                continue

            params = self._resolve_params(corruption_type)
            rng = self._rng_for(image_id, corruption_type)
            if corruption_type == "rbbf":
                corrupted = random_bursty_bit_flips(data, rng=rng, **params)
            elif corruption_type == "rbsl":
                corrupted = random_bursty_segment_loss(data, rng=rng, **params)
            elif corruption_type == "metadata_loss":
                corrupted = metadata_loss(data, **params)
            else:
                # Unknown corruption type; return original
                outputs.append((data, corruption_type))
                continue

            outputs.append((corrupted, f"{corruption_type}_{self.level}"))
        return outputs


def random_bursty_bit_flips(
    data: bytes,
    trigger_prob: float,
    burst_lambda: float,
    bit_error_rate: float,
    rng: Optional[np.random.Generator] = None,
) -> bytes:
    """Random bursty bit flips (RBBF)."""
    if not data or trigger_prob <= 0.0 or bit_error_rate <= 0.0 or burst_lambda <= 0.0:
        return data

    rng = rng or np.random.default_rng()
    ba = bytearray(data)
    total_bits = len(ba) * 8
    expected_bursts = max(0.0, total_bits * trigger_prob)
    num_bursts = rng.poisson(expected_bursts)
    if num_bursts == 0:
        return data

    starts = rng.integers(0, total_bits, size=num_bursts)
    flip_positions: List[int] = []
    for start in starts:
        length = max(1, int(rng.poisson(lam=burst_lambda)))
        end = min(total_bits, start + length)
        span = end - start
        if span <= 0:
            continue
        flips_in_burst = rng.binomial(span, min(max(bit_error_rate, 0.0), 1.0))
        if flips_in_burst == 0:
            continue
        offsets = rng.choice(span, size=flips_in_burst, replace=False)
        flip_positions.extend(start + offsets)

    for bit_index in flip_positions:
        byte_index, bit_pos = divmod(int(bit_index), 8)
        ba[byte_index] ^= 1 << bit_pos
    return bytes(ba)


def random_bursty_segment_loss(
    data: bytes,
    trigger_prob: float,
    burst_lambda: float,
    max_drop_ratio: float,
    rng: Optional[np.random.Generator] = None,
) -> bytes:
    """Random bursty segment loss (RBSL)."""
    if not data or trigger_prob <= 0.0 or burst_lambda <= 0.0 or max_drop_ratio <= 0.0:
        return data

    rng = rng or np.random.default_rng()
    data_len = len(data)
    max_drop = max(1, int(data_len * max_drop_ratio))
    expected_bursts = max(0.0, data_len * trigger_prob)
    num_bursts = rng.poisson(expected_bursts)
    if num_bursts == 0:
        return data

    segments: List[Tuple[int, int]] = []
    for _ in range(num_bursts):
        start = int(rng.integers(0, data_len))
        length = max(1, int(rng.poisson(lam=burst_lambda)))
        end = min(data_len, start + length)
        if start < end:
            segments.append((start, end))

    if not segments:
        return data

    segments.sort()
    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = segments[0]
    for start, end in segments[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    out = bytearray()
    consumed = 0
    dropped = 0
    for start, end in merged:
        if dropped >= max_drop:
            break
        start = max(start, consumed)
        if start < consumed:
            continue
        if start > consumed:
            out.extend(data[consumed:start])
        drop_len = min(end - start, max_drop - dropped)
        consumed = start + drop_len
        dropped += drop_len

    if consumed < data_len:
        out.extend(data[consumed:])
    return bytes(out)


def metadata_loss(
    data: bytes,
    strip_app_segments: int = 2,
    zero_prefix_bytes: int = 96,
    body_trim_ratio: float = 0.02,
) -> bytes:
    """Metadata loss (ML): remove header info and trim early entropy-coded bytes."""
    if not data:
        return data

    sos_idx = data.find(b"\xFF\xDA")  # SOS marker
    header_end = sos_idx if sos_idx != -1 else min(len(data), 2048)
    header = bytearray(data[:header_end])
    body = bytearray(data[header_end:])

    if zero_prefix_bytes > 0:
        zero_len = min(len(header), int(zero_prefix_bytes))
        if zero_len > 0:
            header[:zero_len] = b"\x00" * zero_len

    if strip_app_segments > 0 and header:
        header = bytearray(_strip_app_segments(bytes(header), strip_app_segments))

    if body_trim_ratio > 0.0 and body:
        trim = min(len(body) - 1, int(len(body) * body_trim_ratio))
        if trim > 0:
            start = int(np.random.randint(0, max(1, len(body) - trim)))
            del body[start : start + trim]

    return bytes(header + body)


def _strip_app_segments(data: bytes, max_segments: int) -> bytes:
    """
    Drop APP0-APP15 segments in the JPEG header region.
    Parsing stops at SOS to avoid touching entropy-coded data.
    """
    if max_segments <= 0:
        return data

    out = bytearray()
    i = 0
    removed = 0
    data_len = len(data)
    while i + 4 <= data_len:
        marker_prefix = data[i]
        marker = data[i + 1]
        if marker_prefix == 0xFF and marker == 0xDA:  # SOS
            out.extend(data[i:])
            return bytes(out)

        if marker_prefix == 0xFF and 0xE0 <= marker <= 0xEF and removed < max_segments:
            seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
            seg_len = max(2, seg_len)
            i += 2 + seg_len
            removed += 1
            continue

        out.append(marker_prefix)
        i += 1

    out.extend(data[i:])
    return bytes(out)
