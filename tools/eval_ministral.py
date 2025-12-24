#!/usr/bin/env python3
"""Ministral-3-8B-Instruct evaluation launcher.

Defaults align with ByteCaption_XE_ministral. Pass any extra args to override.

Example:
  python tools/eval_ministral.py --test_samples 500 --corrupt_types rbbf --corrupt_level S0
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SCRIPT = PROJECT_ROOT / "PureT" / "main_test.py"


def _flag_present(argv: List[str], flag: str) -> bool:
    return flag in argv


def _append_default(cmd: List[str], argv: List[str], flag: str, value) -> None:
    if _flag_present(argv, flag):
        return
    cmd.append(flag)
    cmd.append(str(value))


def _build_command(argv: List[str]) -> List[str]:
    cmd = [sys.executable, str(TEST_SCRIPT)]
    _append_default(cmd, argv, "--folder", "PureT/experiments/ByteCaption_XE_ministral")
    if "--disable_wandb" not in argv:
        cmd.append("--disable_wandb")
    cmd.extend(argv)
    return cmd


def main() -> None:
    argv = sys.argv[1:]
    cmd = _build_command(argv)
    print("[Ministral] Eval:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
