"""Drive the real patch_max_tokens_clamp.py against real vLLM input_processor source.

Proves:
  1. The shipped patch script applies cleanly to the image's input_processor.py.
  2. The resulting clamp block (extracted from the patched file, not reimplemented)
     caps max_tokens to MAX_TOKENS_LIMIT.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "patches" / "patch_max_tokens_clamp.py"
# Real source extracted from the runtime image (checked in via test bootstrap, or
# fall back to regenerating from docker when available).
FIXTURE_CANDIDATES = [
    Path(os.environ.get("CLAMP_FIXTURE", "")),
    Path("/tmp/grok-goal-57924a981653/implementer/input_processor_orig.py"),
    REPO / "tests" / "fixtures" / "input_processor_orig.py",
]


def _fixture_source() -> Path:
    for p in FIXTURE_CANDIDATES:
        if p and p.is_file() and p.stat().st_size > 1000:
            return p
    # Last resort: pull from local docker image
    out = Path(tempfile.mkdtemp()) / "input_processor_orig.py"
    img = os.environ.get(
        "MIMO_RUNTIME_IMAGE",
        "ghcr.io/miaai-lab/mimo-v2.5-vllm-dual-dgx-sparks:20260704",
    )
    subprocess.check_call(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "cat",
            img,
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/input_processor.py",
        ],
        stdout=out.open("w"),
    )
    return out


def _apply_patch(src_file: Path) -> Path:
    """Run the real shipped patch against a temp copy; return patched path."""
    assert PATCH.is_file(), f"missing shipped patch: {PATCH}"
    work = Path(tempfile.mkdtemp(prefix="clamp_patch_"))
    target = work / "input_processor.py"
    shutil.copy(src_file, target)
    env = os.environ.copy()
    env["PATCH_TARGET"] = str(target)
    proc = subprocess.run(
        [sys.executable, str(PATCH)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"patch failed rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "MAX_TOKENS_LIMIT" in target.read_text()
    assert "sampling_params.max_tokens = _max_tokens_limit" in target.read_text()
    return target


def _extract_clamp_block(patched_text: str) -> str:
    """Pull the exact clamp assignment block from the patched source."""
    # Executable lines only (consistent indent) — skip the comment prologue.
    m = re.search(
        r"(?m)^([ \t]*)(_max_tokens_limit = int\(os\.environ\.get\("
        r"\"MAX_TOKENS_LIMIT\".*?"
        r"sampling_params\.max_tokens = _max_tokens_limit)",
        patched_text,
        flags=re.S,
    )
    assert m, "clamp block not found in patched source"
    return m.group(1) + m.group(2)


class _SamplingParams:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens


def _run_clamp_block(block: str, max_tokens, limit: str = "32768"):
    """Execute the exact extracted clamp block against a mock sampling_params."""
    sp = _SamplingParams(max_tokens)
    # Indentation in source is nested; dedent by stripping common leading spaces.
    lines = block.splitlines()
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    base = min(indents) if indents else 0
    dedented = "\n".join(l[base:] if len(l) >= base else l for l in lines)
    # logger.warning is used in the block — provide a no-op logger.
    g = {
        "os": os,
        "sampling_params": sp,
        "logger": type("L", (), {"warning": staticmethod(lambda *a, **k: None)})(),
    }
    old = os.environ.get("MAX_TOKENS_LIMIT")
    os.environ["MAX_TOKENS_LIMIT"] = limit
    try:
        exec(dedented, g, g)
    finally:
        if old is None:
            os.environ.pop("MAX_TOKENS_LIMIT", None)
        else:
            os.environ["MAX_TOKENS_LIMIT"] = old
    return sp.max_tokens


def test_patch_applies_and_clamp_caps_extreme_max_tokens():
    fixture = _fixture_source()
    patched = _apply_patch(fixture)
    text = patched.read_text()
    block = _extract_clamp_block(text)

    # Extreme client value (the dual-Spark bug report) must be clamped.
    assert _run_clamp_block(block, 999078, "32768") == 32768
    # Values at/under the limit are unchanged.
    assert _run_clamp_block(block, 32768, "32768") == 32768
    assert _run_clamp_block(block, 64, "32768") == 64
    # Custom env limit is honored by the shipped block.
    assert _run_clamp_block(block, 100000, "1000") == 1000
    # None is left alone (generation path may still set defaults earlier).
    assert _run_clamp_block(block, None, "32768") is None


def test_patch_is_idempotent():
    fixture = _fixture_source()
    patched = _apply_patch(fixture)
    # Second apply must not fail (ALREADY OK path).
    env = os.environ.copy()
    env["PATCH_TARGET"] = str(patched)
    proc = subprocess.run(
        [sys.executable, str(PATCH)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ALREADY OK" in proc.stdout or "already present" in proc.stdout


if __name__ == "__main__":
    # Allow `python tests/test_max_tokens_clamp_patch.py` without pytest.
    test_patch_applies_and_clamp_caps_extreme_max_tokens()
    test_patch_is_idempotent()
    print("OK: clamp patch unit tests passed")
