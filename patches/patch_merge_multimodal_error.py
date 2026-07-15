"""Harden _merge_multimodal_embeddings diagnostics and pre-checks.

Original code only inspected shapes inside `except RuntimeError`, after a
failed GPU index assign. That can poison the CUDA context and then raise a
generic ValueError("Error during index put operation"), which is what killed
EngineCore on the dual-Spark multimodal crash.

This patch:
  1. Compares embedding count vs placeholder count *before* the assignment
     so pure shape mismatches never touch CUDA.
  2. Replaces the generic fallback with CUDA-aware messages so operators
     (and error-isolation) can distinguish device faults from validation.

Applied at bring-up by start.sh (docker exec on both nodes).
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py"
)

MARKER = "Multimodal embedding merge failed"

src = TARGET.read_text()
changes = 0

if MARKER in src and "num_expected_tokens = int(is_multimodal.sum().item())" in src:
    print("  _merge_multimodal_embeddings: already hardened")
else:
    old = """\
    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        # If is_multimodal is on CPU this avoids a D2H sync
        inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)

            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during index put operation") from e

    return inputs_embeds"""

    new = """\
    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    # Pre-validate counts before GPU index assignment so a pure shape
    # mismatch does not poison the CUDA context (dual-Spark crash path).
    num_actual_tokens = len(mm_embeds_flat)
    num_expected_tokens = int(is_multimodal.sum().item())
    if num_actual_tokens != num_expected_tokens:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {num_actual_tokens} "
            f"multimodal tokens to {num_expected_tokens} placeholders"
        )

    try:
        # If is_multimodal is on CPU this avoids a D2H sync on the mask itself;
        # the sum() above may sync once when the mask is on GPU.
        inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
    except Exception as e:
        err_str = str(e)
        err_type = type(e).__name__
        # Preserve CUDA / accelerator detail for logs and isolation policy.
        if (
            "CUDA" in err_str
            or "cuda" in err_str
            or "AcceleratorError" in err_type
            or "CUDA" in err_type
        ):
            raise ValueError(
                f"Multimodal embedding merge failed (CUDA): {err_type}: {err_str}"
            ) from e
        raise ValueError(
            f"Multimodal embedding merge failed: {err_type}: {err_str}"
        ) from e

    return inputs_embeds"""

    if old in src:
        src = src.replace(old, new, 1)
        changes += 1
        print("  _merge_multimodal_embeddings: pre-check + CUDA-aware errors")
    else:
        # Fallback: only rewrite the generic raise if structure drifted slightly.
        fallback_old = 'raise ValueError("Error during index put operation") from e'
        if MARKER not in src and fallback_old in src:
            fallback_new = """err_str = str(e)
        err_type = type(e).__name__
        if (
            "CUDA" in err_str
            or "cuda" in err_str
            or "AcceleratorError" in err_type
            or "CUDA" in err_type
        ):
            raise ValueError(
                f"Multimodal embedding merge failed (CUDA): {err_type}: {err_str}"
            ) from e
        raise ValueError(
            f"Multimodal embedding merge failed: {err_type}: {err_str}"
        ) from e"""
            src = src.replace(fallback_old, fallback_new, 1)
            changes += 1
            print("  _merge_multimodal_embeddings: applied error-text fallback only")
        else:
            print("  FAILED: merge multimodal anchor not found", file=sys.stderr)
            sys.exit(1)

TARGET.write_text(src)
py_compile.compile(str(TARGET), doraise=True)
print(
    f"patch_merge_multimodal_error: "
    f"{'APPLIED' if changes > 0 else 'ALREADY OK'} ({changes} changes)"
)
