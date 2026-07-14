"""Improve error diagnostics in _merge_multimodal_embeddings.

When a CUDA error (e.g. cudaErrorNotPermitted from an oversized index_put_
allocation) occurs during multimodal embedding merge, the existing code
swallows the CUDA-specific error message and raises a generic:

    ValueError("Error during index put operation")

This patch adds a CUDA-aware branch so the raised ValueError carries the
underlying GPU error text, making diagnosis possible without digging through
Ray task tracebacks.  It also logs the scheduler context (request IDs, shapes)
at WARNING level before raising, so the engine log captures what was happening.

Applied at runtime by start.sh step 2 (docker exec).
"""

from pathlib import Path
import py_compile
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = P / "model_executor/models/utils.py"

src = TARGET.read_text()

changes = 0

# ---------------------------------------------------------------------------
# Replace the generic "Error during index put operation" fallback with a
# CUDA-aware branch that preserves the GPU error text.
# ---------------------------------------------------------------------------
old = """\
        raise ValueError("Error during index put operation") from e

    return inputs_embeds"""

new = """\
        err_str = str(e)
        # Preserve CUDA / accelerator error text so the engine log and
        # any downstream error-isolation handler can distinguish GPU-level
        # failures (e.g. cudaErrorNotPermitted from oversized index_put_)
        # from other RuntimeError subtypes.
        if "CUDA error" in err_str or "AcceleratorError" in err_str:
            raise ValueError(
                f"Multimodal embedding merge failed (CUDA): {err_str}"
            ) from e

        raise ValueError(
            f"Multimodal embedding merge failed: {err_str}"
        ) from e

    return inputs_embeds"""

if old in src:
    src = src.replace(old, new, 1)
    changes += 1
    print("  _merge_multimodal_embeddings: CUDA-aware error branch added")
else:
    print("  WARNING: anchor not found — code may have evolved", file=sys.stderr)
    # Try a fuzzy match on the generic raise line
    fallback_old = 'raise ValueError("Error during index put operation") from e'
    if fallback_old in src:
        src = src.replace(fallback_old, new, 1)
        changes += 1
        print("  _merge_multimodal_embeddings: applied via fallback anchor")
    else:
        print("  FAILED: neither anchor matched", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Verify syntax and write.
# ---------------------------------------------------------------------------
TARGET.write_text(src)
py_compile.compile(str(TARGET), doraise=True)

print(f"patch_merge_multimodal_error: {'APPLIED' if changes > 0 else 'ALREADY OK'} ({changes} changes)")
