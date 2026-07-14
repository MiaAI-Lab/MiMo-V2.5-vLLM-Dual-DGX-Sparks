"""Isolate per-request execution errors so they don't crash EngineCore.

When a multimodal embedding merge or other per-request preprocessing step
fails on a worker (e.g. CUDA error from an oversized index_put_ allocation),
the exception propagates as RayTaskError -> step() -> _process_engine_step()
-> run_busy_loop() -> run_engine_core(), where the generic `except Exception`
calls _send_engine_dead() and terminates the API server.

This patch wraps `future.result()` inside `step()` with a try/except that
recognises known per-request error patterns, aborts the affected batch via
the existing scheduler abort_requests() path, and returns empty outputs so
the engine continues serving remaining/queued requests.

Only errors that originate from per-request preprocessing (multimodal
embedding merge, input validation) are caught.  Genuine engine-level failures
(NCCL faults, CUDA device lost, OOM) still propagate and trigger the normal
fatal-shutdown path.

Applied at runtime by start.sh step 2 (docker exec).
"""

from pathlib import Path
import py_compile
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = P / "v1/engine/core.py"

src = TARGET.read_text()

changes = 0

# ---------------------------------------------------------------------------
# Replace the bare `model_output = future.result()` inside the `with` block
# with a try/except that catches per-request errors and aborts the batch.
# ---------------------------------------------------------------------------
old = """\
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)"""

new = """\
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            try:
                model_output = future.result()
            except Exception as _step_err:
                _err_str = str(_step_err)
                # Recognise per-request preprocessing errors (multimodal
                # embedding merge, input validation, etc.) and abort the
                # affected batch instead of crashing the entire engine.
                # Genuine engine-level failures (NCCL, CUDA device lost,
                # OOM) still propagate to the fatal-shutdown path.
                if ("Error during index put operation" in _err_str
                        or "Multimodal embedding merge failed" in _err_str
                        or "multimodal tokens" in _err_str
                        or "CUDA error: operation not permitted" in _err_str):
                    logger.warning(
                        "Per-request error during model execution, "
                        "aborting batch: %s", _step_err)
                    for _req in scheduler_output.scheduled_new_reqs:
                        self.abort_requests([_req.req_id])
                    if scheduler_output.scheduled_cached_reqs.req_ids:
                        self.abort_requests(
                            scheduler_output.scheduled_cached_reqs.req_ids)
                    return {}, False
                raise
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)"""

if old in src:
    src = src.replace(old, new, 1)
    changes += 1
    print("  EngineCore.step(): per-request error isolation added")
else:
    print("  WARNING: primary anchor not found — trying fallback", file=sys.stderr)
    # Fallback: try matching just the inner assignment line
    fallback_old = """\
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)"""
    if fallback_old in src:
        src = src.replace(fallback_old, new, 1)
        changes += 1
        print("  EngineCore.step(): applied via fallback anchor")
    else:
        print("  FAILED: neither anchor matched", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Verify syntax and write.
# ---------------------------------------------------------------------------
TARGET.write_text(src)
py_compile.compile(str(TARGET), doraise=True)

print(f"patch_engine_core_error_isolation: {'APPLIED' if changes > 0 else 'ALREADY OK'} ({changes} changes)")
