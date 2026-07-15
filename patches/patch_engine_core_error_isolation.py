"""Isolate *recoverable* per-request errors so they do not kill EngineCore.

Context (dual-Spark multimodal crash): a worker raised during multimodal
embedding merge; the exception escaped EngineCore.step() and the whole
API server died.

Policy (see docs/grok-fix.md):
  - CUDA / Accelerator / NCCL / OOM in the exception chain → RE-RAISE (fatal).
    A poisoned device cannot safely serve more requests.
  - Pure multimodal *validation* ValueErrors (placeholder/embedding count,
    non-CUDA merge failures) → finish affected reqs with ERROR, notify
    clients when EngineCoreProc helpers exist, return empty step outputs.

Wraps both future.result() and the sample_tokens fallback in step(), and
the corresponding block in step_with_batch_queue().

Applied at bring-up by start.sh (docker exec on both nodes).
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py")

MARKER = "_mimo_is_fatal_device_error"

src = TARGET.read_text()
changes = 0

# ---------------------------------------------------------------------------
# 0. Inject small helpers once (module level, after logger = ...).
# ---------------------------------------------------------------------------
helper_block = '''
def _mimo_is_fatal_device_error(err: BaseException) -> bool:
    """True if err or its cause/context chain looks like a device/runtime fatal."""
    seen: set[int] = set()
    cur: BaseException | None = err
    keys = (
        "CUDA",
        "cudaError",
        "AcceleratorError",
        "NCCL",
        "OutOfMemory",
        "out of memory",
        "CUBLAS",
        "cuDNN",
    )
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = f"{type(cur).__name__}: {cur}"
        if any(k in text for k in keys):
            return True
        cur = cur.__cause__ or cur.__context__  # type: ignore[assignment]
    return False


def _mimo_is_recoverable_request_error(err: BaseException) -> bool:
    """Validation-style multimodal failures that should not kill the engine."""
    if _mimo_is_fatal_device_error(err):
        return False
    text = str(err)
    needles = (
        "multimodal tokens",
        "Multimodal embedding merge failed",
        "Error during index put operation",
        "placeholders",
    )
    return any(n in text for n in needles)


def _mimo_abort_batch_on_request_error(engine, scheduler_output, err: BaseException) -> None:
    """Finish scheduled requests with ERROR and notify API clients if possible."""
    logger.warning(
        "Per-request error during model execution; aborting batch (engine stays up): %s",
        err,
    )
    req_ids: list[str] = []
    for _req in scheduler_output.scheduled_new_reqs:
        req_ids.append(_req.req_id)
    cached = scheduler_output.scheduled_cached_reqs
    if cached is not None and getattr(cached, "req_ids", None):
        req_ids.extend(cached.req_ids)
    # de-dupe, preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for rid in req_ids:
        if rid not in seen:
            seen.add(rid)
            uniq.append(rid)
    if not uniq:
        return
    from vllm.v1.request import RequestStatus

    finished = engine.scheduler.finish_requests(uniq, RequestStatus.FINISHED_ERROR)
    # EngineCoreProc exposes client notification helpers; plain EngineCore does not.
    send_err = getattr(engine, "_send_error_outputs_to_client", None)
    send_abort_batch = getattr(engine, "_send_abort_outputs", None)
    if send_err is not None and finished:
        by_client: dict[int, list[str]] = {}
        for req_id, client_index in finished:
            by_client.setdefault(client_index, []).append(req_id)
        for client_index, ids in by_client.items():
            send_err(ids, client_index)
    elif send_abort_batch is not None and finished:
        send_abort_batch(finished)

'''

if MARKER in src:
    print("  EngineCore: isolation helpers already present")
else:
    # Insert after `logger = init_logger(__name__)` if present, else before first class.
    anchor = "logger = init_logger(__name__)\n"
    if anchor in src:
        # helpers need BaseException in typing scope — always available builtin
        src = src.replace(anchor, anchor + "\n" + helper_block, 1)
        changes += 1
        print("  EngineCore: isolation helpers injected")
    else:
        print("  FAILED: logger anchor not found", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Wrap step() execution block.
# ---------------------------------------------------------------------------
old_step = """\
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def post_step(self, model_executed: bool) -> None:"""

new_step = """\
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            try:
                model_output = future.result()
                if model_output is None:
                    model_output = self.model_executor.sample_tokens(grammar_output)
            except Exception as _step_err:
                if _mimo_is_recoverable_request_error(_step_err):
                    _mimo_abort_batch_on_request_error(
                        self, scheduler_output, _step_err
                    )
                    return {}, False
                raise

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def post_step(self, model_executed: bool) -> None:"""

if "_mimo_is_recoverable_request_error(_step_err)" in src and "def post_step" in src:
    # Detect if step() already wrapped (marker inside step body near sample_tokens)
    step_wrapped = (
        "model_output = future.result()\n"
        "                if model_output is None:\n"
        "                    model_output = self.model_executor.sample_tokens"
    ) in src or (
        "try:\n"
        "                model_output = future.result()\n"
        "                if model_output is None:\n"
        "                    model_output = self.model_executor.sample_tokens"
    ) in src
    if "except Exception as _step_err:" in src:
        print("  EngineCore.step(): isolation already present")
    elif old_step in src:
        src = src.replace(old_step, new_step, 1)
        changes += 1
        print("  EngineCore.step(): recoverable-error isolation added")
    else:
        print("  FAILED: step() anchor not found", file=sys.stderr)
        sys.exit(1)
else:
    if old_step in src:
        src = src.replace(old_step, new_step, 1)
        changes += 1
        print("  EngineCore.step(): recoverable-error isolation added")
    else:
        print("  FAILED: step() anchor not found", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Wrap step_with_batch_queue() result block (second future.result site).
# ---------------------------------------------------------------------------
old_bq = """\
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )"""

new_bq = """\
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            try:
                model_output = future.result()
                if model_output is None:
                    # None from sample_tokens() implies that the original execute_model()
                    # call failed - raise that exception.
                    exec_model_fut.result()
                    raise RuntimeError("unexpected error")
            except Exception as _step_err:
                if _mimo_is_recoverable_request_error(_step_err):
                    _mimo_abort_batch_on_request_error(
                        self, scheduler_output, _step_err
                    )
                    return {}, False
                raise

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )"""

if "batch_queue.pop()" in src and "except Exception as _step_err:" in src:
    # Count how many isolation except blocks we have; step + batch_queue => 2
    if src.count("except Exception as _step_err:") >= 2:
        print("  EngineCore.step_with_batch_queue(): isolation already present")
    elif old_bq in src:
        src = src.replace(old_bq, new_bq, 1)
        changes += 1
        print("  EngineCore.step_with_batch_queue(): isolation added")
    else:
        print(
            "  WARNING: batch_queue anchor not found (async path may be unused)",
            file=sys.stderr,
        )
else:
    if old_bq in src:
        src = src.replace(old_bq, new_bq, 1)
        changes += 1
        print("  EngineCore.step_with_batch_queue(): isolation added")
    else:
        print(
            "  WARNING: batch_queue anchor not found (async path may be unused)",
            file=sys.stderr,
        )

TARGET.write_text(src)
py_compile.compile(str(TARGET), doraise=True)
print(
    f"patch_engine_core_error_isolation: "
    f"{'APPLIED' if changes > 0 else 'ALREADY OK'} ({changes} changes)"
)
