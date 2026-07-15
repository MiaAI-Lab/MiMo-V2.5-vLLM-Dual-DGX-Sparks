# Grok fix: MiMo V2.5 dual-Spark multimodal engine crash

**Date:** 2026-07-15  
**Scope:** `two_sparks_mimo_2_5_bugreport` + failed attempt in `docs/CRASH-FIX.md`  
**Status:** Implemented (boot-safe patches; no invalid CLI flags)

---

## 1. Problem

A single OpenAI-compatible multimodal (image + text) chat request caused:

1. Worker rank 1 to fail in `_merge_multimodal_embeddings` with  
   `torch.AcceleratorError: CUDA error: operation not permitted`
2. That to be re-raised as `ValueError: Error during index put operation`
3. Ray → EngineCore fatal shutdown → API server exit
4. Docker containers still `Up`, but port dead; ~12 min cold recovery

Request characteristics (from bug report):

- 1 image + text, ~922 prompt tokens
- `max_tokens: 999078` (nearly full 1M context; no client cap)
- TP=2, Ray, NVFP4 KV, MTP1, Omni model

---

## 2. Why the previous fix failed

See also analysis of `docs/CRASH-FIX.md` and git history (`95b696c` → reverts).

| Previous approach | Failure mode |
|---|---|
| `--max-tokens 32768` on `launch-omni.sh` | **Not a valid vLLM 0.21 CLI flag** → serve never starts (argparse) |
| EngineCore `try/except` around only `future.result()` | Incomplete path (crash also hits `sample_tokens` / Ray DAG); tries to “survive” CUDA poison |
| String match only on `"Error during index put operation"` | After diagnostic rewrite, messages change; CUDA cases must stay fatal |
| Attributed root cause purely to `max_tokens` | Multimodal merge sizes come from **prompt** placeholders, not generation length. Extreme `max_tokens` is still harmful (KV reservation) but is not a proven direct cause of `index_put` CUDA failure |

`RuntimeError: Engine core initialization failed. … Failed core proc(s): {}` is a **handshake wrapper** from `wait_for_engine_startup`. The real cause is always the log lines above it. Empty `{}` means the child died before an exit code was collected (often rapid restart / OOM / Ray race) — not proof that code clamps are unloadable.

---

## 3. Plan (what we implement)

### Goals

1. **Boot-safe** — no unknown CLI args; patches must match anchors + `py_compile`
2. **Server-side `max_tokens` cap** — enforce recipe 32 768 ceiling even if clients misbehave
3. **Safer multimodal merge** — validate token counts *before* GPU assign when possible; preserve CUDA text in errors
4. **Limited request isolation** — only for **non-device** validation failures; **re-raise** CUDA / NCCL / OOM so we never pretend a poisoned GPU is fine
5. **Document** this file; leave multimodal CUDA root-cause as follow-up (needs image repro)

### Non-goals

- Fixing the underlying dual-Spark / TP image-merge CUDA bug without a minimal repro
- Silent recovery after CUDA device errors
- Adding `--max-tokens` to the launcher

### Patch set

| Patch | File | Role |
|---|---|---|
| A | `patches/patch_max_tokens_clamp.py` | Clamp `sampling_params.max_tokens` after defaults/gen-config to `MAX_TOKENS_LIMIT` (default 32768) |
| B | `patches/patch_merge_multimodal_error.py` | Pre-check embedding vs placeholder counts; CUDA-aware error text |
| C | `patches/patch_engine_core_error_isolation.py` | On recoverable multimodal **validation** errors only: finish requests with ERROR, notify clients, continue engine |

### Wiring

- Append A/B/C to `ALL_PATCHES` in `start.sh` (applied on **both** nodes)
- Export `MAX_TOKENS_LIMIT="${MAX_TOKENS_LIMIT:-32768}"` in `launch_vllm` container env
- **Do not** touch `recipe/launch-omni.sh` CLI flags for max tokens

### Verification

1. Dry-run apply all three patches inside the stopped/started container → expect `APPLIED` / `ALREADY OK`, `py_compile` clean
2. On next `bash start.sh`: boot to `Application startup complete`, `/v1/models` OK
3. Text chat with `max_tokens: 999078` → server should clamp (log warning) and not die
4. Multimodal with modest `max_tokens` still needs separate validation when ready to risk a reload

#### Dry-run results (2026-07-15, container `mimo-nvfp4`)

| Patch | First apply | Re-apply | `py_compile` |
|---|---|---|---|
| `patch_max_tokens_clamp` | APPLIED (2) | ALREADY OK | OK |
| `patch_merge_multimodal_error` | APPLIED (1) | ALREADY OK | OK |
| `patch_engine_core_error_isolation` | APPLIED (3) | ALREADY OK | OK |

Container left **stopped** after dry-run (no full Omni bring-up in this change). Full stack verification is on the next `bash start.sh`.

---

## 4. Implementation notes

### Fix A — clamp (input processor)

Hook after `max_tokens` default + `update_from_generation_config` / tokenizer updates so the limit always wins:

```text
MAX_TOKENS_LIMIT env (default 32768)
if max_tokens > limit → warning log + clamp
```

No CLI flag. Matches recipe docs and agent `maxTokens: 32768`.

### Fix B — multimodal merge

Replace the try/except-only path with:

1. Flatten embeddings
2. Compare `len(mm_embeds_flat)` vs `is_multimodal.sum()` **before** assignment when practical  
   → pure shape bugs become `ValueError` without touching CUDA
3. On remaining failures, raise  
   `Multimodal embedding merge failed (CUDA): …` or  
   `Multimodal embedding merge failed: …`  
   so operators and Fix C can classify

### Fix C — isolation (recoverable only)

In `EngineCore.step()` (and the same pattern in `step_with_batch_queue`):

```text
try:
  model_output = future.result()
  if model_output is None:
    model_output = sample_tokens(...)
except Exception as e:
  if fatal_device_error(e):   # CUDA / Accelerator / NCCL / OOM in chain
    raise
  if recoverable_request_error(e):  # multimodal validation messages
    finish_requests(..., FINISHED_ERROR)
    notify clients via _send_error_outputs_to_client when available
    return {}, False
  raise
```

`fatal_device_error` walks `__cause__` / `__context__` so a generic ValueError wrapping a CUDA error stays fatal.

### Follow-ups (not in this change)

- Reproduce image merge with `max_tokens ≤ 1024` to isolate TP/mm-encoder issues
- Inspect rank-1 placeholder vs embedding counts under `--mm-encoder-tp-mode data`
- Consider OpenAI API HTTP 400 for oversize `max_tokens` (explicit reject vs silent clamp)

---

## 5. What was changed in-repo

| Path | Change |
|---|---|
| `docs/grok-fix.md` | This plan + implementation record |
| `patches/patch_max_tokens_clamp.py` | **New** |
| `patches/patch_merge_multimodal_error.py` | **New** |
| `patches/patch_engine_core_error_isolation.py` | **New** |
| `start.sh` | Register patches; export `MAX_TOKENS_LIMIT` |

Patches apply at bring-up via existing `apply_mods_patches` (docker cp + `python3 /tmp/…` on head and worker).

---

## 6. How to deploy

```bash
cd /mnt/models/MiMo
bash stop.sh          # if anything still running
bash start.sh         # containers → mods/patches → Ray → Omni → chat verify
```

Optional override:

```bash
export MAX_TOKENS_LIMIT=32768   # or lower for stricter agents
bash start.sh
```

Confirm patches in container logs during step “Mods + engine patches”:

```text
patch_max_tokens_clamp: APPLIED …
patch_merge_multimodal_error: APPLIED …
patch_engine_core_error_isolation: APPLIED …
```

---

## 7. Risk assessment

| Risk | Mitigation |
|---|---|
| Anchor mismatch on image update | Patches fail soft (start.sh already tolerates assertion failures); end-state verify still runs for NVFP4 |
| Isolation hides real bugs | Only non-device multimodal validation strings; device errors still kill engine |
| Clamp too aggressive | Env override `MAX_TOKENS_LIMIT`; default matches documented client ceiling |
| Residual half-applied files after failed experiments | `stop.sh` + recreate containers from GHCR tag; patches re-apply idempotently |

---

## 8. Relation to original bug-report questions

1. **Is image input stable on dual-Spark?** — Smoke suite below passed image+text (tiny PNG and hero JPG) including extreme `max_tokens`; not a full stress test of every image size/format.
2. **Can API enforce 32 768 max output?** — **Yes**, via Fix A (`MAX_TOKENS_LIMIT`), not via `--max-tokens` CLI.
3. **Can multimodal failure be isolated?** — **Partially**: shape/validation errors can abort the request; true CUDA faults remain engine-fatal by design. Isolation patch is intentionally not applied at bring-up.

## 9. Crash regression smoke (reusable)

Script: [`scripts/crash_regression_smoke.sh`](../scripts/crash_regression_smoke.sh)

Recreates the bug-report pressure points against any OpenAI-compatible endpoint:

| Case | What it checks |
|---|---|
| health | `GET /v1/models` |
| text safe | short completion |
| text extreme | `max_tokens=999078` must not kill the server |
| mm safe | image + text, modest `max_tokens` |
| mm extreme | **bug-report combo**: image + extreme `max_tokens` (opt-in) |
| final health | API still up |

```bash
# Default MiMo dual-Spark endpoint
bash scripts/crash_regression_smoke.sh

# Other model / host
BASE_URL=http://127.0.0.1:8000/v1 MODEL=my-model \
  bash scripts/crash_regression_smoke.sh --image /path/to.png

# Full bug-report combo (image + extreme max_tokens)
bash scripts/crash_regression_smoke.sh --extreme-mm --image tests/fixtures/tiny_smoke.png

# Text only
bash scripts/crash_regression_smoke.sh --skip-mm
```

**Pass criterion for crash class:** the server still answers `/v1/models` after each case (request 4xx/timeout without death is OK; EngineCore death is FAIL).

### Live results (2026-07-15, this cluster)

| Suite | Result |
|---|---|
| safe (tiny PNG, no extreme-mm) | **5 pass, 1 skip** |
| extreme-mm (tiny PNG) | **6/6 pass** — mm extreme http=200, server alive |
| extreme-mm (`assets/mimo.jpg`) | **6/6 pass** — mm extreme http=200, server alive |

Tiny fixture: `tests/fixtures/tiny_smoke.png` (64×64).
