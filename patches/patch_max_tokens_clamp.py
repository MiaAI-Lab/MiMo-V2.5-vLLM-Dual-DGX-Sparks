"""Clamp per-request max_tokens to a server-side limit.

vLLM 0.21.1rc1.dev85 has no --max-tokens CLI argument. Clients can send
extreme values (e.g. max_tokens=999078 against a 1M context) that pass
validation and reserve nearly the entire KV pool.

This patch clamps sampling_params.max_tokens to MAX_TOKENS_LIMIT (env,
default 32768 — the recipe's documented client ceiling) after the default
max_tokens assignment and generation-config / tokenizer updates.

Applied at bring-up by start.sh (docker exec on both nodes).
"""
from __future__ import annotations

import os
import py_compile
import sys
from pathlib import Path

# Allow tests to redirect the target file via PATCH_TARGET; production uses default.
TARGET = Path(
    os.environ.get(
        "PATCH_TARGET",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/input_processor.py",
    )
)

MARKER = "MAX_TOKENS_LIMIT"  # idempotency marker in the clamped block

src = TARGET.read_text()
changes = 0

# ---------------------------------------------------------------------------
# 1. Ensure `import os` (stdlib) near the top imports.
# ---------------------------------------------------------------------------
if "import os\n" not in src:
    if "import time\n" in src:
        src = src.replace("import time\n", "import os\nimport time\n", 1)
        changes += 1
        print("  input_processor: added import os")
    else:
        print("  FAILED: no 'import time' anchor for import os", file=sys.stderr)
        sys.exit(1)
else:
    print("  input_processor: import os already present")

# ---------------------------------------------------------------------------
# 2. Clamp after default max_tokens + gen-config / tokenizer updates.
# ---------------------------------------------------------------------------
if MARKER in src and "sampling_params.max_tokens = _max_tokens_limit" in src:
    print("  input_processor: max_tokens clamp already present")
else:
    old = """\
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len

            sampling_params.update_from_generation_config(
                self.generation_config_fields,
                self.renderer.get_eos_token_id(),
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)"""

    new = """\
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len

            sampling_params.update_from_generation_config(
                self.generation_config_fields,
                self.renderer.get_eos_token_id(),
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)

            # Server-side output cap (MAX_TOKENS_LIMIT env, default 32768).
            # Prevents extreme client values (e.g. max_tokens ≈ max_model_len)
            # from reserving the full KV pool. See docs/grok-fix.md.
            _max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "32768"))
            if (
                sampling_params.max_tokens is not None
                and sampling_params.max_tokens > _max_tokens_limit
            ):
                logger.warning(
                    "Clamping max_tokens from %s to MAX_TOKENS_LIMIT=%s",
                    sampling_params.max_tokens,
                    _max_tokens_limit,
                )
                sampling_params.max_tokens = _max_tokens_limit"""

    if old in src:
        src = src.replace(old, new, 1)
        changes += 1
        print("  input_processor: max_tokens clamp added")
    else:
        print("  FAILED: max_tokens block anchor not found", file=sys.stderr)
        sys.exit(1)

TARGET.write_text(src)
py_compile.compile(str(TARGET), doraise=True)
print(
    f"patch_max_tokens_clamp: "
    f"{'APPLIED' if changes > 0 else 'ALREADY OK'} ({changes} changes)"
)
