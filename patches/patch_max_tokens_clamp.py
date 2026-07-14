"""Clamp per-request max_tokens to a server-side limit.

vLLM does not have a built-in --max-tokens CLI argument (this build is
0.21.1rc1.dev85).  When a client sends an extreme max_tokens value such
as 999078 (the entire 1M context window), the value passes validation
and reaches the GPU where it causes cudaErrorNotPermitted during
multimodal embedding index_put_.

This patch adds a clamp in the input processor (after the default max_tokens
computation) that caps sampling_params.max_tokens to the value of the
MAX_TOKENS_LIMIT env var (default 32768, matching the recipe's documented
client ceiling).

Applied at runtime by start.sh step 2 (docker exec).
"""

from pathlib import Path
import py_compile
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm")
TARGET = P / "v1/engine/input_processor.py"

src = TARGET.read_text()

changes = 0

# ---------------------------------------------------------------------------
# 1. Add `import os` after the existing `import time` line (if missing).
# ---------------------------------------------------------------------------
if "import os\n" not in src:
    old_import = "import time\n"
    new_import = "import os\nimport time\n"
    if old_import in src:
        src = src.replace(old_import, new_import, 1)
        changes += 1
        print("  input_processor: added import os")
    else:
        print("  WARNING: 'import time' anchor not found for import", file=sys.stderr)
else:
    print("  input_processor: import os already present")

# ---------------------------------------------------------------------------
# 2. Replace the max_tokens default block with one that also clamps.
# ---------------------------------------------------------------------------
old = """\
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len"""

new = """\
            # If unset max tokens, then generate up to the max_model_len.
            # Then clamp to the server-side limit (MAX_TOKENS_LIMIT env, default
            # 32768) to prevent extreme values (e.g. max_tokens=999078) from
            # causing GPU-level crashes like cudaErrorNotPermitted during
            # multimodal embedding merge or other per-request processing.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len
            _max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "32768"))
            if (sampling_params.max_tokens is not None
                    and sampling_params.max_tokens > _max_tokens_limit):
                sampling_params.max_tokens = _max_tokens_limit"""

if old in src:
    src = src.replace(old, new, 1)
    changes += 1
    print("  input_processor: max_tokens clamp added")
else:
    print("  WARNING: max_tokens anchor not found — trying fallback", file=sys.stderr)
    # Fallback: match just the if-block body
    fallback = """\
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len"""
    if fallback in src:
        src = src.replace(fallback, new, 1)
        changes += 1
        print("  input_processor: max_tokens clamp added via fallback")
    else:
        print("  FAILED: neither anchor matched", file=sys.stderr)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Verify syntax and write.
# ---------------------------------------------------------------------------
TARGET.write_text(src)
py_compile.compile(str(TARGET), doraise=True)

print(f"patch_max_tokens_clamp: {'APPLIED' if changes > 0 else 'ALREADY OK'} ({changes} changes)")
