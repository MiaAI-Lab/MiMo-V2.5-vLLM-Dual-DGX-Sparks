"""Pin the DFlash drafter's KV cache dtype to auto. The drafter runs on the
stock triton backend (non-causal+sinks) which cannot consume the diffkv
NVFP4 KV layout; its KV is tiny anyway (SWA-1024-bounded)."""
from pathlib import Path
import py_compile

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py")
s = p.read_text()
old = """        base = replace(
            base,
            attention_config=replace(
                base.attention_config,
                backend=spec_cfg.attention_backend,
            ),
        )

        return base"""
new = """        base = replace(
            base,
            attention_config=replace(
                base.attention_config,
                backend=spec_cfg.attention_backend,
            ),
        )

        # The draft model's attention backend is selected independently and
        # may not support the target's quantized KV layout (e.g. diffkv
        # nvfp4). Keep the drafter's KV cache in native dtype; its KV is
        # small (SWA-bounded) so the memory cost is negligible.
        if base.cache_config is not None and base.cache_config.cache_dtype != "auto":
            base = replace(
                base,
                cache_config=replace(base.cache_config, cache_dtype="auto"),
            )

        return base"""
if new in s:
    print("already patched")
else:
    assert old in s, "anchor not found"
    p.write_text(s.replace(old, new, 1))
    py_compile.compile(str(p), doraise=True)
    print("patch_draft_cache_auto: APPLIED + COMPILED")
