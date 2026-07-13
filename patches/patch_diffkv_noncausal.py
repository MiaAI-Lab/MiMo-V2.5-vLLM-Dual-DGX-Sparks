"""Thread USE_CAUSAL through the diffkv triton kernel (mirrors the stock
triton port). Target layers still run causal at runtime; this satisfies the
global non-causal selector gate under DFlash and applies the same
correctness semantics (seq_len bound, SWA handling) if ever used non-causally.

NOTE: The nvfp4-kv-diffkv mod (applied by apply-mods.sh) overwrites both
triton_unified_attention_diffkv.py and triton_attn_diffkv.py with versions
that lack USE_CAUSAL and supports_non_causal. This patch adds them back."""
from pathlib import Path
import py_compile
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm")
K = P / "v1/attention/ops/triton_unified_attention_diffkv.py"
B = P / "v1/attention/backends/triton_attn_diffkv.py"

changes = 0

# ---- 1. triton_unified_attention_diffkv.py: add USE_CAUSAL constexpr ----
src = K.read_text()
has_use_causal = "USE_CAUSAL" in src
has_assert_causal = 'assert causal, "Only causal attention is supported"' in src

if not has_use_causal:
    # Kernel signature: find def kernel_unified_attention_diffkv(
    if "def kernel_unified_attention_diffkv(" in src:
        i = src.index("def kernel_unified_attention_diffkv(")
        j = src.index("\n):", i)
        src = src[:j] + "\n    USE_CAUSAL: tl.constexpr = True,  # bool" + src[j:]
        changes += 1

    # Patch callsites that pass SLIDING_WINDOW etc.
    # IMPORTANT: compute_tile_loop_bounds accepts at most CHUNK_LOOKBACK/CHUNK_SIZE
    # (15 args). Do NOT pass USE_CAUSAL there — that belongs only on
    # compute_kv_seq_mask (stock unified attention never threads USE_CAUSAL into
    # tile-loop bounds; doing so causes Triton CompilationError / HTTP 500).
    for pair in [
        # call 1: compute_tile_loop_bounds — optional chunk args only
        ("""        SLIDING_WINDOW,
        False,  # USE_MM_PREFIX
        IS_3D,
    )""", """        SLIDING_WINDOW,
        False,  # USE_MM_PREFIX
        IS_3D,
        -1,  # CHUNK_LOOKBACK
        -1,  # CHUNK_SIZE
    )"""),
        # call 2: compute_kv_seq_mask (MAX_MM_RANGES marker)
        ("""            SLIDING_WINDOW,
            False,  # USE_MM_PREFIX
            0,  # MAX_MM_RANGES
        )""", """            SLIDING_WINDOW,
            False,  # USE_MM_PREFIX
            0,  # MAX_MM_RANGES
            -1,  # CHUNK_LOOKBACK
            -1,  # CHUNK_SIZE
            USE_CAUSAL,
            seq_len,
        )"""),
        # SWA V-zeroing
        ("""        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )""", """        if SLIDING_WINDOW and USE_CAUSAL:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )"""),
    ]:
        old, new = pair
        if old in src:
            src = src.replace(old, new, 1)
            changes += 1
        elif new in src:
            pass  # already applied
        else:
            print(f"  NOTE: anchor not found for callsite (code may have evolved)", file=sys.stderr)

    # Replace causal assert
    if has_assert_causal:
        src = src.replace(
            '    assert causal, "Only causal attention is supported"',
            '    use_causal = bool(causal)',
        )
        changes += 1

    # Add USE_CAUSAL to kernel call
    if "USE_CAUSAL=use_causal" not in src:
        src = src.replace(
            "SLIDING_WINDOW=(1 + window_size[0]),",
            "SLIDING_WINDOW=(1 + window_size[0]),\n        USE_CAUSAL=use_causal,",
        )
        changes += 1

    print(f"  diffkv kernel: USE_CAUSAL added ({changes} changes)")
else:
    print("  diffkv kernel: USE_CAUSAL already present")

# Heal prior broken patch that passed USE_CAUSAL into tile-loop bounds
# (runs whether or not the rest of the patch already applied).
broken = """        SLIDING_WINDOW,
        False,  # USE_MM_PREFIX
        IS_3D,
        -1,  # CHUNK_LOOKBACK
        -1,  # CHUNK_SIZE
        USE_CAUSAL,
    )"""
fixed = """        SLIDING_WINDOW,
        False,  # USE_MM_PREFIX
        IS_3D,
        -1,  # CHUNK_LOOKBACK
        -1,  # CHUNK_SIZE
    )"""
if broken in src:
    src = src.replace(broken, fixed, 1)
    changes += 1
    print("  healed broken compute_tile_loop_bounds(USE_CAUSAL) call", file=sys.stderr)

K.write_text(src)
# ---- 2. triton_attn_diffkv.py: add supports_non_causal + causal patching ----
bsrc = B.read_text()
bchanges = 0

# Check if supports_non_causal already exists
if "supports_non_causal" in bsrc:
    # Check if it returns True (either directly or through parent)
    print("  diffkv backend: supports_non_causal already exists")
else:
    # Add supports_non_causal after the class definition opener
    if "class TritonAttentionDiffKVBackend(TritonAttentionBackend):" in bsrc:
        bsrc = bsrc.replace(
            "class TritonAttentionDiffKVBackend(TritonAttentionBackend):",
            "class TritonAttentionDiffKVBackend(TritonAttentionBackend):\n"
            "    @classmethod\n    def supports_non_causal(cls) -> bool:\n        return True\n",
            1,
        )
        bchanges += 1
        print("  diffkv backend: added supports_non_causal")

# Also update causal=True -> causal=getattr(attn_metadata, "causal", True)
if 'causal=True' not in bsrc or 'causal=getattr' in bsrc:
    pass  # already handled or not needed
else:
    bsrc = bsrc.replace(
        '            causal=True,',
        '            causal=getattr(attn_metadata, "causal", True),',
    )
    bchanges += 1

if bchanges > 0:
    B.write_text(bsrc)
    print(f"  diffkv backend: {bchanges} change(s)")

# ---- 3. Compile both ----
for f in (K, B):
    py_compile.compile(str(f), doraise=True)

total = changes + bchanges
print(f"patch_diffkv_noncausal: {'APPLIED' if total > 0 else 'ALREADY OK'} ({total} changes)")
