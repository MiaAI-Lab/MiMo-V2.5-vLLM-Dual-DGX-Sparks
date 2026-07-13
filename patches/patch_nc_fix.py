"""Fix non-causal correctness: seq_len bound + SWA V-zeroing skip."""
from pathlib import Path
import py_compile

P = Path("/usr/local/lib/python3.12/dist-packages/vllm")
H = P / "v1/attention/ops/triton_attention_helpers.py"
K = P / "v1/attention/ops/triton_unified_attention.py"

def edit(path, old, new):
    src = path.read_text()
    if new in src and old not in src:
        print(f"  already patched: {path.name}")
        return True
    if old not in src:
        print(f"  anchor not found: {path.name}: {old[:60]!r}")
        return False
    path.write_text(src.replace(old, new, 1))
    return True

# 1) Add seq_len param to compute_kv_seq_mask (use MAX_MM_RANGES to disambiguate)
edit(H, """    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    USE_CAUSAL: tl.constexpr = True,
):""", """    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    USE_CAUSAL: tl.constexpr = True,
    seq_len=0,
):""")

# 2) Non-causal mask: bound by seq_len
edit(H, """    if USE_CAUSAL:
        seq_mask = seq_offset[None, :] <= query_abs_pos
    else:
        # Non-causal: all keys visible (tile_mask already bounds seq_len).
        seq_mask = (seq_offset[None, :] >= 0) & (query_abs_pos >= 0)""",
"""    if USE_CAUSAL:
        seq_mask = seq_offset[None, :] <= query_abs_pos
    else:
        # Non-causal: bound by seq_len to exclude tile-overhang.
        seq_mask = (seq_offset[None, :] < seq_len) & (query_abs_pos >= 0)""")

# 3) Pass seq_len to compute_kv_seq_mask in kernel
edit(K, """            CHUNK_LOOKBACK,
            CHUNK_SIZE,
            USE_CAUSAL,
        )""", """            CHUNK_LOOKBACK,
            CHUNK_SIZE,
            USE_CAUSAL,
            seq_len,
        )""")

# 4) SWA V-zeroing: only for causal
edit(K, """        if SLIDING_WINDOW:
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
            )""")

for f in (H, K):
    py_compile.compile(str(f), doraise=True)
print("patch_nc_fix: DONE")
