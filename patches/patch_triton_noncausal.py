"""Add non-causal attention support to triton_attn backend for DFlash drafter."""
from pathlib import Path
import py_compile

P = Path("/usr/local/lib/python3.12/dist-packages/vllm")
H = P / "v1/attention/ops/triton_attention_helpers.py"
K = P / "v1/attention/ops/triton_unified_attention.py"
B = P / "v1/attention/backends/triton_attn.py"

def edit(path, old, new, count=1):
    src = path.read_text()
    if new in src and old not in src:
        print(f"  already patched: {path.name}")
        return True
    if old not in src:
        print(f"  anchor not found: {path.name}: {old[:60]!r}")
        return False
    path.write_text(src.replace(old, new, count))
    return True

# ===== 1) compute_kv_seq_mask: add USE_CAUSAL (+seq_len) + conditional mask =====
# NOTE: CHUNK_LOOKBACK/CHUNK_SIZE appear on multiple helpers — anchor on the
# function name or the patch lands on the wrong signature (breaks Triton compile).
edit(H, """def compute_kv_seq_mask(
    query_abs_pos,
    seq_offset,
    seq_idx,
    mm_prefix_range_ptr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
):""", """def compute_kv_seq_mask(
    query_abs_pos,
    seq_offset,
    seq_idx,
    mm_prefix_range_ptr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    USE_CAUSAL: tl.constexpr = True,
    seq_len=0,
):""")

edit(H, """    # Compute attention mask: causal by default (key <= query)
    seq_mask = seq_offset[None, :] <= query_abs_pos""",
"""    # Compute attention mask: causal by default (key <= query)
    if USE_CAUSAL:
        seq_mask = seq_offset[None, :] <= query_abs_pos
    else:
        # Non-causal: bound by seq_len to exclude tile-overhang.
        seq_mask = (seq_offset[None, :] < seq_len) & (query_abs_pos >= 0)""")

# ===== 2) unified_attention: remove assert, add USE_CAUSAL to kernel call =====
edit(K, """    assert causal, "Only causal attention is supported"
""", """    use_causal = bool(causal)
""")

# Add USE_CAUSAL param to kernel_unified_attention signature
src = K.read_text()
if "USE_CAUSAL" not in src:
    # Find the kernel signature and add USE_CAUSAL before the closing ):
    i = src.index("def kernel_unified_attention(")
    # Find the ): that closes the signature
    j = src.index("\n):", i)
    src = src[:j] + "\n    USE_CAUSAL: tl.constexpr = True,  # bool" + src[j:]
    K.write_text(src)
    print("  added USE_CAUSAL to kernel_unified_attention signature")

# Pass USE_CAUSAL (+seq_len when upstream already threads it) to compute_kv_seq_mask
if not edit(K, """        query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
            seq_len,
        )""", """        query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
            USE_CAUSAL,
            seq_len,
        )"""):
    edit(K, """        query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
        )""", """        query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
            USE_CAUSAL,
            seq_len,
        )""")

# Pass USE_CAUSAL=use_causal in the kernel launch
edit(K, """        SLIDING_WINDOW=(1 + window_size[0]),
""", """        SLIDING_WINDOW=(1 + window_size[0]),
        USE_CAUSAL=use_causal,
""")

# ===== 3) triton_attn backend: add supports_non_causal + causal metadata =====
# Add supports_non_causal method
edit(B, """    @classmethod
    def supports_sink(cls) -> bool:
        return True
""", """    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True
""")

# Add causal field to TritonAttentionMetadata
edit(B, """    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None
""", """    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None
    causal: bool = True
""")

# Pass causal from common_attn_metadata to TritonAttentionMetadata.
# Keep use_cascade=use_cascade (local bool); do NOT rename to use_causal —
# that NameError kills Ray workers on first sample_tokens.
edit(B, """            slot_mapping=slot_mapping,
            use_cascade=use_cascade,""", """            slot_mapping=slot_mapping,
            causal=common_attn_metadata.causal,
            use_cascade=use_cascade,""")
# Use metadata causal in the kernel call
edit(B, """            softmax_scale=self.scale,
            causal=True,""", """            softmax_scale=self.scale,
            causal=attn_metadata.causal,""")

# ===== compile =====
import py_compile
for f in (H, K, B):
    if f.exists():
        py_compile.compile(str(f), doraise=True)
print("patch_triton_noncausal: DONE")
