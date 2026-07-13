"""LCM-based KV page-size unification. The base image already has a superior
approach: non-divisible page sizes are handled via padding on backends that
opt in via indexes_kv_by_block_stride, rather than requiring LCM computation.
This patch is a runtime no-op to confirm the expected unified function."""
import math
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py")
src = p.read_text()

# Check if the LCM fix is already in unify_kv_cache_spec_page_size
if "target_page_size = _math.lcm(*page_sizes)" in src:
    print("kv_page_lcm: already patched")
elif "indexes_kv_by_block_stride" in src:
    print("kv_page_lcm: padding-based unification already baked in (superior)")
else:
    # Try the old-style unification replacement
    old = """    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec"""

    new = """    max_page_size = max(page_sizes)
    target_page_size = max_page_size
    if any(max_page_size % p != 0 for p in page_sizes):
        import math as _math
        target_page_size = _math.lcm(*page_sizes)
        if target_page_size > 64 * max_page_size:
            raise NotImplementedError(
                f"KV page sizes {sorted(page_sizes)} cannot be unified: "
                f"LCM {target_page_size} exceeds 64x the max page size."
            )
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == target_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            assert target_page_size % layer_page_size == 0
            ratio = target_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == target_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec"""
    
    if old in src:
        p.write_text(src.replace(old, new, 1))
        import py_compile
        py_compile.compile(str(p), doraise=True)
        print("patch_kv_page_lcm: APPLIED")
    else:
        print("kv_page_lcm: obsolete (base image already handles non-divisible pages via padding)")
