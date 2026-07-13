"""Backport upstream fix: DFlash target_layer_ids use 'hidden state AFTER
layer i' semantics; vLLM's aux-hidden-state mixin indexes 'state ENTERING
layer k' (k=0 is embeddings). The overlay image already has this fix
baked in with the +1 conversion. This patch is a runtime no-op."""
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py")
src = p.read_text()

if "i + 1 for i in" in src:
    print("aux_layer_off_by_one: already baked in (i+1 conversion present)")
else:
    old = """            dflash_config = getattr(hf_config, "dflash_config", None)
            if dflash_config and isinstance(dflash_config, dict):
                layer_ids = dflash_config.get("target_layer_ids")"""
    new = """            dflash_config = getattr(hf_config, "dflash_config", None)
            if dflash_config and isinstance(dflash_config, dict):
                # Add 1 to convert DFlash's aux layer id semantics
                layer_ids = [
                    i + 1 for i in (dflash_config.get("target_layer_ids") or [])
                ]"""
    if new in src:
        print("already patched")
    elif old in src:
        p.write_text(src.replace(old, new, 1))
        import py_compile
        py_compile.compile(str(p), doraise=True)
        print("patch_aux_layer_off_by_one: APPLIED")
    else:
        print("aux_layer_off_by_one: anchor not found (code evolved with eagle_config handling)")
