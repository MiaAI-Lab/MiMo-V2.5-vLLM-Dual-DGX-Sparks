#!/bin/bash
set -euo pipefail
SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
cd "$SITE_PACKAGES"

echo "[fix-mimo-mtp-patches] get_top_tokens + MTP1_GREEDY_FAST + draft/target align + MTP debug + #41834"

python3 - <<'PY'
from pathlib import Path

patches = [
    (
        Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py'),
        '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
''',
        '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
''',
        'MiMoV2FlashForCausalLM',
    ),
    (
        Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py'),
        '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)
''',
        '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model.get_top_tokens(hidden_states)
''',
        'MiMoV2OmniForCausalLM',
    ),
]

for path, old, new, class_name in patches:
    text = path.read_text()
    if new in text:
        print(f'[fix-mimo-mtp-patches] {class_name}.get_top_tokens already patched')
        continue
    if old not in text:
        raise SystemExit(f'[fix-mimo-mtp-patches] ERROR: {class_name} anchor not found')
    path.write_text(text.replace(old, new, 1))
    print(f'[fix-mimo-mtp-patches] patched {class_name}.get_top_tokens')
PY

python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = path.read_text()
orig = text

if 'VLLM_MIMO_MTP1_GREEDY_FAST' in text:
    print('[fix-mimo-mtp-patches] greedy MTP1 fast path already patched')
else:
    if '\nimport os\n' not in text:
        text = text.replace('import itertools\n', 'import itertools\nimport os\n', 1)

    old = '''    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
'''
    new = '''    logits: torch.Tensor | None
    greedy_spec_top_token_ids: torch.Tensor | None
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
'''
    if new not in text:
        text = text.replace(old, new, 1)

    old = '''    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
'''
    new = '''    def _mimo_mtp1_greedy_fast_guard(
        self,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> bool:
        if os.environ.get("VLLM_MIMO_MTP1_GREEDY_FAST", "0") != "1":
            return False
        if spec_decode_metadata is None or self.num_spec_tokens != 1:
            return False
        if not hasattr(self.model, "get_top_tokens"):
            return False
        if any(num_draft != 1 for num_draft in spec_decode_metadata.num_draft_tokens):
            return False
        sampling_metadata = self.input_batch.sampling_metadata
        logitsprocs = sampling_metadata.logitsprocs
        has_logitsprocs = bool(logitsprocs.non_argmax_invariant)
        thinking_holder = sampling_metadata.thinking_budget_state_holder
        has_thinking_budget = (
            thinking_holder is not None
            and thinking_holder.has_tracked_requests()
        )
        ok = (
            sampling_metadata.all_greedy
            and sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logprob_token_ids
            and sampling_metadata.no_penalties
            and sampling_metadata.allowed_token_ids_mask is None
            and not sampling_metadata.bad_words_token_ids
            and not has_logitsprocs
            and not self.num_prompt_logprobs
            and not has_thinking_budget
        )
        if not ok and not getattr(self, "_mimo_mtp_guard_debug_logged", False):
            logger.warning(
                "MiMo MTP1 greedy-fast guard blocked: all_greedy=%s "
                "logprobs=%s penalties=%s logitsprocs=%s thinking=%s",
                sampling_metadata.all_greedy,
                sampling_metadata.max_num_logprobs is not None,
                not sampling_metadata.no_penalties,
                has_logitsprocs,
                has_thinking_budget,
            )
            self._mimo_mtp_guard_debug_logged = True
        return ok

    def _sample_mimo_mtp1_greedy_fast(
        self,
        greedy_spec_top_token_ids: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata,
    ) -> SamplerOutput:
        target_token_ids = greedy_spec_top_token_ids.index_select(
            0,
            spec_decode_metadata.target_logits_indices.long(),
        ).long()
        bonus_token_ids = greedy_spec_top_token_ids.index_select(
            0,
            spec_decode_metadata.bonus_logits_indices.long(),
        ).long()
        draft_token_ids = spec_decode_metadata.draft_token_ids.long()
        accepted = draft_token_ids.eq(target_token_ids)
        if not getattr(self, "_mimo_mtp_accept_debug_logged", False):
            n = min(8, draft_token_ids.numel())
            if n > 0:
                logger.warning(
                    "MiMo MTP1 greedy-fast accept debug: "
                    "draft=%s target=%s match=%s/%s",
                    draft_token_ids[:n].tolist(),
                    target_token_ids[:n].tolist(),
                    int(accepted[:n].sum().item()),
                    n,
                )
            self._mimo_mtp_accept_debug_logged = True

        output_token_ids = torch.full(
            (len(spec_decode_metadata.num_draft_tokens), 2),
            -1,
            dtype=torch.int32,
            device=draft_token_ids.device,
        )
        output_token_ids[:, 0] = torch.where(
            accepted,
            draft_token_ids,
            target_token_ids,
        ).to(torch.int32)
        output_token_ids[:, 1] = torch.where(
            accepted,
            bonus_token_ids,
            output_token_ids[:, 1].long(),
        ).to(torch.int32)
        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=None,
        )

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        greedy_spec_top_token_ids: torch.Tensor | None = None,
    ) -> SamplerOutput:
'''
    if new not in text:
        text = text.replace(old, new, 1)

    old = '''        draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            draft_probs,
            logits,
            sampling_metadata,
        )
        return sampler_output
'''
    new = '''        if greedy_spec_top_token_ids is not None:
            return self._sample_mimo_mtp1_greedy_fast(
                greedy_spec_top_token_ids,
                spec_decode_metadata,
            )

        draft_probs = self._get_spec_decode_draft_probs(spec_decode_metadata)
        sampler_output = self.rejection_sampler(
            spec_decode_metadata,
            draft_probs,
            logits,
            sampling_metadata,
        )
        return sampler_output
'''
    if new not in text:
        text = text.replace(old, new, 1)

    old = '''                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
'''
    new = '''                sample_hidden_states = hidden_states[logits_indices]
                greedy_spec_top_token_ids = None
                if self._mimo_mtp1_greedy_fast_guard(spec_decode_metadata):
                    greedy_spec_top_token_ids = self.model.get_top_tokens(
                        sample_hidden_states
                    )
                    logits = None
                else:
                    logits = self.model.compute_logits(sample_hidden_states)
'''
    if new not in text:
        text = text.replace(old, new, 1)

    old = '''                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
'''
    new = '''                sample_hidden_states = hidden_states[logits_indices]
                greedy_spec_top_token_ids = None
                if not get_pp_group().is_last_rank:
'''
    if new not in text:
        text = text.replace(old, new, 1)

    old = '''            scheduler_output,
            logits,
            spec_decode_metadata,
'''
    new = '''            scheduler_output,
            logits,
            greedy_spec_top_token_ids,
            spec_decode_metadata,
'''
    while old in text:
        text = text.replace(old, new, 1)

    old = '''        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )
'''
    new = '''        # Apply structured output bitmasks if present. Structured output needs
        # full logits, so any speculative top-token fast path falls back here.
        if grammar_output is not None:
            if logits is None:
                logits = self.model.compute_logits(sample_hidden_states)
                greedy_spec_top_token_ids = None
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )
'''
    if new not in text:
        text = text.replace(old, new, 1)

    old = '''            sampler_output = self._sample(logits, spec_decode_metadata)
'''
    new = '''            sampler_output = self._sample(
                logits,
                spec_decode_metadata,
                greedy_spec_top_token_ids,
            )
'''
    if new not in text:
        text = text.replace(old, new, 1)

    path.write_text(text)
    print('[fix-mimo-mtp-patches] patched greedy MTP1 target top-token fast path')

import ast
ast.parse(path.read_text(), filename=str(path))
PY

python3 - <<'PY'
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py')
text = path.read_text()
orig = text

runner_anchor = '''        self._share_mtp_indices = False

        self.device = device'''
runner_patch = '''        self._share_mtp_indices = False
        self.runner = runner

        self.device = device'''
if 'self.runner = runner' not in text:
    if runner_anchor not in text:
        raise SystemExit('[fix-mimo-mtp-patches] ERROR: proposer runner anchor not found')
    text = text.replace(runner_anchor, runner_patch, 1)
    print('[fix-mimo-mtp-patches] patched proposer to retain gpu_model_runner (rejection_sampler access)')
else:
    print('[fix-mimo-mtp-patches] proposer runner retention already patched')

align_helpers = '''    def _mimo_draft_needs_target_logits_processors(
        self,
        sampling_metadata: SamplingMetadata,
    ) -> bool:
        logitsprocs = sampling_metadata.logitsprocs
        holder = sampling_metadata.thinking_budget_state_holder
        return (
            not sampling_metadata.no_penalties
            or bool(sampling_metadata.bad_words_token_ids)
            or bool(logitsprocs.non_argmax_invariant)
            or (
                holder is not None
                and holder.has_tracked_requests()
            )
            or sampling_metadata.allowed_token_ids_mask is not None
        )

    def _mimo_greedy_sample_aligned(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if not self._mimo_draft_needs_target_logits_processors(sampling_metadata):
            if self.use_local_argmax_reduction and hasattr(self.model, "get_top_tokens"):
                return self.model.get_top_tokens(hidden_states)
            return self.model.compute_logits(hidden_states).argmax(dim=-1)
        logits = self.model.compute_logits(hidden_states)
        runner = getattr(self, "runner", None)
        rejection_sampler = (
            getattr(runner, "rejection_sampler", None)
            if runner is not None
            else None
        )
        if rejection_sampler is None:
            return logits.argmax(dim=-1)
        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
        from vllm.v1.sample.rejection_sampler import apply_sampling_constraints

        batch_size = logits.shape[0]
        metadata = SpecDecodeMetadata.make_dummy(
            [[0]] * batch_size,
            device=logits.device,
        )
        logits = logits.to(torch.float32).clone()
        logits = rejection_sampler.apply_logits_processors(
            logits,
            sampling_metadata,
            metadata,
        )
        logits = apply_sampling_constraints(
            logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )
        return logits.argmax(dim=-1)

'''
if '_mimo_greedy_sample_aligned' not in text:
    anchor = '    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:'
    if anchor not in text:
        raise SystemExit('[fix-mimo-mtp-patches] ERROR: _greedy_sample anchor not found')
    text = text.replace(anchor, align_helpers + anchor, 1)
    print('[fix-mimo-mtp-patches] patched draft greedy sampling to mirror target logits processors')
else:
    print('[fix-mimo-mtp-patches] draft/target aligned greedy sampling already patched')

if 'self._mimo_greedy_sample_aligned(hidden_states, sampling_metadata)' not in text:
    if 'return self._greedy_sample(hidden_states), None' not in text:
        raise SystemExit('[fix-mimo-mtp-patches] ERROR: _sample_draft_tokens greedy anchor not found')
    text = text.replace(
        'return self._greedy_sample(hidden_states), None',
        'return self._mimo_greedy_sample_aligned(hidden_states, sampling_metadata), None',
        1,
    )
    print('[fix-mimo-mtp-patches] wired _sample_draft_tokens to aligned greedy path')
else:
    print('[fix-mimo-mtp-patches] _sample_draft_tokens already uses aligned greedy path')

if text != orig:
    path.write_text(text)

import ast
ast.parse(path.read_text(), filename=str(path))
PY

python3 - <<'PY'
from pathlib import Path
import re

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py')
text = path.read_text()
orig = text

clean_sample_draft = '''    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            return self._mimo_greedy_sample_aligned(hidden_states, sampling_metadata), None
        logits = self.model.compute_logits(hidden_states)
        return self._sample_from_logits(logits, sampling_metadata)
'''

if clean_sample_draft not in text:
    match = re.search(
        r'    def _sample_draft_tokens\([\s\S]*?'
        r'return self\._sample_from_logits\(logits, sampling_metadata\)\n',
        text,
    )
    if match is None:
        raise SystemExit('[fix-mimo-mtp-patches] ERROR: _sample_draft_tokens anchor not found')
    text = text[: match.start()] + clean_sample_draft + text[match.end() :]
    print('[fix-mimo-mtp-patches] normalized _sample_draft_tokens (no os/debug; aligned greedy)')
else:
    print('[fix-mimo-mtp-patches] _sample_draft_tokens already normalized')

if 'os.environ.get("VLLM_MIMO_MTP_DEBUG"' in text:
    raise SystemExit('[fix-mimo-mtp-patches] ERROR: failed to remove VLLM_MIMO_MTP_DEBUG block')

if text != orig:
    path.write_text(text)

import ast
ast.parse(path.read_text(), filename=str(path))
PY

python3 - <<'PY'
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py')
text = path.read_text()
orig = text

anchor = '''        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
'''
debug = '''        target_argmax = target_logits.argmax(dim=-1)
        _mimo_dbg_n = getattr(rejection_sample, "_mimo_mtp_accept_debug_n", 0)
        if _mimo_dbg_n < 5:
            n = min(8, draft_token_ids.numel())
            if n > 0:
                draft = draft_token_ids[:n].tolist()
                target = target_argmax[:n].tolist()
                logger.warning(
                    "MiMo MTP accept debug: draft=%s target=%s match=%s/%s "
                    "all_greedy=%s no_penalties=%s",
                    draft,
                    target,
                    sum(d == t for d, t in zip(draft, target)),
                    n,
                    sampling_metadata.all_greedy,
                    sampling_metadata.no_penalties,
                )
            rejection_sample._mimo_mtp_accept_debug_n = _mimo_dbg_n + 1
        rejection_greedy_sample_kernel[(batch_size,)](
'''
buggy_debug = debug.replace('draft_token_ids', 'metadata.draft_token_ids')
if debug in text:
    print('[fix-mimo-mtp-patches] MTP accept debug already patched in rejection_sampler')
elif buggy_debug in text:
    text = text.replace(buggy_debug, debug, 1)
    path.write_text(text)
    print('[fix-mimo-mtp-patches] repaired buggy MTP accept debug in rejection_sampler')
elif anchor in text:
    text = text.replace(anchor, debug, 1)
    path.write_text(text)
    print('[fix-mimo-mtp-patches] patched MTP accept debug in rejection_sampler')
else:
    print('[fix-mimo-mtp-patches] MTP accept debug anchor not found (will repair if needed)')

import ast
ast.parse(path.read_text(), filename=str(path))
PY

python3 - <<'PY'
from pathlib import Path
import ast

# Call-site live debug: some v0.23 runs do not reach the function-level debug
# because metadata shape/dummy filtering hides it.  Log immediately before the
# classic RejectionSampler calls rejection_sample().
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py')
text = path.read_text()
old = '''        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=self.synthetic_mode,
            synthetic_conditional_rates=self.synthetic_conditional_rates,
            use_fp64_gumbel=self.use_fp64_gumbel,
            mimo_dummy_metadata=(
                metadata.draft_token_ids.numel() > 0
                and bool(torch.all(metadata.draft_token_ids == 0).item())
                and bool(torch.all(metadata.target_logits_indices == 0).item())
                and bool(torch.all(metadata.logits_indices == 0).item())
            ),
        )
'''
old_plain = '''        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=self.synthetic_mode,
            synthetic_conditional_rates=self.synthetic_conditional_rates,
            use_fp64_gumbel=self.use_fp64_gumbel,
        )
'''
new = '''        _mimo_call_dbg_n = getattr(self, "_mimo_mtp_call_debug_n", 0)
        if _mimo_call_dbg_n < 20 and metadata.draft_token_ids.numel() > 0:
            n = min(8, metadata.draft_token_ids.numel())
            draft_dbg = metadata.draft_token_ids[:n].long()
            target_dbg = target_logits.argmax(dim=-1)[:n].long()
            logger.warning(
                "MiMo MTP rejection call debug live: draft=%s target=%s "
                "match=%s/%s all_greedy=%s no_penalties=%s num_draft=%s "
                "target_idx=%s logits_idx=%s",
                draft_dbg.tolist(),
                target_dbg.tolist(),
                int(draft_dbg.eq(target_dbg).sum().item()),
                n,
                sampling_metadata.all_greedy,
                sampling_metadata.no_penalties,
                metadata.num_draft_tokens[:8],
                metadata.target_logits_indices[:n].tolist(),
                metadata.logits_indices[: min(16, metadata.logits_indices.numel())].tolist(),
            )
            self._mimo_mtp_call_debug_n = _mimo_call_dbg_n + 1

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=self.synthetic_mode,
            synthetic_conditional_rates=self.synthetic_conditional_rates,
            use_fp64_gumbel=self.use_fp64_gumbel,
            mimo_dummy_metadata=(
                metadata.draft_token_ids.numel() > 0
                and bool(torch.all(metadata.draft_token_ids == 0).item())
                and bool(torch.all(metadata.target_logits_indices == 0).item())
                and bool(torch.all(metadata.logits_indices == 0).item())
            ),
        )
'''
if new in text:
    print('[fix-mimo-mtp-patches] rejection call live debug already patched')
elif old in text:
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-mtp-patches] patched rejection call live debug')
elif old_plain in text:
    path.write_text(text.replace(old_plain, new, 1))
    print('[fix-mimo-mtp-patches] patched rejection call live debug')
else:
    raise SystemExit('[fix-mimo-mtp-patches] ERROR: rejection call live debug anchor not found')

ast.parse(path.read_text(), filename=str(path))
PY

python3 - <<'PY'
from pathlib import Path
import ast

# Some v0.23 builds route V1 speculative sampling through
# vllm/v1/worker/gpu/spec_decode/rejection_sampler.py instead of
# vllm/v1/sample/rejection_sampler.py.  Instrument that wrapper too so live
# logs show real draft IDs against target argmax before the Triton kernel.
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py')
if path.exists():
    text = path.read_text()
    if 'from vllm.logger import init_logger' not in text:
        text = text.replace('import torch\n', 'import torch\n\nfrom vllm.logger import init_logger\n', 1)
    if 'logger = init_logger(__name__)' not in text:
        text = text.replace(
            'from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (\n'
            '    rejection_sample,\n'
            ')\n',
            'from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (\n'
            '    rejection_sample,\n'
            ')\n\n'
            'logger = init_logger(__name__)\n',
            1,
        )
    old = '''        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            input_batch.cu_num_logits,
            pos,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
        )
'''
    new = '''        _mimo_dbg_n = getattr(self, "_mimo_mtp_gpu_accept_debug_n", 0)
        if _mimo_dbg_n < 20:
            verify_mask = input_batch.expanded_local_pos < self.num_speculative_steps
            verify_indices = torch.nonzero(verify_mask, as_tuple=False).flatten()
            n = min(8, int(verify_indices.numel()))
            if n > 0:
                idx = verify_indices[:n]
                draft_dbg = draft_sampled.index_select(0, idx).long()
                target_dbg = processed_logits.index_select(0, idx).argmax(dim=-1).long()
                local_pos_dbg = input_batch.expanded_local_pos.index_select(0, idx).long()
                logger.warning(
                    "MiMo MTP GPU accept debug live: draft=%s target=%s "
                    "match=%s/%s local_pos=%s",
                    draft_dbg.tolist(),
                    target_dbg.tolist(),
                    int(draft_dbg.eq(target_dbg).sum().item()),
                    n,
                    local_pos_dbg.tolist(),
                )
            self._mimo_mtp_gpu_accept_debug_n = _mimo_dbg_n + 1

        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            input_batch.cu_num_logits,
            pos,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
        )
'''
    if new in text:
        print('[fix-mimo-mtp-patches] GPU rejection accept debug already patched')
    elif old in text:
        path.write_text(text.replace(old, new, 1))
        print('[fix-mimo-mtp-patches] patched GPU rejection accept debug')
    else:
        print('[fix-mimo-mtp-patches] GPU rejection accept debug anchor not found (non-fatal)')
    ast.parse(path.read_text(), filename=str(path))
else:
    print('[fix-mimo-mtp-patches] GPU rejection sampler not present; skipping')
PY

python3 - <<'PY'
from pathlib import Path
import ast

# The warmup/profile path calls rejection_sampler with
# SpecDecodeMetadata.make_dummy([[0], ...]), which used to consume the first
# MiMo accept-debug slots and made live failures look like draft token 0.  Log
# only non-dummy metadata so the decisive debug line is from real requests.
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py')
text = path.read_text()
call_old = '''            use_fp64_gumbel=self.use_fp64_gumbel,
        )
'''
call_new = '''            use_fp64_gumbel=self.use_fp64_gumbel,
            mimo_dummy_metadata=(
                metadata.draft_token_ids.numel() > 0
                and bool(torch.all(metadata.draft_token_ids == 0).item())
                and bool(torch.all(metadata.target_logits_indices == 0).item())
                and bool(torch.all(metadata.logits_indices == 0).item())
            ),
        )
'''
if call_new not in text and call_old in text:
    text = text.replace(call_old, call_new, 1)

sig_old = '''    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
'''
sig_new = '''    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64_gumbel: bool = False,
    mimo_dummy_metadata: bool = False,
) -> torch.Tensor:
'''
if sig_new not in text and sig_old in text:
    text = text.replace(sig_old, sig_new, 1)

old = '''        _mimo_dbg_n = getattr(rejection_sample, "_mimo_mtp_accept_debug_n", 0)
        if _mimo_dbg_n < 5:
            n = min(8, draft_token_ids.numel())
            if n > 0:
                draft = draft_token_ids[:n].tolist()
                target = target_argmax[:n].tolist()
                logger.warning(
                    "MiMo MTP accept debug: draft=%s target=%s match=%s/%s "
                    "all_greedy=%s no_penalties=%s",
                    draft,
                    target,
                    sum(d == t for d, t in zip(draft, target)),
                    n,
                    sampling_metadata.all_greedy,
                    sampling_metadata.no_penalties,
                )
            rejection_sample._mimo_mtp_accept_debug_n = _mimo_dbg_n + 1
'''
new = '''        _mimo_dbg_n = getattr(rejection_sample, "_mimo_mtp_accept_debug_n", 0)
        if not mimo_dummy_metadata and _mimo_dbg_n < 20:
            n = min(8, draft_token_ids.numel())
            if n > 0:
                draft = draft_token_ids[:n].tolist()
                target = target_argmax[:n].tolist()
                logger.warning(
                    "MiMo MTP accept debug live: draft=%s target=%s match=%s/%s "
                    "all_greedy=%s no_penalties=%s",
                    draft,
                    target,
                    sum(d == t for d, t in zip(draft, target)),
                    n,
                    sampling_metadata.all_greedy,
                    sampling_metadata.no_penalties,
                )
            rejection_sample._mimo_mtp_accept_debug_n = _mimo_dbg_n + 1
'''
if new in text:
    print('[fix-mimo-mtp-patches] live-only MTP accept debug already patched')
elif old in text:
    text = text.replace(old, new, 1)
    print('[fix-mimo-mtp-patches] patched MTP accept debug to skip dummy/profile metadata')
else:
    print('[fix-mimo-mtp-patches] MTP accept debug live-only anchor not found (non-fatal)')

path.write_text(text)
ast.parse(path.read_text(), filename=str(path))
PY

python3 - <<'PY'
from pathlib import Path
import sys

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = path.read_text()

if 'sync_without_prev_positions' in text:
    print('[fix-mimo-mtp-patches] #41834 already patched')
    raise SystemExit(0)

v023_snippet = '''        # PR #41834: sync_without_prev_positions fallback (v0.23 port).
        num_draft_tokens = spec_decode_metadata.num_draft_tokens
        total_num_draft_tokens = sum(num_draft_tokens)
        prev_positions = self.prev_positions.np[: len(num_draft_tokens)]
        sync_without_prev_positions = (
            not self.use_async_scheduling and np.all(prev_positions < 0)
        )
        if sync_without_prev_positions:
            draft_probs = self._draft_probs
            if draft_probs.ndim == 2:
                return draft_probs[:total_num_draft_tokens].contiguous()
            if draft_probs.shape[0] >= len(num_draft_tokens):
                packed_probs = []
                draft_row = 0
                for num_tokens in num_draft_tokens:
                    if num_tokens == 0:
                        continue
                    if draft_row >= draft_probs.shape[0]:
                        raise RuntimeError(
                            "Spec decode metadata references more draft token "
                            "rows than were recorded by the draft model."
                        )
                    packed_probs.append(draft_probs[draft_row, :num_tokens])
                    draft_row += 1
                if not packed_probs:
                    return None
                return torch.cat(packed_probs, dim=0).contiguous()

'''

anchor_v023 = '''        if self._draft_probs is None or self._draft_prob_req_ids is None:
            return None

        row_by_req_id = {'''
replacement_v023 = '''        if self._draft_probs is None or self._draft_prob_req_ids is None:
            return None

''' + v023_snippet + '''        row_by_req_id = {'''

if anchor_v023 not in text:
    raise SystemExit('[fix-mimo-mtp-patches] ERROR: #41834 v0.23 anchor not found')

path.write_text(text.replace(anchor_v023, replacement_v023, 1))
print('[fix-mimo-mtp-patches] patched #41834 v0.23 _get_spec_decode_draft_probs')
import ast
ast.parse(path.read_text(), filename=str(path))
PY

python3 - <<'PY'
from pathlib import Path
import ast

# vLLM's generic overflow path used token id 0 as a "no draft" placeholder.
# For MTP this is catastrophic: the scheduler treats 0 as a real draft token on
# the next step, rejection compares it to target argmax, and acceptance stays 0%.
# Publish per-request empty draft lists instead.
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
text = path.read_text()
old = '''            if not input_fits_in_drafter:
                # Zero out draft tokens so the scheduler doesn't schedule
                # stale drafts from the previous step.
                # For Nemotron-H: it is necessary to zero out the draft tokens,
                # otherwise the stale tokens will corrupt Mamba recurrent
                # state and logprobs for sequences near max_model_len.
                self._draft_token_ids = torch.zeros(
                    1, device=self.device, dtype=torch.int32
                ).expand(len(self.input_batch.req_ids), self.num_spec_tokens)
                self._draft_probs = None
                self._draft_prob_req_ids = None
                self._copy_draft_token_ids_to_cpu(scheduler_output, zeros_only=True)
'''
new = '''            if not input_fits_in_drafter:
                # Do not publish token id 0 as a no-draft placeholder. The
                # scheduler treats every listed id as a real draft token, so
                # zeros force 0% acceptance on the next target step.
                self._draft_token_ids = [[] for _ in self.input_batch.req_ids]
                self._draft_probs = None
                self._draft_prob_req_ids = None
                self._draft_token_req_ids = self.input_batch.req_ids.copy()
                if not getattr(self, "_mimo_no_draft_debug_logged", False):
                    logger.warning(
                        "MiMo MTP drafter skipped: input does not fit drafter "
                        "max len; publishing empty draft lists instead of "
                        "token id 0 placeholders"
                    )
                    self._mimo_no_draft_debug_logged = True
'''
if new in text:
    print('[fix-mimo-mtp-patches] no-draft empty-list fallback already patched')
elif old in text:
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-mtp-patches] patched no-draft fallback to publish empty draft lists')
else:
    print('[fix-mimo-mtp-patches] no-draft fallback anchor not found (non-fatal)')

ast.parse(path.read_text(), filename=str(path))
PY

python3 - <<'PY'
from pathlib import Path
import ast

proposer = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py')
text = proposer.read_text()
ast.parse(text, filename=str(proposer))
if '_mimo_greedy_sample_aligned' not in text:
    raise SystemExit('[fix-mimo-mtp-patches] ERROR: _mimo_greedy_sample_aligned missing after patch')
if 'self.runner = runner' not in text:
    raise SystemExit('[fix-mimo-mtp-patches] ERROR: proposer runner retention missing after patch')
if 'os.environ.get("VLLM_MIMO_MTP_DEBUG"' in text:
    raise SystemExit('[fix-mimo-mtp-patches] ERROR: stale VLLM_MIMO_MTP_DEBUG block still present')
rs = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py').read_text()
if 'MiMo MTP rejection call debug live' not in rs:
    raise SystemExit('[fix-mimo-mtp-patches] ERROR: rejection call live debug missing after patch')
print('[fix-mimo-mtp-patches] runtime check OK (draft/target align + runner)')
PY

python3 - <<'PY'
from pathlib import Path
import ast

# Repair already-patched containers: logitsprocs guard + aligned draft sampling.
runner = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py')
proposer = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/llm_base_proposer.py')
rs = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py')

rtext = runner.read_text()
old_guard = '''        has_logitsprocs = bool(logitsprocs.argmax_invariant) or bool(
            logitsprocs.non_argmax_invariant
        )'''
new_guard = '        has_logitsprocs = bool(logitsprocs.non_argmax_invariant)'
if old_guard in rtext:
    rtext = rtext.replace(old_guard, new_guard, 1)
    runner.write_text(rtext)
    print('[fix-mimo-mtp-patches] repaired greedy-fast guard (non_argmax logitsprocs only)')

ptext = proposer.read_text()
old_needs = '''            or bool(logitsprocs.argmax_invariant)
            or bool(logitsprocs.non_argmax_invariant)'''
new_needs = '            or bool(logitsprocs.non_argmax_invariant)'
if old_needs in ptext:
    ptext = ptext.replace(old_needs, new_needs, 1)
    print('[fix-mimo-mtp-patches] repaired draft processor guard (non_argmax only)')

old_aligned = '''        if (
            self.use_local_argmax_reduction
            and not self._mimo_draft_needs_target_logits_processors(
                sampling_metadata
            )
        ):
            return self.model.get_top_tokens(hidden_states)
        logits = self.model.compute_logits(hidden_states)
        if not self._mimo_draft_needs_target_logits_processors(sampling_metadata):
            return logits.argmax(dim=-1)'''
new_aligned = '''        if not self._mimo_draft_needs_target_logits_processors(sampling_metadata):
            if self.use_local_argmax_reduction and hasattr(self.model, "get_top_tokens"):
                return self.model.get_top_tokens(hidden_states)
            return self.model.compute_logits(hidden_states).argmax(dim=-1)
        logits = self.model.compute_logits(hidden_states)'''
if old_aligned in ptext:
    ptext = ptext.replace(old_aligned, new_aligned, 1)
    print('[fix-mimo-mtp-patches] repaired draft greedy path (respect use_local_argmax_reduction)')

old_aligned_always = '''        if not self._mimo_draft_needs_target_logits_processors(sampling_metadata):
            if hasattr(self.model, "get_top_tokens"):
                return self.model.get_top_tokens(hidden_states)
            return self.model.compute_logits(hidden_states).argmax(dim=-1)
        logits = self.model.compute_logits(hidden_states)'''
if old_aligned_always in ptext:
    ptext = ptext.replace(old_aligned_always, new_aligned, 1)
    print('[fix-mimo-mtp-patches] repaired draft greedy path (respect use_local_argmax_reduction)')

if 'apply_sampling_constraints' not in ptext and '_mimo_greedy_sample_aligned' in ptext:
    ptext = ptext.replace(
        '        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata\n',
        '        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata\n'
        '        from vllm.v1.sample.rejection_sampler import apply_sampling_constraints\n',
        1,
    )
    ptext = ptext.replace(
        '''        logits = rejection_sampler.apply_logits_processors(
            logits,
            sampling_metadata,
            metadata,
        )
        return logits.argmax(dim=-1)''',
        '''        logits = rejection_sampler.apply_logits_processors(
            logits,
            sampling_metadata,
            metadata,
        )
        logits = apply_sampling_constraints(
            logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )
        return logits.argmax(dim=-1)''',
        1,
    )
    print('[fix-mimo-mtp-patches] repaired draft processor path (apply_sampling_constraints)')

proposer.write_text(ptext)

rstext = rs.read_text()
old_dbg_flag = 'getattr(rejection_sample, "_mimo_mtp_accept_debug_logged", False)'
new_dbg_flag = 'getattr(rejection_sample, "_mimo_mtp_accept_debug_n", 0) < 5'
if old_dbg_flag in rstext:
    rstext = rstext.replace(old_dbg_flag, new_dbg_flag, 1)
    rstext = rstext.replace(
        'rejection_sample._mimo_mtp_accept_debug_logged = True',
        'rejection_sample._mimo_mtp_accept_debug_n = _mimo_dbg_n + 1',
        1,
    )
    if '_mimo_dbg_n = getattr' not in rstext:
        rstext = rstext.replace(
            '        target_argmax = target_logits.argmax(dim=-1)\n        if',
            '        target_argmax = target_logits.argmax(dim=-1)\n'
            '        _mimo_dbg_n = getattr(rejection_sample, "_mimo_mtp_accept_debug_n", 0)\n'
            '        if',
            1,
        )
    rs.write_text(rstext)
    print('[fix-mimo-mtp-patches] repaired MTP accept debug (first 5 samples)')

ast.parse(runner.read_text(), filename=str(runner))
ast.parse(proposer.read_text(), filename=str(proposer))
ast.parse(rs.read_text(), filename=str(rs))
PY

find "$SITE_PACKAGES/vllm" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "[fix-mimo-mtp-patches] done"
