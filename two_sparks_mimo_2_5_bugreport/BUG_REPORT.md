# MiMo V2.5 dual-DGX-Spark engine crash on multimodal request

## Summary

A multimodal OpenAI-compatible chat-completion request caused the worker on
the second DGX Spark to fail while merging image embeddings. The vLLM
EngineCore then terminated and the API server shut down. Both Docker
containers continued to report an `Up` status, but the API port no longer
accepted connections.

Recovering required stopping and restarting the complete two-node service.
Cold recovery took approximately 12 minutes because the model had to reload.

## Environment

- Recipe: `MiaAI-Lab/MiMo-V2.5-vLLM-Dual-DGX-Sparks`
- Recipe commit: `28960fbbe9b16740a40c6f5fc464f16263e29a36`
- Runtime image: `ghcr.io/miaai-lab/mimo-v2.5-vllm-dual-dgx-sparks:20260704`
- vLLM: `0.21.1rc1.dev85+gd87ee1893.d20260518`
- Served model: `MiMo-V2.5-NVFP4`
- Model source: `lukealonso/MiMo-V2.5-NVFP4`
- Topology: two DGX Sparks, tensor parallel size 2
- Context length: 1,000,000 tokens
- KV cache: NVFP4
- Speculative decoding: MTP, one speculative token
- Transport: NCCL over the direct RoCE link

## Request characteristics

The request came from an OpenAI-compatible client using the model as an
auxiliary vision model. The original user prompt and image are intentionally
excluded from this report.

- Modality: one image plus text
- Prompt tokens scheduled: 922
- `max_tokens`: 999078
- `temperature`: 0.1
- `top_p`: 1.0
- `top_k`: 0
- `repetition_penalty`: 1.0
- `thinking_token_budget`: null

The unusually large `max_tokens` value resulted from a client with no explicit
output-token ceiling. The recipe documentation recommends a client cap of
32,768 tokens. The client has since been corrected.

## Observed behavior

1. The request was accepted and image preprocessing completed.
2. The worker on the second node failed in `embed_input_ids` while calling
   `_merge_multimodal_embeddings`.
3. The underlying exception was `torch.AcceleratorError: CUDA error: operation
   not permitted`.
4. vLLM wrapped this as `ValueError: Error during index put operation`.
5. EngineCore reported a fatal error and shut down the Ray distributed
   executor.
6. The API server exited, while both Docker containers remained running.
7. Subsequent text-only requests failed with connection errors.

## Expected behavior

- Reject or clamp an output-token request above the documented 32,768 client
  ceiling.
- Return an HTTP 4xx or 5xx response for a failed multimodal request.
- Keep EngineCore and the API server available for subsequent requests.
- Ideally expose a server-side maximum-output setting so a client cannot
  request nearly the entire context window as output.

## Recovery and validation

The cluster was stopped and started using the repository scripts. After the
model reloaded:

- the repository chat smoke test returned HTTP 200;
- a direct text request succeeded;
- a text request through the client succeeded.

Image input has not been retried because another failure would require a full
model reload. The client is temporarily configured as text-only.

## Questions

1. Is image input expected to be stable with this exact dual-Spark runtime?
2. Can the API enforce the documented 32,768 maximum output-token limit?
3. Can a multimodal CUDA/index-put failure be isolated to the request instead
   of terminating EngineCore?

## Privacy note

This package contains no original prompt, image, username, hostname, LAN or
RoCE address, machine identifier, access token, API key, session identifier,
or home-directory path.
