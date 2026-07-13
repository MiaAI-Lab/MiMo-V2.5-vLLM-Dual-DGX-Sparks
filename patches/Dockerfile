FROM vllm-node-mimo-v25-nvfp4:latest

COPY qwen3_dflash.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py
COPY dflash_proposer.py /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dflash.py
COPY patch_mimo_v2_eagle3.py /tmp/patch_mimo_v2_eagle3.py

RUN python3 /tmp/patch_mimo_v2_eagle3.py && \
    python3 -m py_compile \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dflash.py
