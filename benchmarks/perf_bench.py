#!/usr/bin/env python3
"""MiMo Omni MTP1 performance bench against a live OpenAI-compatible endpoint.

Matches Tony-style tables:
  - single-stream decode at 512 / 1024 / 2048 completion tokens
  - static concurrency aggregate tok/s
  - MTP acceptance from /metrics deltas

Usage:
  MIMO_BASE_URL=http://10.0.0.1:8888/v1 python3 benchmarks/perf_bench.py
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

BASE_URL = os.environ.get("MIMO_BASE_URL", "http://10.0.0.1:8888/v1").rstrip("/")
MODEL = os.environ.get("MIMO_MODEL", "MiMo-V2.5-NVFP4")
SINGLE_LENS = [int(x) for x in os.environ.get("SINGLE_LENS", "512,1024,2048").split(",") if x.strip()]
CONCURRENCY_LIST = [
    int(x) for x in os.environ.get("CONCURRENCY", "1,2,3,4,6,8").split(",") if x.strip()
]
CONCUR_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
TIMEOUT = int(os.environ.get("REQ_TIMEOUT", "900"))
OUT_JSON = os.environ.get("OUT_JSON", "")


def _post(path: str, payload: dict, timeout: int = TIMEOUT) -> dict:
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get_text(url: str, timeout: int = 10) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def metrics_snapshot() -> dict[str, float]:
    """Parse key vLLM Prometheus counters from /metrics (strip :port → root)."""
    root = BASE_URL.rsplit("/v1", 1)[0]
    text = _get_text(root + "/metrics")
    want = {
        "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
        "vllm:spec_decode_num_accepted_tokens_total": "accepted_tokens",
        "vllm:spec_decode_num_drafts_total": "drafts",
    }
    out: dict[str, float] = {v: 0.0 for v in want.values()}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for metric, key in want.items():
            if line.startswith(metric + "{") or line.startswith(metric + " "):
                out[key] = float(line.rsplit(" ", 1)[-1])
                break
    return out


def acceptance(before: dict[str, float], after: dict[str, float]) -> float | None:
    d_draft = after["draft_tokens"] - before["draft_tokens"]
    d_acc = after["accepted_tokens"] - before["accepted_tokens"]
    if d_draft <= 0:
        return None
    return round(d_acc / d_draft, 3)


def make_prompt(i: int, words: int = 360) -> str:
    filler = " ".join(f"mimo{i}_{j}" for j in range(words))
    return (
        "Write a practical agent implementation note in English. "
        "Do not switch languages. Do not repeat characters. Do not output XML. "
        "Keep the answer useful and concise.\n\n"
        f"Context salt {i}: {filler}"
    )


def chat(max_tokens: int, prompt: str, req_id: int = 0, *, ignore_eos: bool = False) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,  # raw tok/s (Tony speed tables); stability default is 1.08
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    t0 = time.perf_counter()
    data = _post("/chat/completions", payload)
    dt = time.perf_counter() - t0
    usage = data.get("usage") or {}
    content = (data["choices"][0]["message"].get("content") or "")
    completion = int(usage.get("completion_tokens") or 0)
    prompt_toks = int(usage.get("prompt_tokens") or 0)
    return {
        "id": req_id,
        "seconds": round(dt, 3),
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion,
        "tok_s": round(completion / dt, 2) if dt else 0.0,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "sample": content[:160].replace("\n", " "),
    }


def warmup() -> None:
    print("warmup…", flush=True)
    chat(16, "Reply exactly: WARM", 0)
    print("warmup done", flush=True)


def bench_single() -> list[dict]:
    """Forced-length single-stream (ignore_eos) — matches Tony's 512/1024/2048 tables."""
    rows = []
    essay = (
        "Write a very long technical essay. Never stop early. "
        "Continue with new numbered sections forever. Do not conclude."
    )
    for n in SINGLE_LENS:
        print(f"single-stream max_tokens={n} (ignore_eos) …", flush=True)
        before = metrics_snapshot()
        r = chat(n, essay, 1, ignore_eos=True)
        after = metrics_snapshot()
        r["acceptance"] = acceptance(before, after)
        r["max_tokens"] = n
        print(
            f"  → {r['completion_tokens']} toks in {r['seconds']}s = "
            f"{r['tok_s']} tok/s  accept={r['acceptance']}",
            flush=True,
        )
        rows.append(r)
    return rows


def bench_concurrency() -> list[dict]:
    results = []
    for c in CONCURRENCY_LIST:
        print(f"concurrency={c} max_tokens={CONCUR_MAX_TOKENS} …", flush=True)
        before = metrics_snapshot()
        wall0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            futs = [
                ex.submit(chat, CONCUR_MAX_TOKENS, make_prompt(i), i) for i in range(c)
            ]
            rows = [f.result() for f in futs]
        wall = time.perf_counter() - wall0
        after = metrics_snapshot()
        total = sum(r["completion_tokens"] for r in rows)
        agg = round(total / wall, 2) if wall else 0.0
        per = round(statistics.mean(r["tok_s"] for r in rows), 2)
        result = {
            "concurrency": c,
            "max_tokens": CONCUR_MAX_TOKENS,
            "wall_seconds": round(wall, 3),
            "completion_tokens": total,
            "aggregate_tok_s": agg,
            "per_stream_tok_s_mean": per,
            "per_stream_derived": round(agg / c, 2) if c else 0.0,
            "acceptance": acceptance(before, after),
            "rows": rows,
        }
        print(
            f"  → agg {agg} tok/s  derived/stream {result['per_stream_derived']}  "
            f"accept={result['acceptance']}",
            flush=True,
        )
        results.append(result)
    return results


def main() -> int:
    # connectivity
    try:
        models = _get_text(BASE_URL + "/models")
    except urllib.error.URLError as e:
        print(f"ERROR: cannot reach {BASE_URL}: {e}", file=sys.stderr)
        return 2
    if MODEL not in models:
        print(f"ERROR: model {MODEL} not in /v1/models", file=sys.stderr)
        return 2

    warmup()
    single = bench_single()
    concur = bench_concurrency()

    report = {
        "when": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "model": MODEL,
        "shape_note": "server defaults: max_num_seqs=3, GMU=0.83, max_model_len=1M, MTP1, nvfp4 KV",
        "single_stream": single,
        "concurrency": [
            {k: v for k, v in r.items() if k != "rows"} | {"detail": r["rows"]}
            for r in concur
        ],
    }
    # cleaner concurrency without huge nesting for print
    report_print = {
        **report,
        "concurrency": [{k: v for k, v in r.items() if k != "rows"} for r in concur],
    }
    print(json.dumps(report_print, indent=2), flush=True)

    out = OUT_JSON or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"perf_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
