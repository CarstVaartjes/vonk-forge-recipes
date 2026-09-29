#!/usr/bin/env python3
"""Closed-loop concurrency sweep for Puzzle catch-up vs jvr0x numbers.

Workload mirrors jvr0x harness defaults: ~1024 prompt / 256 output, unique
prefix per request (defeats prefix cache), streaming TTFT + pure decode tok/s.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from typing import Any

import httpx

# jvr0x published (results/nemotron-labs-3-puzzle-75b-nvfp4-aeon7.json)
JVR = {
    1: {"agg": 31.3, "sess": 32.8, "ttft_p95": 384},
    8: {"agg": 129.4, "sess": 17.3, "ttft_p95": 944},
    16: {"agg": 183.5, "sess": 12.5, "ttft_p95": 1742},
    32: {"agg": 260.3, "sess": 8.3, "ttft_p95": 6479},
    64: {"agg": 330.0, "sess": 5.7, "ttft_p95": 13398},
}


def make_prompt(target_tokens: int = 1024) -> str:
    # ~1 token / word-ish; pad with unique material so cache cannot hit
    uid = uuid.uuid4().hex
    unit = f"token{uid[:8]} the quick brown fox jumps over the lazy dog. "
    # ~9-10 tokens per unit typically for English-ish filler
    n = max(1, target_tokens // 9)
    return (
        f"[{uid}] Summarize the following text in one short sentence. "
        f"Ignore uniqueness markers.\n\n" + (unit * n)
    )


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        # force length when backend honors these
        "ignore_eos": True,
        "min_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    t_first = None
    usage = None
    n_chars = 0
    async with client.stream("POST", url, json=body, timeout=timeout) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            c = delta.get("content") or ""
            if c and t_first is None:
                t_first = time.perf_counter()
            n_chars += len(c)
    t1 = time.perf_counter()
    completion = (usage or {}).get("completion_tokens") or 0
    prompt_tok = (usage or {}).get("prompt_tokens") or 0
    pure = None
    if t_first and completion > 1:
        pure = (completion - 1) / (t1 - t_first)
    return {
        "ttft_s": (t_first - t0) if t_first else None,
        "elapsed_s": t1 - t0,
        "completion_tokens": completion,
        "prompt_tokens": prompt_tok,
        "pure_tok_s": pure,
        "chars": n_chars,
    }


async def session_loop(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt_tokens: int,
    output_tokens: int,
    end_at: float,
    timeout: float,
    results: list,
) -> None:
    while time.perf_counter() < end_at:
        prompt = make_prompt(prompt_tokens)
        try:
            r = await one_request(client, url, model, prompt, output_tokens, timeout)
            results.append(r)
        except Exception as e:  # noqa: BLE001
            results.append({"error": str(e), "ttft_s": None, "pure_tok_s": None, "completion_tokens": 0})


async def run_level(
    base_url: str,
    model: str,
    n: int,
    warmup_s: float,
    measure_s: float,
    prompt_tokens: int,
    output_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    results: list[dict[str, Any]] = []
    limits = httpx.Limits(max_connections=n + 4, max_keepalive_connections=n + 4)
    async with httpx.AsyncClient(limits=limits) as client:
        # warmup
        w_end = time.perf_counter() + warmup_s
        wres: list = []
        tasks = [
            asyncio.create_task(
                session_loop(client, url, model, prompt_tokens, output_tokens, w_end, timeout, wres)
            )
            for _ in range(n)
        ]
        await asyncio.gather(*tasks)

        # measure
        results.clear()
        m_end = time.perf_counter() + measure_s
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                session_loop(client, url, model, prompt_tokens, output_tokens, m_end, timeout, results)
            )
            for _ in range(n)
        ]
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    ok = [r for r in results if r.get("pure_tok_s") and not r.get("error")]
    errs = [r for r in results if r.get("error")]
    total_completion = sum(r.get("completion_tokens") or 0 for r in ok)
    # aggregate = total completion tokens / wall (jvr closed-loop style)
    agg = total_completion / wall if wall > 0 else 0.0
    pure_list = [r["pure_tok_s"] for r in ok if r.get("pure_tok_s")]
    ttfts = [r["ttft_s"] * 1000 for r in ok if r.get("ttft_s") is not None]
    sess = statistics.mean(pure_list) if pure_list else 0.0

    def pct(xs: list[float], p: float) -> float | None:
        if not xs:
            return None
        xs = sorted(xs)
        i = min(len(xs) - 1, max(0, int(round((p / 100) * (len(xs) - 1)))))
        return xs[i]

    return {
        "concurrency": n,
        "agg_tok_s": round(agg, 1),
        "per_session_tok_s": round(sess, 1),
        "ttft_ms": {
            "p50": round(pct(ttfts, 50) or 0, 0),
            "p95": round(pct(ttfts, 95) or 0, 0),
            "p99": round(pct(ttfts, 99) or 0, 0),
        },
        "samples": len(ok),
        "errors": len(errs),
        "wall_s": round(wall, 1),
        "total_completion_tokens": total_completion,
        "jvr": JVR.get(n),
    }


async def amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.100.10.1:8243/v1")
    ap.add_argument("--model", default="puzzle-jvr")
    ap.add_argument("--levels", default="1,16,32", help="comma concurrencies")
    ap.add_argument("--warmup-s", type=float, default=20)
    ap.add_argument("--measure-s", type=float, default=90)
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("-o", "--out", default="/tmp/puzzle-jvr-catchup-bench.json")
    args = ap.parse_args()

    # readiness
    async with httpx.AsyncClient() as c:
        for i in range(180):
            try:
                r = await c.get(args.base_url.rstrip("/") + "/models", timeout=5)
                if r.status_code == 200:
                    print("ready", r.text[:120])
                    break
            except Exception:
                pass
            await asyncio.sleep(5)
        else:
            raise SystemExit("server not ready")

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    out = {
        "meta": {
            "base_url": args.base_url,
            "model": args.model,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "warmup_s": args.warmup_s,
            "measure_s": args.measure_s,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "catch-up vs jvr0x aeon7 recipe",
        },
        "points": [],
    }
    print(f"{'N':>4} {'agg':>8} {'sess':>8} {'ttft_p95':>10}  | jvr agg/sess/p95")
    for n in levels:
        print(f"\n=== N={n} warmup={args.warmup_s}s measure={args.measure_s}s ===", flush=True)
        p = await run_level(
            args.base_url,
            args.model,
            n,
            args.warmup_s,
            args.measure_s,
            args.prompt_tokens,
            args.output_tokens,
            args.timeout,
        )
        out["points"].append(p)
        j = p.get("jvr") or {}
        print(
            f"{p['concurrency']:4d} {p['agg_tok_s']:8.1f} {p['per_session_tok_s']:8.1f} "
            f"{p['ttft_ms']['p95']:10.0f}  | {j.get('agg','?'):>5} / {j.get('sess','?'):>5} / {j.get('ttft_p95','?'):>5}  "
            f"samples={p['samples']} err={p['errors']}",
            flush=True,
        )
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print("\nWrote", args.out)


if __name__ == "__main__":
    asyncio.run(amain())
