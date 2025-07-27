#!/usr/bin/env python3
"""
Batch‑test Alibaba Cloud **Qwen 2.5** on the **MATH500** dataset stored as **JSONL** (one JSON object per line).

Key features
============
* 💯 **samples** per question, every completion saved to `outputs/<qid>.json`.
* **Rate‑limit aware** – sleeps 60 s on HTTP 429 before retrying.
* **Output control** – `--max-tokens` caps assistant response length.
* **Deterministic** – `--seed` sets both Python RNG *and* Qwen parameter.
* Fully **async/parallel** via `asyncio` + `aiohttp`.
* 🪵 **Rich, configurable logging** – use `--debug` and/or `--log-file` to get detailed
  per‑request diagnostics including retries, back‑offs, sleeps, and timings.

Quick start
-----------
```bash
pip install aiohttp tqdm python-dotenv
export DASHSCOPE_API_KEY="<your‑key>"
python run.py \
       --data test.jsonl \
       --out outputs \
       --samples 100 \
       --max-tokens 512 \
       --seed 42 \
       --concurrency 20 \
       --debug                # optional: turn on verbose console logging
```
The script streams progress and prints the final accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio as tqdm  # type: ignore

# ------------------------  CONSTANTS  ------------------------ #
ENDPOINT = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
HEADERS_BASE = {"Content-Type": "application/json"}
BACKOFF_BASE = 1.8  # exponential back‑off base seconds
DEFAULT_SLEEP_ON_429 = 60  # seconds to pause when RPM quota is hit

logger = logging.getLogger("qwen_math500")

# ------------------------  API CALL  ------------------------ #


async def call_qwen(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    max_retries: int,
    debug_prefix: str = "",
) -> str:
    """Low‑level helper – returns raw assistant completion text.

    Adds rich logging of every attempt, HTTP code, back‑off sleep, and total RT.
    """

    payload = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {
            "temperature": temperature,
            "top_p": 0.85,
            "n": 1,
            "max_tokens": max_tokens,
            "seed": seed,
        },
    }
    headers = HEADERS_BASE | {"Authorization": f"Bearer {api_key}"}

    for attempt in range(1, max_retries + 1):
        start_ts = time.perf_counter()
        try:
            logger.debug("%sAttempt %d POST …", debug_prefix, attempt)
            async with session.post(
                ENDPOINT, headers=headers, json=payload, timeout=120
            ) as resp:
                dur = time.perf_counter() - start_ts
                if resp.status == 200:
                    data: Dict[str, Any] = await resp.json()
                    logger.debug(
                        "%s✅ 200 OK in %.2fs (tokens≈%d)",
                        debug_prefix,
                        dur,
                        len(data.get("output", {}).get("text", "")),
                    )
                    return data["output"]["text"]  # type: ignore[index]
                # --- recoverable errors ---
                if resp.status == 429:  # RPM limit – take a nap
                    logger.warning(
                        "%s⚠️ 429 Rate‑limit. Sleeping %ds before retrying…",
                        debug_prefix,
                        DEFAULT_SLEEP_ON_429,
                    )
                    await asyncio.sleep(DEFAULT_SLEEP_ON_429)
                    continue
                if resp.status in {500, 502, 503, 504}:
                    backoff = BACKOFF_BASE ** attempt + random.random()
                    logger.warning(
                        "%s⚠️ %d Server error. Back‑off %.1fs then retry…",
                        debug_prefix,
                        resp.status,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue
                # --- other HTTP errors are fatal ---
                msg = await resp.text()
                logger.error("%s❌ HTTP %d: %s", debug_prefix, resp.status, msg)
                raise RuntimeError(f"HTTP {resp.status}: {msg}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries:
                logger.error("%s❌ %s (final attempt)", debug_prefix, type(e).__name__)
                raise RuntimeError("Qwen call exceeded retry budget") from e
            backoff = BACKOFF_BASE ** attempt + random.random()
            logger.warning(
                "%s⚠️ %s on attempt %d. Back‑off %.1fs then retry…",
                debug_prefix,
                type(e).__name__,
                attempt,
                backoff,
            )
            await asyncio.sleep(backoff)
    raise RuntimeError("Unreachable – exceeded retries")


# ------------------------  DATA I/O  ------------------------ #


def load_math500(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file -> list[dict]. Raises if any line fails to parse."""

    items: List[Dict[str, Any]] = []
    system_prompt="""Please think step by step and place the final answer in \boxed{answer}."""
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error at line {lineno}: {e}") from None
            if not isinstance(obj, dict):
                raise ValueError(f"Line {lineno} is not a JSON object")
            items.append(obj)
    logger.info("Loaded %d problems from %s", len(items), path)
    return items


import re

ANSWER_RE = re.compile(r"(-?\\d+\\.?\\d*|\\[.*?\\]|[A-Za-z]+)")


def extract_final_answer(text: str) -> str:
    markers = ["####", "Answer:"]
    for m in markers:
        if m in text:
            tail = text.split(m)[-1]
            z = ANSWER_RE.search(tail)
            if z:
                return z.group(0).strip()
    toks = ANSWER_RE.findall(text)
    return toks[-1].strip() if toks else text.strip()


# ---------------------  PER‑QUESTION EVAL  -------------------- #


async def evaluate_question(idx: int, item: Dict[str, Any], args) -> bool:
    """Return True if any of the N samples is correct. Also dump all samples."""

    prompt = item.get("problem")
    if prompt is None:
        raise KeyError(f"Missing 'problem' in item {idx}")
    gt = str(item.get("answer", "")).strip()
    qid = item.get("id", idx)
    out_file = Path(args.out) / f"{qid}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(args.concurrency)
        samples: List[str] = []
        found = asyncio.Event()

        async def worker(sample_idx: int):
            nonlocal samples
            async with sem:
                prefix = f"Q{idx}‑S{sample_idx}: " if args.debug else ""
                txt = await call_qwen(
                    session,
                    args.api_key,
                    prompt,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    seed=args.seed + sample_idx,  # decorrelate seeds per sample
                    max_retries=args.max_retries,
                    debug_prefix=prefix,
                )
            samples.append(txt)
            ans = extract_final_answer(txt)
            logger.debug("Q%d‑S%d ⇒ %s", idx, sample_idx, ans)
            if not found.is_set() and ans == gt:
                found.set()

        tasks = [asyncio.create_task(worker(i)) for i in range(args.samples)]
        await asyncio.gather(*tasks, return_exceptions=True)  # keep all texts even if some raise

    out_file.write_text(
        json.dumps({"problem": prompt, "answer": gt, "samples": samples}, ensure_ascii=False, indent=2)
    )
    logger.info("Saved %s (%d samples)", out_file, len(samples))
    return found.is_set()


# ------------------------  MAIN DRIVER  ---------------------- #


async def main_async(args):
    random.seed(args.seed)
    data = load_math500(Path(args.data))
    total, correct = len(data), 0
    for idx, item in enumerate(tqdm(data, desc="Questions")):
        try:
            ok = await evaluate_question(idx, item, args)
        except Exception as e:
            tqdm.write(f"[Error] Q{idx}: {e}")
            logger.exception("Unhandled error in Q%d", idx)
            ok = False
        correct += ok
        tqdm.write(
            f"Q{idx+1}/{total} {'✔' if ok else '✘'} | Acc: {correct}/{idx+1} = {correct/(idx+1):.3%}"
        )
    print("\n======== FINAL RESULT =======")
    print(f"Accuracy: {correct}/{total} = {correct/total:.3%}")


# ------------------------  ENTRYPOINT  ----------------------- #


def parse_args():
    p = argparse.ArgumentParser(
        "Qwen 2.5 evaluation on MATH500 JSONL (full logging)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="test.jsonl", help="Path to JSONL questions file")
    p.add_argument("--out", default="outputs", help="Directory to save per‑question JSON logs")
    p.add_argument("--model", default="qwen2.5-7b-instruct", help="Model name on DashScope")
    p.add_argument("--samples", type=int, default=100, help="Samples per question")
    p.add_argument("--concurrency", type=int, default=20, help="Concurrent API calls")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, dest="max_tokens", default=8192)
    p.add_argument("--seed", type=int, default=514, help="Global random seed")
    p.add_argument("--max-retries", type=int, default=10)
    p.add_argument("--api-key", type=str, default="", help="DashScope API key (or env var)")
    p.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    p.add_argument(
        "--log-file", type=str, default="", help="Optional path to write log file"
    )
    return p.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()

    # --- logging config ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *( [logging.FileHandler(args.log_file, mode="w")] if args.log_file else [] ),
        ],
    )
    if args.debug:
        logger.info("Debug logging enabled")
    if not args.api_key:
        args.api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not args.api_key:
        logger.critical("Provide API key via --api-key or env DASHSCOPE_API_KEY")
        sys.exit(1)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
