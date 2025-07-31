#!/usr/bin/env python3
"""
Batch test language model on WinoGrande dataset from Hugging Face.

Key features:
- Multiple samples per question, saves completions to outputs
- Rate-limit aware with automatic retry on HTTP 429
- Configurable output length via max-tokens parameter
- Deterministic seeding for reproducible results
- Async/parallel processing for efficiency
- Rich logging with debug options

Usage:
    python run_winogrande.py --data winogrande_xs --out outputs --samples 30
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
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio as tqdm # type: ignore
from datasets import load_dataset

# ------------------------ CONSTANTS ------------------------
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BACKOFF_BASE = 1.8 # exponential back-off base seconds
DEFAULT_SLEEP_ON_429 = 60 # seconds to pause when RPM quota is hit
logger = logging.getLogger("qwen_winogrande")

# ------------------------ PROMPT & DATA PROCESSING ------------------------
def process_sample(item: dict) -> list:
    """
    Process data sample to create question-answering prompt.
    Takes WinoGrande-style problem and formats it for conversational AI.

    Args:
        item: Dictionary with keys 'sentence', 'option1', 'option2'.

    Returns:
        List of dictionaries for the prompt, containing system and user messages.
    """
    # Generate random step count for reasoning
    rand_int = random.randint(3, 15)
    system_prompt = f"Please reason {rand_int} steps and complete the following sentence by choosing the most logical option to fill in the blank."

    user_prompt = f"""
    Sentence: "{item['sentence']}"
    Options:
    1. {item['option1']}
    2. {item['option2']}
    Which option is the correct choice?
    """
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

    return messages

# ------------------------ API CALL ------------------------
async def call_qwen(
    api_key: str,
    messages: List[Dict[str, Any]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    max_retries: int,
    debug_prefix: str = "",
    ) -> str:
    """Call language model API with retry logic and logging.
    Returns raw completion text with detailed logging of attempts and timing.
    """
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )

    for attempt in range(1, max_retries + 1):
        start_ts = time.perf_counter()
        try:
            logger.debug("%sAttempt %d POST", debug_prefix, attempt)
            
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            
            dur = time.perf_counter() - start_ts
            response_text = completion.choices[0].message.content
            
            logger.debug(
                "%s 200 OK in %.2fs (tokens~%d)",
                debug_prefix,
                dur,
                len(response_text or ""),
            )
            return response_text or ""
            
        except Exception as e:
            dur = time.perf_counter() - start_ts
            if "rate limit" in str(e).lower() or "429" in str(e):
                logger.warning(
                    "%s Rate limit hit. Sleeping %ds before retrying",
                    debug_prefix,
                    DEFAULT_SLEEP_ON_429,
                )
                await asyncio.sleep(DEFAULT_SLEEP_ON_429)
                continue
            elif "server error" in str(e).lower() or any(str(code) in str(e) for code in [500, 502, 503, 504]):
                backoff = BACKOFF_BASE ** attempt + random.random()
                logger.warning(
                    "%s Server error. Backing off %.1fs then retry",
                    debug_prefix,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue
            elif attempt == max_retries:
                logger.error("%s %s (final attempt)", debug_prefix, type(e).__name__)
                raise RuntimeError("API call exceeded retry budget") from e
            else:
                backoff = BACKOFF_BASE ** attempt + random.random()
                logger.warning(
                    "%s %s on attempt %d. Backing off %.1fs then retry",
                    debug_prefix,
                    type(e).__name__,
                    attempt,
                    backoff,
                )
                await asyncio.sleep(backoff)

    raise RuntimeError("Unreachable - exceeded retries")

# ------------------------ OUTPUT PROCESSING ------------------------
def extract_final_answer(text: str) -> str:
    """
    Extract final answer ('1' or '2') from model response.
    Finds last occurrence of '1' or '2' in text after 'option'.
    """
    processed_text = text.split("option")[-1]
    last_pos_1 = processed_text.rfind('1')
    last_pos_2 = processed_text.rfind('2')

    if last_pos_1 == -1 and last_pos_2 == -1:
        return "" # Neither '1' nor '2' found

    if last_pos_1 > last_pos_2:
        return "1"
    else:
        return "2"

# --------------------- PER-QUESTION EVAL --------------------
async def evaluate_question(idx: int, item: Dict[str, Any], args) -> bool:
    """Evaluate question with multiple samples, return True if any sample is correct."""
    messages = process_sample(item)
    gt = str(item.get("answer", "")).strip()
    qid = item.get("qID", f"q{idx}") # Use qID if available, otherwise use index
    out_file = Path(args.out) / Path(args.data) / Path(args.model) / f"{qid}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    samples: List[str] = []
    found = asyncio.Event()

    async def worker(sample_idx: int):
        nonlocal samples
        async with sem:
            prefix = f"Q{idx}-S{sample_idx}: " if args.debug else ""
            txt = await call_qwen(
                args.api_key,
                messages,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                seed=args.seed + sample_idx,  # decorrelate seeds per sample
                max_retries=args.max_retries,
                debug_prefix=prefix,
            )
        samples.append(txt)
        ans = extract_final_answer(txt)
        logger.debug("Q%d-S%d => Extracted: '%s', Ground Truth: '%s'", idx, sample_idx, ans, gt)
        if not found.is_set() and ans == gt:
            found.set()

    tasks = [asyncio.create_task(worker(i)) for i in range(args.samples)]
    await asyncio.gather(*tasks, return_exceptions=True)  # keep all texts even if some raise

    # Create a serializable version of the item for JSON dump
    serializable_item = {k: v for k, v in item.items()}
    output_data = {
        "item": serializable_item,
        "ground_truth": gt,
        "samples": samples
    }
    out_file.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2)
    )
    logger.info("Saved %s (%d samples)", out_file, len(samples))
    return found.is_set()

# ------------------------ MAIN DRIVER ----------------------
async def main_async(args):
    random.seed(args.seed)
    # Load WinoGrande dataset from Hugging Face
    try:
        dataset = load_dataset("winogrande", args.data, split="train")
        data = list(dataset) # Convert to list for tqdm and easier handling
        logger.info("Loaded %d problems.", len(data))
    except Exception as e:
        logger.critical("Failed to load dataset from Hugging Face: %s", e)
        sys.exit(1)

    total, correct = len(data), 0
    progress_bar = tqdm(enumerate(data), total=total, desc="Questions")

    for idx, item in progress_bar:
        try:
            ok = await evaluate_question(idx, item, args)
        except Exception as e:
            tqdm.write(f"[Error] Q{idx}: {e}")
            logger.exception("Unhandled error in Q%d", idx)
            ok = False
        
        if ok:
            correct += 1
        
        accuracy = correct / (idx + 1)
        progress_bar.set_postfix_str(f"Acc: {correct}/{idx + 1} = {accuracy:.3%}")

    print("\n======== FINAL RESULT =======")
    print(f"Accuracy: {correct}/{total} = {correct/total:.3%}")

# ------------------------ ENTRYPOINT -----------------------
def parse_args():
    p = argparse.ArgumentParser(
        "Language model evaluation on WinoGrande",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="winogrande_xs", help="Hugging Face dataset name for WinoGrande")
    p.add_argument("--out", default="outputs", help="Directory to save per-question JSON logs")
    p.add_argument("--model", default="qwen2.5-72b-instruct", help="Model name")
    p.add_argument("--samples", type=int, default=30, help="Samples per question")
    p.add_argument("--concurrency", type=int, default=20, help="Concurrent API calls")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, dest="max_tokens", default=512)
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument("--max-retries", type=int, default=10)
    p.add_argument("--api-key", type=str, default="", help="API key (or env var ALIYUN_API_KEY)")
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
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file, mode="w"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=log_handlers,
    )

    if args.debug:
        logger.info("Debug logging enabled")
    if not args.api_key:
        args.api_key = os.getenv("ALIYUN_API_KEY", "")
    if not args.api_key:
        logger.critical("Provide API key via --api-key or env var ALIYUN_API_KEY")
        sys.exit(1)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user.")