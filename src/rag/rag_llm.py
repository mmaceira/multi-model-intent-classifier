"""
Simplified & safer Rag‑LLM classifier with rate‑limiting & batched LLM calls
==========================================================================

Key improvements:
  * **Rate limit control**: `rate_limit_per_minute` setting with min‑interval sleeps
  * **Batched LLM calls**: classify multiple docs per request to reduce call count
  * Maintains robust retry logic & clear fallbacks

Created: 2025‑05‑04
Author: ChatGPT
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

from .classifier_base import RagClassifierBase
from .retrieval import Retriever
from .vector_store import VectorStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in environment")

_PROMPT_SYSTEM = """You are a news-topic classifier. You MUST respond with ONLY a valid JSON object containing a 'labels' array.
For example: {"labels": ["earn", "acq", "grain"]}
Do NOT use code blocks, markdown, or explanations. Return ONLY valid JSON."""


def _exponential_backoff(attempt: int) -> float:
    """Exponential back‑off with jitter, capped at 30s."""
    delay = min((2 ** attempt) + random.random(), 30.0)
    return delay

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class RagLLM(RagClassifierBase):
    """RAG‑style news classifier with batching & rate‑limiting."""

    def __init__(
        self,
        retriever: Retriever,
        labels: Sequence[str],
        *,
        model: str = "gpt-4o-mini",
        top_k: int = 5,
        batch_size: int = 8,
        max_concurrency: int = 2,
        rate_limit_per_minute: int = 120,
        embedder: Callable[[Sequence[str]], Iterable[Sequence[float]]] | None = None,
        max_retries: int = 5,
    ):
        super().__init__(labels)
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.batch_size = batch_size
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = AsyncOpenAI(api_key=openai_api_key)
        self.embedder = embedder
        self.max_retries = max_retries
        # compute minimum interval between calls
        self._min_interval = 60.0 / rate_limit_per_minute

    @classmethod
    def load_default(cls, *, use_openai: bool = False, **kwargs):
        retriever = Retriever.from_default(use_openai=use_openai)
        labels = sorted({m['label'] for m in retriever.store.meta})
        return cls(retriever, labels, **kwargs)

    def _embed(self, docs: Sequence[str]) -> np.ndarray:
        if self.embedder is None:
            return VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", docs)
        if hasattr(self.embedder, "encode"):
            return np.array(self.embedder.encode(list(docs)), dtype="float32")
        return np.array(self.embedder(list(docs)), dtype="float32")

    async def _chat_with_retry(self, messages) -> openai.chat.completion.ChatCompletion:
        last_request_time = time.time()
        for attempt in range(self.max_retries):
            try:
                # Calculate time since last request and wait if needed
                now = time.time()
                time_since_last = now - last_request_time
                if time_since_last < self._min_interval:
                    wait_time = max(0, self._min_interval - time_since_last)
                    await asyncio.sleep(wait_time)
                
                # Update last request time
                last_request_time = time.time()
                
                # Make the API call with JSON response format
                return await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
            except RateLimitError as exc:
                wait = _exponential_backoff(attempt)
                logger.warning("Rate‑limit hit: waiting %.1fs (attempt %d/%d)", wait, attempt+1, self.max_retries)
                await asyncio.sleep(wait)
            except Exception as e:
                wait = _exponential_backoff(attempt)
                logger.error(f"API error: {str(e)} - waiting {wait:.1f}s (attempt {attempt+1}/{self.max_retries})")
                await asyncio.sleep(wait)
                
        raise RuntimeError("OpenAI API failed after retries")

    async def _classify_batch(
        self, docs: List[str], vectors: List[np.ndarray]
    ) -> List[str]:
        # prepare contexts
        contexts: List[str] = []
        for vec in vectors:
            neigh = self.retriever.top_k(vec[None,:], self.top_k)
            contexts.append("\n\n".join(n['text'] for n in neigh))

        # build a single prompt with multiple items
        allowed = ", ".join(self.labels)
        lines = []
        for i,(doc,ctx) in enumerate(zip(docs, contexts), start=1):
            lines.append(f"{i}. Article: {doc}\nContext:\n{ctx}")
        user_content = (
            f"Classify these news articles into ONE of these categories: {allowed}\n\n"
            + "\n\n".join(lines)
            + "\n\nRespond with a JSON object that has a 'labels' property containing an array of labels: {'labels': ['label1', 'label2']}. "
            + "No explanation, no formatting, just JSON."
        )
        messages = [
            {"role": "system", "content": _PROMPT_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        # Add a specific response format instruction
        messages.append({
            "role": "assistant", 
            "content": "I'll respond with only a JSON object: {\"labels\": [\"label1\", \"label2\"]}"
        })

        resp = await self._chat_with_retry(messages)
        text = resp.choices[0].message.content.strip()
        
        # Improved JSON parsing with better error handling
        try:
            import json
            response_data = json.loads(text)
            
            # Check if the response has a 'labels' property
            if isinstance(response_data, dict) and 'labels' in response_data:
                labels_out = response_data['labels']
            # Fallback to assuming the entire object is the array
            elif isinstance(response_data, list):
                labels_out = response_data
            # Handle case when response doesn't match expected format
            else:
                # Try to find any array in the response
                for key, value in response_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        labels_out = value
                        logger.warning(f"Using array found at key '{key}' instead of 'labels'")
                        break
                else:
                    logger.error(f"No array found in response: {response_data}")
                    return [self.labels[0]] * len(docs)
            
            if isinstance(labels_out, list):
                # Handle case where number of labels doesn't match docs
                if len(labels_out) != len(docs):
                    logger.warning(f"Expected {len(docs)} labels but got {len(labels_out)}. Adjusting...")
                    # Extend with first label if too short
                    if len(labels_out) < len(docs):
                        labels_out.extend([labels_out[0] if labels_out else self.labels[0]] * (len(docs) - len(labels_out)))
                    # Truncate if too long
                    else:
                        labels_out = labels_out[:len(docs)]
                
                # Validate that all labels are in allowed list
                return [lab if lab in self.labels else self.labels[0] for lab in labels_out]
            else:
                logger.error(f"Expected list but got {type(labels_out)}: {labels_out}")
                return [self.labels[0]] * len(docs)
        except json.JSONDecodeError as e:
            logger.error(f"Failed parsing JSON batch response: {text} | Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error handling batch response: {text} | Error: {e}")
            
        # fallback: label all with first
        return [self.labels[0]] * len(docs)

    def predict(self, docs: Sequence[str]) -> List[str]:
        async def _run_all() -> List[str]:
            logger.info(f"Starting prediction for {len(docs)} documents with batch size {self.batch_size}")
            embeddings = self._embed(docs)
            batches = [docs[i:i+self.batch_size] for i in range(0,len(docs),self.batch_size)]
            vec_batches = [embeddings[i:i+self.batch_size] for i in range(0,len(docs),self.batch_size)]
            logger.info(f"Split into {len(batches)} batches")
            results: List[str] = []
            for i, (docs_batch,vec_batch) in enumerate(zip(batches,vec_batches), 1):
                logger.info(f"Processing batch {i}/{len(batches)} with {len(docs_batch)} documents")
                async with self._sem:
                    preds = await self._classify_batch(docs_batch, list(vec_batch))
                results.extend(preds)
            return results

        # run async
        try:
            # Always try the simple way first
            return asyncio.run(_run_all())
        except RuntimeError:          # "event loop is already running"
            import nest_asyncio, concurrent.futures
            
            nest_asyncio.apply()      # ↯ *patches* the running loop
            # off‑load to a worker thread so .result() won't dead‑lock
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(
                    lambda: asyncio.run(_run_all())
                ).result()
