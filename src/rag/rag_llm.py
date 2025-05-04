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

_PROMPT_SYSTEM = "You are a news‑topic classifier. Respond with a JSON list of labels."


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
        batch_size: int = 16,
        max_concurrency: int = 4,
        rate_limit_per_minute: int = 240,
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
        for attempt in range(self.max_retries):
            try:
                # rate limit sleep
                await asyncio.sleep(self._min_interval)
                return await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
            except RateLimitError as exc:
                wait = _exponential_backoff(attempt)
                logger.warning("Rate‑limit hit: waiting %.1fs (attempt %d/%d)", wait, attempt+1, self.max_retries)
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
            f"Allowed labels: {allowed}\n\n"
            + "\n\n".join(lines)
            + "\n\nRespond with a JSON array of labels in order."
        )
        messages = [
            {"role": "system", "content": _PROMPT_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        resp = await self._chat_with_retry(messages)
        text = resp.choices[0].message.content.strip()
        try:
            labels_out = __import__('json').loads(text)
            if isinstance(labels_out, list) and len(labels_out)==len(docs):
                return [lab if lab in self.labels else self.labels[0] for lab in labels_out]
        except Exception:
            logger.error("Failed parsing JSON batch response: %s", text)
        # fallback: label all with first
        return [self.labels[0]] * len(docs)

    def predict(self, docs: Sequence[str]) -> List[str]:
        async def _run_all() -> List[str]:
            embeddings = self._embed(docs)
            batches = [docs[i:i+self.batch_size] for i in range(0,len(docs),self.batch_size)]
            vec_batches = [embeddings[i:i+self.batch_size] for i in range(0,len(docs),self.batch_size)]
            results: List[str] = []
            for docs_batch,vec_batch in zip(batches,vec_batches):
                async with self._sem:
                    preds = await self._classify_batch(docs_batch, list(vec_batch))
                results.extend(preds)
            return results

        # run async
        try:
            return asyncio.run(_run_all())
        except RuntimeError:
            import nest_asyncio, concurrent.futures
            nest_asyncio.apply()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(lambda: asyncio.run(_run_all())).result()
