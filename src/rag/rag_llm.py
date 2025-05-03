"""LLM classifier (requires OPENAI_API_KEY)."""
import os
from typing import Sequence
import openai
from .classifier_base import RagClassifierBase
from .retrieval import Retriever
from .vector_store import VectorStore
from dotenv import load_dotenv  # Import the load_dotenv function
import asyncio

_PROMPT = ("You are a news‑topic classifier. Choose one label from the list on the first line.")

# Load environment variables from the .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found! Please ensure it is set in the .env file.")
else:
    print("OPENAI_API_KEY loaded successfully.")

class RagLLM(RagClassifierBase):
    def __init__(self, retriever: Retriever, labels: Sequence[str], model='gpt-4o-mini', top_k=5, batch_size=8):
        super().__init__(labels)
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.batch_size = batch_size
        openai.api_key = openai_api_key

    @classmethod
    def load_default(cls, **cfg):
        retriever = Retriever.from_default()
        labels = {m['label'] for m in retriever.store.meta}
        return cls(retriever, sorted(labels), **cfg)

    async def _predict_batch_async(self, docs: Sequence[str], **_):
        from openai import AsyncOpenAI
        emb = VectorStore.embed("sentence-transformers/all-MiniLM-L6-v2", docs)
        client = AsyncOpenAI(api_key=openai_api_key)
        tasks = []
        for doc, vec in zip(docs, emb):
            neigh = self.retriever.top_k(vec[None,:], self.top_k)
            label_list = ", ".join(self.labels)
            ctx = "\n\n".join([n['text'] for n in neigh])
            msg = f"{_PROMPT}\nAllowed: {label_list}\n\nArticle: {doc}\n\nContext:\n{ctx}"
            tasks.append(client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': _PROMPT},
                    {'role': 'user', 'content': msg}
                ],
                temperature=0.0
            ))
        responses = await asyncio.gather(*tasks)
        return [resp.choices[0].message.content.splitlines()[0].strip() for resp in responses]

    def predict(self, docs: Sequence[str], **_):
        # Batch the docs for async processing
        batch_size = self.batch_size
        all_preds = []
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
        for batch in batches:
            if loop and loop.is_running():
                # If already in an event loop (e.g. FastAPI), use create_task
                preds = asyncio.run_coroutine_threadsafe(self._predict_batch_async(batch, **_), loop).result()
            else:
                preds = asyncio.run(self._predict_batch_async(batch, **_))
            all_preds.extend(preds)
        return all_preds
