
"""FAISS vector store wrapper."""
import json
from pathlib import Path
from typing import List, Sequence
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, faiss_path: Path, meta_path: Path):
        self.index = faiss.read_index(str(faiss_path))
        self.meta: List[dict] = [json.loads(l) for l in Path(meta_path).read_text().splitlines()]

    @staticmethod
    def build(emb: np.ndarray, meta: List[dict], dim: int, faiss_path: Path, meta_path: Path):
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(emb)
        index.add(emb.astype('float32'))
        faiss.write_index(index, str(faiss_path))
        meta_path.write_text('\n'.join(json.dumps(m) for m in meta))

    def search(self, q: np.ndarray, k: int) -> List[dict]:
        faiss.normalize_L2(q)
        _, idx = self.index.search(q.astype('float32'), k)
        return [self.meta[i] for i in idx[0]]

    @staticmethod
    def embed(model_name: str, docs: Sequence[str]) -> np.ndarray:
        model = SentenceTransformer(model_name)
        return model.encode(list(docs), batch_size=32, show_progress_bar=False).astype('float32')
