
"""search.py – Simple semantic search helper."""
import numpy as np
from typing import List, Tuple

def build_index(texts: List[str], embedder, batch_size: int = 64):
    """Return embeddings numpy array of shape (N, dim)."""
    return embedder.encode(texts, show_progress_bar=True, batch_size=batch_size)

def semantic_search(query: str, texts: List[str], embedder, embeddings: np.ndarray,
                    top_k: int = 5) -> List[Tuple[int, float]]:
    """Return list of (index, cosine_sim) sorted desc."""
    q_emb = embedder.encode([query])
    sims = (embeddings @ q_emb.T).flatten()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [(int(idx), float(sims[idx])) for idx in top_idx]
