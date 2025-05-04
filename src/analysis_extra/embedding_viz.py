
from __future__ import annotations
from typing import Sequence
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def plot_tsne(embeddings: np.ndarray, labels: Sequence[str|int], *, perplexity: int = 30, early_exaggeration: float = 12.0):
    """t‑SNE projection of embedding space."""
    tsne = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, init="random", learning_rate="auto")
    z = tsne.fit_transform(embeddings)
    plt.figure()
    scatter = plt.scatter(z[:,0], z[:,1], c=labels, cmap="tab20", s=10, alpha=0.8)
    plt.title("t‑SNE of document embeddings")
    plt.tight_layout()
    return scatter
