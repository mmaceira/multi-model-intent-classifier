"""Shared helpers for generating stable model identifiers.

The goal is to have **one** canonical implementation that is used everywhere
model folder names or ``model_id`` values are derived from human‑readable
display names.
"""

from __future__ import annotations

import re
import unicodedata


def _normalise_ascii(text: str) -> str:
    """Best‑effort ASCII normalisation (strip accents and other marks)."""
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def slugify_model_id(display_name: str) -> str:
    """Create a stable, filesystem‑friendly model identifier.

    Rules
    -----
    - Lowercase
    - ASCII only (accents stripped)
    - Replace ``+`` with a word separator (no explicit ``plus`` token)
    - Collapse whitespace / punctuation into single underscores
    - Strip leading / trailing underscores
    - Ensure \"TF-IDF\" style tokens become ``tfidf`` (not ``tf_idf``)

    Examples
    --------
    >>> slugify_model_id("TF-IDF bigrams + SVM")
    'tfidf_bigrams_svm'
    >>> slugify_model_id("Embedding + LogReg (Qwen/Ollama)")
    'embedding_logreg_qwen_ollama'
    """
    if not display_name:
        return "model"

    text = _normalise_ascii(display_name)

    # Normalise common TF‑IDF spellings first so dashes don't create extra
    # underscores and the token is treated as a single word.
    text = text.replace("TF-IDF", "TFIDF").replace("tf-idf", "tfidf").replace("Tf-Idf", "TFIDF")

    # Treat "+" as a separator between tokens rather than a literal "plus".
    text = text.replace("+", " ")

    # Lowercase and collapse any remaining non‑alphanumeric runs into "_".
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text or "model"


__all__ = ["slugify_model_id"]
