"""
OpenAI‑Embedding Logistic Regression module for text classification (scaled + C‑tuned).

⚠️  DEPRECATED: This module is kept for backward compatibility only.
Please use `EmbeddingLogReg` from `intent_classifier.algorithms.embedding_logreg` instead.

The new `EmbeddingLogReg` class supports both OpenAI and SBERT embeddings via the
`use_openai` parameter, making it more flexible and not requiring an API key for
local SBERT embeddings.

This file now just imports and re-exports `EmbeddingLogReg` as `OpenAIEmbedLogReg`
for backward compatibility.
"""

from intent_classifier.algorithms.embedding_logreg import EmbeddingLogReg

# Backward compatibility alias
OpenAIEmbedLogReg = EmbeddingLogReg

__all__ = ["OpenAIEmbedLogReg"]
