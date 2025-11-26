"""RAG-LLM implementation and classifier.

This module contains the core RAG-LLM building blocks (retriever, LLM
caller, JSON parsing utilities) and the `RagLLM` classifier, so LLM-based
RAG behaves like any other algorithm under `intent_classifier.rag`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv
from litellm import completion
from sklearn.feature_extraction.text import TfidfVectorizer

from ...utils.method_logger import get_logger as get_method_logger, log_method
from ..classifier_base import RagClassifierBase

# Load environment variables from .env file
load_dotenv()

# LiteLLM can be quite chatty; keep logs focused on errors unless overridden.
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LITELLM_SUPPRESS_LOGGING", "true")

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class Example:
    """Simple text+label record for retrieval shots."""

    text: str
    label: str


class Retriever:
    """TF-IDF bi-gram retriever with min-label constraint."""

    def __init__(self, examples: Sequence[Example]):
        if not examples:
            raise ValueError("Retriever needs at least one training example.")
        self.examples: List[Example] = list(examples)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform([ex.text for ex in self.examples])

    def select_topk_with_min_labels(
        self, query: str, k: int, m: int
    ) -> List[Tuple[Example, float]]:
        """Return top-k most similar examples but keep scanning until >= m labels."""
        if k <= 0:
            raise ValueError("k must be positive.")
        if m <= 0:
            raise ValueError("m must be positive.")

        q_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ q_vec.T).toarray().ravel()
        order = np.argsort(-scores)

        selected: List[Tuple[Example, float]] = []
        seen_labels: set[str] = set()

        for idx in order:
            example = self.examples[int(idx)]
            score = float(scores[int(idx)])
            selected.append((example, score))
            seen_labels.add(example.label)

            if len(selected) >= k and len(seen_labels) >= m:
                break

        if len(selected) > k and len(seen_labels) >= m:
            selected = selected[:k]

        return selected


_JSON_OBJ_RE = re.compile(r"\{.*?\}", flags=re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Remove ```json fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, count=1).strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Parse JSON object, falling back to first {...} fragment."""
    stripped = _strip_code_fences(text)
    try:
        return json.loads(stripped)
    except Exception as err:
        match = _JSON_OBJ_RE.search(stripped)
        if not match:
            raise ValueError("No JSON object found in response.") from err
        return json.loads(match.group(0))


def _call_llm(messages: List[Dict[str, str]], model: str) -> str:
    """Send chat completion request (temperature fixed at 0)."""
    # Only use response_format for models that support it (OpenAI, Anthropic, etc.)
    # Ollama and some other providers don't support this parameter
    supports_json_schema = not model.startswith("ollama/")

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }

    if supports_json_schema:
        kwargs["response_format"] = {"type": "json_object"}

    resp = completion(**kwargs)
    return resp["choices"][0]["message"]["content"]


def classify_single(
    *,
    model: str,
    query: str,
    retriever: Retriever,
    label_defs: Dict[str, str] | None = None,
    k: int = 10,
    m: int = 4,
) -> Dict[str, Any]:
    """Classify a single query via retrieval + constrained LLM JSON response."""
    label_defs = label_defs or {}
    retrieved = retriever.select_topk_with_min_labels(query, k=k, m=m)
    if not retrieved:
        raise ValueError("Retriever returned no examples; cannot classify.")

    allowed_labels = sorted({ex.label for ex, _ in retrieved})
    if not allowed_labels:
        raise ValueError("No labels available from retrieval results.")

    defs_block = "\n".join(
        f"- {label}: {label_defs.get(label, '').strip()}" for label in allowed_labels
    ).strip()
    if not defs_block:
        defs_block = "- (no definitions available)"

    few_shots = "\n".join(f"- text: {ex.text}\n  label: {ex.label}" for ex, _ in retrieved)

    system_prompt = (
        "You are an intent classifier.\n"
        "Return ONLY a valid JSON object.\n"
        "Choose exactly ONE label from Allowed labels.\n"
        "No markdown. No prose. No code fences.\n"
    )

    user_prompt = (
        f"Allowed labels: {allowed_labels}\n"
        f"Label definitions:\n{defs_block}\n\n"
        f"Retrieved labeled examples:\n{few_shots}\n\n"
        f"CLASSIFY THIS QUERY:\n{query}\n\n"
        "Return exactly this JSON schema:\n"
        '{ "label": "<allowed label>", "confidence": <number 0..1> }\n'
        "Rules:\n"
        "- Choose exactly ONE label from Allowed labels.\n"
        "- If unsure, pick the best label and set a low confidence.\n"
        "- confidence must be between 0 and 1.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = _call_llm(messages, model=model)

    try:
        parsed = _extract_json_object(content)
    except Exception:
        repair_prompt = (
            "You previously returned an invalid JSON object.\n"
            "Return ONLY a corrected JSON object with this schema:\n"
            '{ "label": "<allowed label>", "confidence": <number 0..1> }\n'
            f"Allowed labels: {allowed_labels}\n"
            f"Bad output:\n{content}\n"
        )
        repair_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": repair_prompt},
        ]
        repaired = _call_llm(repair_messages, model=model)
        parsed = _extract_json_object(repaired)

    label = parsed.get("label")
    if label not in allowed_labels:
        # Optional fallback: use most similar retrieved example's label.
        top_label = retrieved[0][0].label
        raise ValueError(
            f"Invalid label '{label}' not in allowed set {allowed_labels}. "
            f"Suggested fallback: '{top_label}'."
        )

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    parsed["label"] = label
    parsed["confidence"] = max(0.0, min(1.0, confidence))
    parsed["allowed_labels"] = allowed_labels
    parsed["shots_used"] = len(retrieved)
    return parsed


def _load_examples(
    *,
    max_train_samples: int | None = None,
    max_classes: int | None = None,
    dataset_name: str = "clinc150",
    use_oos: bool = False,
) -> Tuple[List[Example], Dict[str, str]]:
    """Load training data via existing dataset loader."""
    from intent_classifier.datasets.dataset import get_dataset

    X_train, y_train, *_rest = get_dataset(
        dataset_name=dataset_name,
        use_oos=use_oos,
        max_train_samples=max_train_samples,
        max_classes=max_classes,
    )

    examples = [
        Example(text=text, label=label) for text, label in zip(X_train, y_train, strict=False)
    ]

    label_defs: Dict[str, str] = {}
    for ex in examples:
        label_defs.setdefault(ex.label, ex.text[:160])

    return examples, label_defs


class RagLLM(RagClassifierBase):
    """RAG-style intent classifier built on the local `Example`/`Retriever`/LLM pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        labels: Sequence[str],
        *,
        model: str = "ollama/llama3.1:8b",
        top_k: int = 10,
        min_labels: int = 4,
        label_defs: Dict[str, str] | None = None,
    ):
        super().__init__(labels)
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.min_labels = min_labels
        self.examples = retriever.examples
        self.label_defs = label_defs or {}

    @classmethod
    def load_default(
        cls,
        *,
        use_openai: bool = False,  # kept for compatibility
        model: str = "ollama/llama3.1:8b",
        top_k: int = 10,
        min_labels: int = 4,
        **kwargs: Any,
    ) -> "RagLLM":
        """Create a classifier with default configuration."""
        examples, label_defs = _load_examples()
        retriever = Retriever(examples)
        labels = sorted({ex.label for ex in examples})

        return cls(
            retriever=retriever,
            labels=labels,
            model=model,
            top_k=top_k,
            min_labels=min_labels,
            label_defs=label_defs,
        )

    @log_method
    def predict(self, docs: Sequence[str], **kwargs: Any) -> List[str]:
        """Classify multiple documents."""
        method_logger = get_method_logger()
        results: List[str] = []

        for doc in docs:
            try:
                result = classify_single(
                    model=self.model,
                    query=doc,
                    retriever=self.retriever,
                    label_defs=self.label_defs,
                    k=self.top_k,
                    m=self.min_labels,
                )
                prediction = result["label"]
                method_logger.log_prediction(
                    input=doc,
                    output=prediction,
                    metadata={"confidence": result.get("confidence"), "model": self.model},
                )
                results.append(prediction)
            except Exception as e:  # pragma: no cover - defensive logging
                logger.error(f"Error classifying document '{doc[:50]}...': {e}")
                results.append(self.labels[0] if self.labels else "unknown")

        return results

    @log_method
    def predict_proba(self, docs: Sequence[str], **kwargs: Any) -> np.ndarray:
        """Generate probability estimates for each class using retrieved label counts."""
        sorted_labels = sorted(self.labels)
        label_to_idx = {label: i for i, label in enumerate(sorted_labels)}
        all_probas = np.zeros((len(docs), len(sorted_labels)))

        for i, doc in enumerate(docs):
            try:
                retrieved = self.retriever.select_topk_with_min_labels(
                    doc, k=self.top_k, m=self.min_labels
                )

                label_counts: Dict[str, int] = {}
                for ex, _ in retrieved:
                    label = ex.label
                    label_counts[label] = label_counts.get(label, 0) + 1

                total = len(retrieved) if retrieved else 1
                for label, count in label_counts.items():
                    if label in label_to_idx:
                        all_probas[i, label_to_idx[label]] = count / total

                row_sum = all_probas[i, :].sum()
                if row_sum > 0:
                    all_probas[i, :] /= row_sum
                else:
                    all_probas[i, :] = 1.0 / len(sorted_labels)

            except Exception as e:  # pragma: no cover - defensive logging
                logger.error(f"Error computing probabilities for document '{doc[:50]}...': {e}")
                all_probas[i, :] = 1.0 / len(sorted_labels)

        return all_probas
