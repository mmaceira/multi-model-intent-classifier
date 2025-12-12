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
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from ...utils.method_logger import get_logger as get_method_logger, log_method
from ..classifier_base import RagClassifierBase

# Load environment variables from .env file
load_dotenv()

# LiteLLM can be quite chatty; keep logs focused on errors unless overridden.
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("LITELLM_SUPPRESS_LOGGING", "true")

# Suppress Pydantic serialization warnings BEFORE importing litellm
# These warnings come from litellm/openai when deserializing API responses
# They're not serious - the code works correctly, just verbose warnings

# Store original warning handler
_original_showwarning = warnings.showwarning


def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
    """Custom warning handler that filters out Pydantic serialization warnings."""
    # Check if this is a Pydantic warning we want to suppress
    if issubclass(category, UserWarning):
        msg_str = str(message)
        filename_str = str(filename) if filename else ""

        # Suppress if it's from pydantic or contains our target messages
        if (
            "pydantic" in filename_str.lower()
            or "PydanticSerializationUnexpectedValue" in msg_str
            or "Expected `Usage`" in msg_str
            or "serialized value may not be as expected" in msg_str
        ):
            return  # Suppress this warning

    # For all other warnings, use the original handler
    _original_showwarning(message, category, filename, lineno, file, line)


# Install our custom warning handler
warnings.showwarning = _filtered_showwarning

# Also set up filterwarnings as backup
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal")
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings("ignore", message=".*Expected `Usage`.*")
warnings.filterwarnings("ignore", message=".*serialized value may not be as expected.*")

# Now import litellm after warnings are configured
from litellm import completion  # noqa: E402

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
        self.examples: list[Example] = list(examples)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform([ex.text for ex in self.examples])

    def select_topk_with_min_labels(
        self, query: str, k: int, m: int
    ) -> list[tuple[Example, float]]:
        """Return top-k most similar examples but keep scanning until >= m labels."""
        if k <= 0:
            raise ValueError("k must be positive.")
        if m <= 0:
            raise ValueError("m must be positive.")

        q_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ q_vec.T).toarray().ravel()
        order = np.argsort(-scores)

        selected: list[tuple[Example, float]] = []
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


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON object, falling back to first {...} fragment."""
    stripped = _strip_code_fences(text)
    try:
        return json.loads(stripped)
    except Exception as err:
        match = _JSON_OBJ_RE.search(stripped)
        if not match:
            raise ValueError("No JSON object found in response.") from err
        return json.loads(match.group(0))


def _normalize_endpoint(endpoint: str) -> str:
    """Normalize endpoint URL by adding http:// if protocol is missing.

    Args:
        endpoint: Endpoint URL (with or without protocol)

    Returns:
        Normalized endpoint URL with protocol
    """
    endpoint = endpoint.strip()
    if not endpoint.startswith(("http://", "https://")):
        # Default to http:// if no protocol specified
        endpoint = f"http://{endpoint}"
    return endpoint


def _call_llm(messages: list[dict[str, str]], model: str) -> str:
    """Send chat completion request (temperature fixed at 0)."""
    # Only use response_format for models that support it (OpenAI, Anthropic, etc.)
    # Ollama and some other providers don't support this parameter
    supports_json_schema = not model.startswith("ollama/")

    # For Ollama models, check config for endpoint and set OLLAMA_API_BASE if needed
    if model.startswith("ollama/") and "OLLAMA_API_BASE" not in os.environ:
        try:
            from intent_classifier.utils.config_loader import discover_config_file, load_config

            config = load_config(discover_config_file())
            ollama_endpoint = config.get("model", {}).get("ollama_endpoint")
            if ollama_endpoint:
                normalized_endpoint = _normalize_endpoint(ollama_endpoint)
                os.environ["OLLAMA_API_BASE"] = normalized_endpoint
                logger.debug(f"Set OLLAMA_API_BASE from config: {normalized_endpoint}")
        except Exception:
            # Silently fall back if config loading fails
            pass

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }

    if supports_json_schema:
        kwargs["response_format"] = {"type": "json_object"}

    # Log model call (only at debug level to reduce verbosity)
    # The progress bar will show overall progress
    resp = completion(**kwargs)
    return resp["choices"][0]["message"]["content"]


def _normalize_label(label: str) -> str:
    """Normalize a label for comparison (case-insensitive, normalize spacing around commas).

    Args:
        label: Label string to normalize

    Returns:
        Normalized label string
    """
    # Normalize: lowercase, remove spaces around commas, remove extra spaces
    normalized = label.strip().lower()
    # Remove spaces around commas: "Alta, assegurança" -> "alta,asseguranca"
    normalized = ",".join(part.strip() for part in normalized.split(","))
    return normalized


def _is_multilabel_mode(allowed_labels: list[str]) -> bool:
    """Detect if we're in multilabel mode by checking if labels contain commas.

    Args:
        allowed_labels: List of allowed label strings

    Returns:
        True if multilabel mode detected (labels contain commas), False otherwise
    """
    return any("," in label for label in allowed_labels)


def _normalize_label_set(labels: list[str]) -> dict[str, str]:
    """Create a mapping from normalized labels to original labels.

    Args:
        labels: List of label strings

    Returns:
        Dictionary mapping normalized label -> original label
    """
    return {_normalize_label(label): label for label in labels}


def _load_prompt_template(prompt_style: str, prompt_type: str, is_multilabel: bool) -> str:
    """Load a prompt template from file.

    Args:
        prompt_style: Prompt style ("default" or "short")
        prompt_type: Type of prompt ("system" or "user")
        is_multilabel: Whether in multilabel mode

    Returns:
        Prompt template string

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    from pathlib import Path

    # Determine label mode suffix
    label_mode = "multilabel" if is_multilabel else "singlelabel"

    # Build path to prompt file
    prompts_dir = Path(__file__).parent / "prompts" / prompt_style
    prompt_file = prompts_dir / f"{prompt_type}_{label_mode}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {prompt_file}\nExpected prompt files in: {prompts_dir}"
        )

    return prompt_file.read_text(encoding="utf-8").strip()


def _build_prompts(
    allowed_labels: list[str],
    defs_block: str,
    few_shots: str,
    query: str,
    is_multilabel: bool,
    prompt_style: str = "default",
) -> tuple[str, str]:
    """Build system and user prompts based on style and multilabel mode.

    Loads prompts from template files and formats them with the provided values.

    Args:
        allowed_labels: List of allowed label strings
        defs_block: Formatted label definitions block
        few_shots: Formatted few-shot examples
        query: Query text to classify
        is_multilabel: Whether in multilabel mode
        prompt_style: Prompt style ("default" or "short")

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Load prompt templates
    system_template = _load_prompt_template(prompt_style, "system", is_multilabel)
    user_template = _load_prompt_template(prompt_style, "user", is_multilabel)

    # Format user prompt with provided values
    user_prompt = user_template.format(
        allowed_labels=allowed_labels,
        defs_block=defs_block,
        few_shots=few_shots,
        query=query,
    )

    return system_template, user_prompt


def classify_single(
    *,
    model: str,
    query: str,
    retriever: Retriever,
    label_defs: dict[str, str] | None = None,
    k: int = 10,
    m: int = 4,
    prompt_style: str = "default",
) -> dict[str, Any]:
    """Classify a single query via retrieval + constrained LLM JSON response.

    Supports both single-label and multi-label classification. Multi-label mode
    is automatically detected when labels contain commas (e.g., "Alta,asseguranca").

    Args:
        model: LLM model identifier
        query: Text to classify
        retriever: Retriever instance
        label_defs: Optional label definitions dictionary
        k: Number of neighbors to retrieve
        m: Minimum distinct labels required
        prompt_style: Prompt style ("default" or "short")
    """
    label_defs = label_defs or {}
    retrieved = retriever.select_topk_with_min_labels(query, k=k, m=m)
    if not retrieved:
        raise ValueError("Retriever returned no examples; cannot classify.")

    allowed_labels = sorted({ex.label for ex, _ in retrieved})
    if not allowed_labels:
        raise ValueError("No labels available from retrieval results.")

    # Detect multilabel mode and create normalization mapping
    is_multilabel = _is_multilabel_mode(allowed_labels)
    normalized_to_original = _normalize_label_set(allowed_labels)

    defs_block = "\n".join(
        f"- {label}: {label_defs.get(label, '').strip()}" for label in allowed_labels
    ).strip()
    if not defs_block:
        defs_block = "- (no definitions available)"

    few_shots = "\n".join(f"- text: {ex.text}\n  label: {ex.label}" for ex, _ in retrieved)

    # Build prompts using the prompt builder
    system_prompt, user_prompt = _build_prompts(
        allowed_labels=allowed_labels,
        defs_block=defs_block,
        few_shots=few_shots,
        query=query,
        is_multilabel=is_multilabel,
        prompt_style=prompt_style,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = _call_llm(messages, model=model)

    try:
        parsed = _extract_json_object(content)
    except Exception:
        if is_multilabel:
            repair_prompt = (
                "You previously returned an invalid label.\n"
                "Return ONLY a corrected JSON object with this schema:\n"
                '{ "label": "<comma-separated labels>", "confidence": <number 0..1> }\n'
                f"CRITICAL: The label MUST be EXACTLY one of these allowed labels: "
                f"{allowed_labels}\n"
                "Copy the label EXACTLY as it appears in the list above. "
                "Do not create new combinations.\n"
                "Use EXACT format: comma-separated, NO spaces (e.g., 'Alta,asseguranca').\n"
                f"Bad output:\n{content}\n"
            )
        else:
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
    if label is None:
        # Fallback to top retrieved label if parsing failed
        top_label = retrieved[0][0].label
        logger.warning("Failed to extract label from LLM response, using fallback")
        return {
            "label": top_label,
            "confidence": 0.5,
            "allowed_labels": allowed_labels,
            "shots_used": len(retrieved),
        }

    # Normalize the returned label for comparison
    normalized_label = _normalize_label(str(label))

    # Check if normalized label matches any allowed label
    if normalized_label not in normalized_to_original:
        # Try to repair: ask LLM to fix the label
        if is_multilabel:
            repair_label_prompt = (
                f"You returned label '{label}' which is not in the allowed set.\n"
                f"CRITICAL: The label MUST be EXACTLY one of these: {allowed_labels}\n"
                "Return ONLY a corrected JSON object with this schema:\n"
                '{ "label": "<exact label from allowed list>", "confidence": <number 0..1> }\n'
                "Copy the label EXACTLY as it appears in the allowed list. "
                "Do not create new combinations.\n"
            )
        else:
            repair_label_prompt = (
                f"You returned label '{label}' which is not in the allowed set.\n"
                f"CRITICAL: The label MUST be EXACTLY one of these: {allowed_labels}\n"
                "Return ONLY a corrected JSON object with this schema:\n"
                '{ "label": "<exact label from allowed list>", "confidence": <number 0..1> }\n'
            )
        repair_label_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": repair_label_prompt},
        ]
        try:
            repaired_content = _call_llm(repair_label_messages, model=model)
            repaired_parsed = _extract_json_object(repaired_content)
            repaired_label = repaired_parsed.get("label")
            if repaired_label is None:
                raise ValueError("Repaired response missing label")
            repaired_normalized = _normalize_label(str(repaired_label))
            if repaired_normalized in normalized_to_original:
                # Repair succeeded
                original_label = normalized_to_original[repaired_normalized]
                parsed["label"] = original_label
                logger.info(f"Successfully repaired invalid label '{label}' to '{original_label}'")
            else:
                # Repair failed, use fallback
                top_label = retrieved[0][0].label
                logger.warning(
                    f"Invalid label '{label}' not in allowed set {allowed_labels}. "
                    f"Repair attempt also failed. Using fallback label '{top_label}' "
                    f"from most similar retrieved example."
                )
                original_label = top_label
        except Exception as e:
            # Repair attempt failed, use fallback
            top_label = retrieved[0][0].label
            logger.warning(
                f"Invalid label '{label}' not in allowed set {allowed_labels}. "
                f"Repair attempt failed: {e}. Using fallback label '{top_label}' "
                f"from most similar retrieved example."
            )
            original_label = top_label
    else:
        # Use the original (correctly formatted) label from allowed_labels
        original_label = normalized_to_original[normalized_label]

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    parsed["label"] = original_label
    parsed["confidence"] = max(0.0, min(1.0, confidence))
    parsed["allowed_labels"] = allowed_labels
    parsed["shots_used"] = len(retrieved)
    return parsed


def _load_examples(
    *,
    max_train_samples: int | None = None,
    max_classes: int | None = None,
    dataset_name: str | None = None,
    use_oos: bool = False,
) -> tuple[list[Example], dict[str, str]]:
    """Load training data via existing dataset loader."""
    from intent_classifier.datasets.dataset import get_dataset

    # Discover dataset name and multilabel setting if not provided
    multilabel = False
    if dataset_name is None:
        import os

        # Try to get from config file
        config_file = os.environ.get("CONFIG_FILE")
        if config_file:
            try:
                from intent_classifier.utils.config_loader import load_config

                config = load_config(config_file, apply_variable_substitution=False)
                dataset_name = config.get("dataset", {}).get("name")
                multilabel = config.get("dataset", {}).get("multilabel", False)
            except Exception:
                pass

        # Fallback: discover first available dataset
        if dataset_name is None:
            from intent_classifier.utils.config_loader import get_first_available_dataset

            dataset_name = get_first_available_dataset()

        # Final fallback
        if dataset_name is None:
            dataset_name = "clinc150"  # For backward compatibility
    else:
        # If dataset_name is provided, try to get multilabel from config
        import os

        config_file = os.environ.get("CONFIG_FILE")
        if config_file:
            try:
                from intent_classifier.utils.config_loader import load_config

                config = load_config(config_file, apply_variable_substitution=False)
                multilabel = config.get("dataset", {}).get("multilabel", False)
            except Exception:
                pass

    X_train, y_train, *_rest = get_dataset(
        dataset_name=dataset_name or "clinc150",
        use_oos=use_oos,
        max_train_samples=max_train_samples,
        max_classes=max_classes,
        multilabel=multilabel,
    )

    # Convert labels to strings (handle both single-label strings and multi-label lists)
    def label_to_string(label: Any) -> str:
        """Convert a label to a string representation.

        Args:
            label: Can be a string, list, tuple, or other type

        Returns:
            String representation (comma-separated for lists)
        """
        if isinstance(label, (list, tuple)):
            # Multi-label: convert to comma-separated string
            return ",".join(
                str(label_item)
                for label_item in label
                if label_item is not None and str(label_item).strip()
            )
        else:
            # Single-label: convert to string
            return str(label) if label is not None else ""

    examples = [
        Example(text=text, label=label_to_string(label))
        for text, label in zip(X_train, y_train, strict=False)
    ]

    label_defs: dict[str, str] = {}
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
        label_defs: dict[str, str] | None = None,
        prompt_style: str = "default",
    ):
        super().__init__(labels)
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.min_labels = min_labels
        self.examples = retriever.examples
        self.label_defs = label_defs or {}
        self.prompt_style = prompt_style

    @classmethod
    def load_default(
        cls,
        *,
        use_openai: bool = False,  # kept for compatibility
        model: str = "ollama/llama3.1:8b",
        top_k: int = 10,
        min_labels: int = 4,
        prompt_style: str = "default",
        **kwargs: Any,
    ) -> RagLLM:
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
            prompt_style=prompt_style,
        )

    @log_method()
    def predict(self, docs: Sequence[str], **kwargs: Any) -> list[str]:
        """Classify multiple documents."""
        method_logger = get_method_logger()
        results: list[str] = []

        # Log model call start (only for batches)
        if len(docs) > 1:
            logger.info(f"Starting classification with {self.model} for {len(docs)} documents")

        # Use tqdm for progress bar if we have multiple documents
        use_progress = len(docs) > 1
        iterator = (
            tqdm(
                docs,
                desc=f"Classifying ({self.model})",
                unit="doc",
                disable=not use_progress,
                ncols=100,  # Limit width to avoid clutter
            )
            if use_progress
            else docs
        )

        for doc in iterator:
            try:
                result = classify_single(
                    model=self.model,
                    query=doc,
                    retriever=self.retriever,
                    label_defs=self.label_defs,
                    k=self.top_k,
                    m=self.min_labels,
                    prompt_style=self.prompt_style,
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
                # Try to get a fallback from retriever if possible
                try:
                    retrieved = self.retriever.select_topk_with_min_labels(doc, k=1, m=1)
                    if retrieved:
                        fallback_label = retrieved[0][0].label
                        logger.warning(f"Using fallback label '{fallback_label}' from retriever")
                        results.append(fallback_label)
                    else:
                        results.append(self.labels[0] if self.labels else "unknown")
                except Exception:
                    # Last resort fallback
                    results.append(self.labels[0] if self.labels else "unknown")

        if len(docs) > 1:
            logger.info(f"Completed classification: {len(results)} predictions")

        return results

    @log_method()
    def predict_proba(self, docs: Sequence[str], **kwargs: Any) -> np.ndarray:
        """Generate probability estimates for each class using retrieved label counts."""
        sorted_labels = sorted(self.labels)
        label_to_idx = {label: i for i, label in enumerate(sorted_labels)}
        all_probas = np.zeros((len(docs), len(sorted_labels)))

        # Use tqdm for progress bar if we have multiple documents
        use_progress = len(docs) > 1
        iterator = (
            tqdm(
                enumerate(docs),
                total=len(docs),
                desc="Computing probabilities",
                unit="doc",
                disable=not use_progress,
            )
            if use_progress
            else enumerate(docs)
        )

        for i, doc in iterator:
            try:
                retrieved = self.retriever.select_topk_with_min_labels(
                    doc, k=self.top_k, m=self.min_labels
                )

                label_counts: dict[str, int] = {}
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
