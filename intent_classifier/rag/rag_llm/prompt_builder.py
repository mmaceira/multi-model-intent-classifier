"""Prompt building utilities for RAG LLM classifier."""

import logging

from .config import PROMPT_SYSTEM

logger = logging.getLogger(__name__)


def build_classification_prompt(
    docs: list[str],
    contexts: list[str],
    labels: list[str],
) -> list[dict]:
    """Build a prompt for classifying multiple documents.

    Args:
        docs: List of documents to classify
        contexts: List of context strings (retrieved examples) for each document
        labels: List of valid classification labels

    Returns:
        List of message dictionaries ready for LLM API call
    """
    # Build a single prompt with multiple items
    lines = []
    for i, (doc, ctx) in enumerate(zip(docs, contexts, strict=False), start=1):
        # Ensure doc is a string and not empty
        doc_text = str(doc).strip() if doc else " "
        ctx_text = str(ctx).strip() if ctx else " "
        lines.append(f"{i}. Utterance: {doc_text}\nContext:\n{ctx_text}")

    # Create the examples part of the JSON structure using actual labels from the list
    # Use diverse labels in examples, not just the first one
    # Always show valid JSON - never include "..." inside JSON
    num_examples = min(len(docs), len(labels), 3)
    example_labels = [labels[i % len(labels)] for i in range(num_examples)]
    examples_json = ", ".join(f'"{label}"' for label in example_labels)

    # Show ALL labels - the LLM needs to see all available options
    # For large label sets, we'll show them all but format them compactly
    full_labels_list = "\n".join(f"  - {label}" for label in labels)
    # Note: We show all labels even if there are many, because the LLM
    # MUST see all options to make correct predictions

    user_content = (
        f"Classify these {len(docs)} user utterances into ONE of these EXACT "
        f"intent categories (ALL {len(labels)} labels are shown below - "
        f"you MUST use one of these EXACT labels):\n\n{full_labels_list}\n\n"
        + "CRITICAL RULES - READ CAREFULLY:\n"
        + f"1. You MUST use one of the EXACT {len(labels)} labels from the list above - "
        "copy them EXACTLY (case-sensitive, including underscores/hyphens)\n"
        + "2. ALL available labels are shown above - do NOT create new labels, "
        "do NOT use synonyms, do NOT modify labels, do NOT paraphrase\n"
        + "3. If an utterance seems similar to a label but doesn't match exactly, "
        "choose the CLOSEST matching label from the list above\n"
        + "4. Look at the context examples - they show similar utterances and "
        "their correct EXACT labels\n"
        + "5. Your response MUST be ONLY a valid JSON object with a 'labels' array - "
        "NO other text before or after\n"
        + f"6. The 'labels' array MUST contain EXACTLY {len(docs)} labels, "
        "one for each utterance in order\n"
        + "7. Each label MUST be copied EXACTLY from the list above - "
        "no modifications, no variations\n\n"
        + "\n\n".join(lines)
        + "\n\nRESPOND WITH ONLY THIS JSON FORMAT "
        "(no explanations, no markdown, no code blocks):\n"
        + f'{{"labels": [{examples_json}]}}\n'
        + f"\nNote: The 'labels' array must contain EXACTLY {len(docs)} labels "
        f"(one for each utterance above). "
        + f"Each label must EXACTLY match one of the {len(labels)} labels from the list above."
    )

    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    # Debug logging: log the prompt being sent (first time only, to avoid spam)
    if logger.isEnabledFor(logging.DEBUG) or len(docs) <= 3:
        first_chars = min(200, len(user_content))
        logger.debug(
            f"Prompt being sent to LLM (first {first_chars} chars):\n" f"{user_content[:200]}..."
        )
        logger.debug(f"Number of labels in prompt: {len(labels)}")
        logger.debug(f"Labels: {labels}")

    return messages
