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
    # Show ALL labels - the LLM needs to see all available options
    full_labels_list = "\n".join(f"  - {label}" for label in labels)

    # Build a numbered list of utterances with limited context (top 2 examples only)
    utterances_section = []
    for i, (doc, ctx) in enumerate(zip(docs, contexts, strict=False), start=1):
        doc_text = str(doc).strip() if doc else " "
        ctx_text = str(ctx).strip() if ctx else " "
        # Limit context to top 2 examples to reduce prompt length
        ctx_lines = ctx_text.split("\n")[:2] if ctx_text else []
        ctx_short = "\n".join(ctx_lines) if ctx_lines else "No examples"
        utterances_section.append(f"{i}. {doc_text}\n   Similar: {ctx_short}")

    # Build a concrete example using the first utterance's context to show the pattern
    first_utterance_example = ""
    if utterances_section:
        first_ctx = contexts[0] if contexts else ""
        first_ctx_lines = first_ctx.split("\n")[:1] if first_ctx else []
        if first_ctx_lines:
            # Extract label from first context example
            first_example_line = first_ctx_lines[0]
            if "→ Label:" in first_example_line:
                example_label = first_example_line.split("→ Label:")[-1].strip().strip('"')
                first_utterance_example = (
                    f"\nEXAMPLE MAPPING:\n"
                    f'  Utterance 1: "{docs[0][:50]}..."\n'
                    f"  Similar examples show: {example_label}\n"
                    f'  → So label[0] should be: "{example_label}"\n'
                )

    user_content = (
        f"TASK: Classify {len(docs)} utterances. Return EXACTLY {len(docs)} labels.\n\n"
        + f"VALID LABELS (use ONLY these - copy exactly):\n{full_labels_list}\n\n"
        + "HOW TO CLASSIFY:\n"
        + "1. Read utterance 1, look at its 'Similar' examples to see which label to use\n"
        + "2. Put that label in position 1 of your array\n"
        + "3. Repeat for utterance 2 (label in position 2), utterance 3 (position 3), etc.\n"
        + f"4. You must return {len(docs)} labels total, one for each utterance\n"
        + "5. The order matters: label[0] for utterance 1, label[1] for utterance 2, etc.\n\n"
        + first_utterance_example
        + "\nUTTERANCES TO CLASSIFY:\n"
        + "\n\n".join(utterances_section)
        + "\n\n"
        + f"YOUR RESPONSE (JSON format with {len(docs)} labels in order):\n"
        + '{"labels": ["label_for_utterance_1", "label_for_utterance_2", ...]}\n\n'
        + f"⚠️ CRITICAL: Return {len(docs)} labels, one for each utterance in order.\n"
        + f"⚠️ Use labels from: {', '.join(labels)}\n"
        + "⚠️ Return ONLY the JSON, nothing else."
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
