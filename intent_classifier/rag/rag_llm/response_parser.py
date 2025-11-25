"""Response parsing and validation for LLM classification results."""

import json
import logging
import re
from collections import Counter
from typing import List

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> str:
    """Extract JSON object from text, handling markdown code blocks and extra text.

    Args:
        text: Raw text response from LLM

    Returns:
        Extracted JSON string
    """
    # Strip markdown code blocks if present
    # (LLM sometimes wraps JSON in ```json ... ```)
    # Handle cases like: ```json\n{...}\n``` or ```\n{...}\n```
    if text.startswith("```"):
        # Find the first newline after the opening backticks
        first_newline = text.find("\n")
        if first_newline != -1:
            # Extract everything after the first newline
            text = text[first_newline + 1 :]
        else:
            # No newline, just strip backticks
            text = text.lstrip("`")
        # Remove closing backticks (handle multiple backticks)
        text = text.rstrip("`").strip()

    # Extract JSON from text - handle cases where LLM adds explanations after JSON
    # Find the first { and last } to extract just the JSON object
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        # Extract just the JSON portion
        json_text = text[first_brace : last_brace + 1]
    else:
        # Fallback: try to parse the whole text
        json_text = text

    return json_text


def parse_json_response(text: str) -> dict | list:
    """Parse JSON response from LLM with robust error handling.

    Args:
        text: Raw text response from LLM

    Returns:
        Parsed JSON data (dict or list)

    Raises:
        json.JSONDecodeError: If JSON cannot be parsed
    """
    json_text = extract_json_from_text(text)

    # Improved JSON parsing with better error handling
    try:
        response_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        # Try to find JSON object in the text more aggressively
        # Look for patterns like {"labels": [...]}
        json_match = re.search(r'\{[^{]*"labels"[^{]*\[[^\]]*\][^{]*\}', text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            try:
                response_data = json.loads(json_text)
            except json.JSONDecodeError:
                raise e from None
        else:
            raise e from None

    return response_data


def extract_labels_from_response(
    response_data: dict | list, expected_count: int
) -> List[str] | None:
    """Extract labels array from parsed response data.

    Args:
        response_data: Parsed JSON response (dict or list)
        expected_count: Expected number of labels

    Returns:
        List of labels or None if extraction fails
    """
    # Check if the response has a 'labels' property
    if isinstance(response_data, dict) and "labels" in response_data:
        labels_out = response_data["labels"]
        # Ensure labels_out is a list
        if not isinstance(labels_out, list):
            logger.error(
                f"❌ LLM Response Error: Expected 'labels' to be a list but got "
                f"{type(labels_out).__name__}: {labels_out}"
            )
            # Convert single string to list if that's what we got
            if isinstance(labels_out, str):
                logger.warning(
                    f"⚠️  LLM returned single string label instead of array. "
                    f"Converting to list: ['{labels_out}']"
                )
                labels_out = [labels_out]
            else:
                logger.error(
                    f"❌ Cannot convert {type(labels_out).__name__} to list. "
                    f"Response structure is invalid."
                )
                return None
        return labels_out

    # Fallback to assuming the entire object is the array
    elif isinstance(response_data, list):
        logger.info(
            "ℹ️  LLM returned a list directly instead of {'labels': [...]} format. "
            "Using list as-is."
        )
        return response_data

    # Handle case when response doesn't match expected format
    else:
        # Try to find any array in the response
        labels_out = None
        for key, value in response_data.items():
            if isinstance(value, list) and len(value) > 0:
                labels_out = value
                logger.warning(
                    f"⚠️  LLM Response Warning: Found array at key '{key}' instead of 'labels'. "
                    f"Using '{key}' as labels array."
                )
                break

        if labels_out is None:
            # Look for any string that might be a valid label
            for key, value in response_data.items():
                if isinstance(value, str):
                    logger.warning(
                        f"⚠️  LLM Response Warning: Found single string '{value}' at key '{key}'. "
                        f"This will be used for all documents."
                    )
                    # Return as single-item list - caller will handle expansion
                    return [value]

            response_keys = list(response_data.keys()) if isinstance(response_data, dict) else "N/A"
            logger.error(
                f"❌ LLM Response Error: No valid labels array found in response. "
                f"Response keys: {response_keys}"
            )

        return labels_out


def normalize_label_count(labels: List[str], expected_count: int) -> List[str]:
    """Normalize the number of labels to match expected count.

    Args:
        labels: List of labels
        expected_count: Expected number of labels

    Returns:
        Normalized list with correct count
    """
    if len(labels) == expected_count:
        return labels

    logger.warning(
        f"⚠️  Label Count Mismatch: Expected {expected_count} labels but got {len(labels)}. "
        f"Adjusting to match expected count..."
    )

    # Extend with last label if too short
    if len(labels) < expected_count:
        extension_label = labels[-1] if labels else None
        if extension_label:
            labels.extend([extension_label] * (expected_count - len(labels)))
        else:
            # Fallback: repeat first label if available
            if labels:
                labels.extend([labels[0]] * (expected_count - len(labels)))
    # Truncate if too long
    else:
        labels = labels[:expected_count]

    return labels


def validate_and_normalize_labels(
    labels: List[str],
    valid_labels: List[str],
    neighbors_list: List[List[dict]],
) -> tuple[List[str], int]:
    """Validate labels against valid set and normalize using fuzzy matching.

    Args:
        labels: List of labels to validate
        valid_labels: List of valid label strings
        neighbors_list: List of neighbor lists for fallback (majority vote)

    Returns:
        Tuple of (validated_labels, invalid_count)
    """
    validated_labels = []
    invalid_count = 0

    for _idx, lab in enumerate(labels):
        lab_str = str(lab).strip()

        # Exact match - if found, use it and skip fuzzy matching
        if lab_str in valid_labels:
            validated_labels.append(lab_str)
            continue  # Skip to next label

        # No exact match - try fuzzy matching
        invalid_count += 1
        lab_lower = lab_str.lower()
        matched = None

        # Try case-insensitive match
        for valid_label in valid_labels:
            if valid_label.lower() == lab_lower:
                matched = valid_label
                break

        # Try matching with underscores/spaces normalized
        if matched is None:
            lab_normalized = lab_lower.replace(" ", "_").replace("-", "_")
            for valid_label in valid_labels:
                valid_normalized = valid_label.lower().replace(" ", "_").replace("-", "_")
                if valid_normalized == lab_normalized:
                    matched = valid_label
                    break

        if matched:
            validated_labels.append(matched)
            invalid_count -= 1  # Adjust count since we found a match
            logger.debug(f"Normalized label '{lab_str}' to '{matched}'")
        else:
            # Fallback: use majority vote from neighbors for this document
            # This is better than always using valid_labels[0]
            doc_idx = len(validated_labels)
            if doc_idx < len(neighbors_list) and neighbors_list[doc_idx]:
                # Get majority label from neighbors
                neighbor_labels = [n["label"] for n in neighbors_list[doc_idx]]
                if neighbor_labels:
                    majority_label = Counter(neighbor_labels).most_common(1)[0][0]
                    if majority_label in valid_labels:
                        validated_labels.append(majority_label)
                        logger.warning(
                            f"LLM returned invalid label '{lab_str}' "
                            f"for doc {doc_idx+1}. "
                            f"Using majority vote from neighbors: "
                            f"{majority_label}"
                        )
                    else:
                        validated_labels.append(valid_labels[0])
                        logger.warning(
                            f"LLM returned invalid label '{lab_str}' "
                            f"and neighbor majority '{majority_label}' "
                            f"not in valid labels. Using fallback: "
                            f"{valid_labels[0]}"
                        )
                else:
                    validated_labels.append(valid_labels[0])
                    logger.warning(
                        f"LLM returned invalid label '{lab_str}' and no neighbors. "
                        f"Using fallback: {valid_labels[0]}"
                    )
            else:
                validated_labels.append(valid_labels[0])
                logger.warning(
                    f"LLM returned invalid label '{lab_str}'. "
                    f"Valid labels: {valid_labels}. Using fallback: {valid_labels[0]}"
                )

    return validated_labels, invalid_count
