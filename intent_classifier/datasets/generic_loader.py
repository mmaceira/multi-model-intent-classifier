"""
Generic dataset loader that reads configuration from config/dataset/{name}/loader.yaml.

This allows adding new datasets by simply creating a config file without writing Python code.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.request import urlopen

import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from intent_classifier.utils.paths import get_repo_root


def load_dataset_from_config(
    dataset_name: str,
    use_oos: bool = False,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    max_val_samples: int | None = None,
    max_classes: int | None = None,
    seed: int = 42,
    csv_path: str | Path | None = None,
    multilabel: bool | None = None,  # If None, read from config
    **kwargs,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Load a dataset using configuration from config/dataset/{dataset_name}/loader.yaml.

    This is a generic loader that supports multiple source types:
    - huggingface: Load from HuggingFace datasets
    - github_json: Load JSON files from GitHub URLs
    - csv: Load from CSV file
    - local_json: Load from local JSON file

    Args:
        dataset_name: Name of the dataset (must have config/dataset/{name}/loader.yaml)
        use_oos: Whether to include out-of-scope examples (if supported)
        max_train_samples: Maximum training samples
        max_test_samples: Maximum test samples
        max_val_samples: Maximum validation samples
        max_classes: Maximum number of classes
        seed: Random seed
        csv_path: Path to CSV file (overrides config if provided)
        multilabel: Override multilabel setting from config
        **kwargs: Additional arguments passed to source-specific loaders

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, classes
    """
    # Load config
    repo_root = get_repo_root()
    config_path = repo_root / "config" / "dataset" / dataset_name / "loader.yaml"

    if not config_path.exists():
        raise ValueError(
            f"Dataset loader config not found: {config_path}. "
            f"Create config/dataset/{dataset_name}/loader.yaml to add this dataset."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Determine multilabel (config takes precedence unless explicitly overridden)
    is_multilabel = multilabel if multilabel is not None else config.get("multilabel", False)

    # Load raw data based on source type
    source_config = config["source"]
    source_type = source_config["type"]

    if source_type == "huggingface":
        raw_data = _load_from_huggingface(source_config)
    elif source_type == "github_json":
        raw_data = _load_from_github_json(source_config)
    elif source_type == "csv":
        csv_path = csv_path or source_config.get("path")
        if not csv_path:
            raise ValueError("CSV source requires 'path' in config or csv_path parameter")
        raw_data = _load_from_csv(csv_path, source_config)
    elif source_type == "local_json":
        json_path = source_config.get("path")
        if not json_path:
            raise ValueError("local_json source requires 'path' in config")
        raw_data = _load_from_local_json(json_path, source_config)
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    # Extract text and labels using field mappings
    fields_config = config["fields"]
    all_texts, all_labels = _extract_fields(raw_data, fields_config, is_multilabel, config)

    # Apply filters
    filters_config = config.get("filters", {})
    if filters_config.get("min_samples_per_label"):
        all_texts, all_labels = _filter_by_min_samples(
            all_texts, all_labels, filters_config["min_samples_per_label"], is_multilabel
        )

    # Handle OOS filtering
    if config.get("has_oos") and not use_oos:
        oos_label = config.get("oos_label", "oos")
        all_texts, all_labels = _filter_oos(all_texts, all_labels, oos_label, is_multilabel)

    # Create or use predefined splits
    splits_config = config["splits"]
    if splits_config["type"] == "predefined":
        X_train, y_train, X_val, y_val, X_test, y_test = _use_predefined_splits(
            raw_data, splits_config, fields_config, is_multilabel, config, use_oos
        )
    else:
        # Create splits from combined data
        X_train, y_train, X_val, y_val, X_test, y_test = _create_splits(
            all_texts, all_labels, splits_config, seed, is_multilabel
        )

    # Apply class and sample limits
    X_train, y_train, X_val, y_val, X_test, y_test, classes = _apply_limits(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        max_classes,
        max_train_samples,
        max_val_samples,
        max_test_samples,
        seed,
        is_multilabel,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, classes


def _load_from_huggingface(source_config: Dict[str, Any]) -> Any:
    """Load dataset from HuggingFace."""
    identifier = source_config["identifier"]
    config_name = source_config.get("config")
    if config_name:
        return load_dataset(identifier, config_name)
    return load_dataset(identifier)


def _load_from_github_json(source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load JSON files from GitHub."""
    base_url = source_config["base_url"]
    domains = source_config.get("domains", [""])
    file_pattern = source_config.get("file_pattern", "*.json")
    fold_range = source_config.get("fold_range", [0, 0])

    all_data = []
    for domain in domains:
        domain_path = f"{domain}/" if domain else ""
        for fold_num in range(fold_range[0], fold_range[1] + 1):
            file_path = file_pattern.format(fold_num=fold_num)
            url = f"{base_url}/{domain_path}{file_path}"
            try:
                with urlopen(url) as response:
                    fold_data = json.loads(response.read())
                    if isinstance(fold_data, list):
                        all_data.extend(fold_data)
                    else:
                        all_data.append(fold_data)
            except Exception as e:
                raise ValueError(f"Failed to load {url}: {e}") from e
    return all_data


def _load_from_csv(csv_path: str | Path, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load data from CSV file."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    delimiter = source_config.get("delimiter", ",")
    encoding = source_config.get("encoding", "utf-8")

    data = []
    with open(csv_path, encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            data.append(dict(row))
    return data


def _load_from_local_json(
    json_path: str | Path, source_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Load data from local JSON file."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data)}")


def _extract_fields(
    raw_data: Any,
    fields_config: Dict[str, Any],
    is_multilabel: bool,
    config: Dict[str, Any],
) -> Tuple[List[str], List[Any]]:
    """Extract text and label fields from raw data."""
    text_field = fields_config["text"]
    label_field = fields_config["label"]

    # Handle field names that might be lists (try multiple field names)
    if isinstance(text_field, list):
        text_field_candidates = text_field
    else:
        text_field_candidates = [text_field]

    if isinstance(label_field, list):
        label_field_candidates = label_field
    else:
        label_field_candidates = [label_field]

    texts = []
    labels = []

    # Handle HuggingFace dataset format
    combine_text = fields_config.get("combine_text_fields", False)
    if hasattr(raw_data, "keys") and "train" in raw_data.keys():
        # HuggingFace dataset with splits - extract from all splits
        for split_name in raw_data.keys():
            split = raw_data[split_name]
            for row in split:
                if (
                    combine_text
                    and isinstance(text_field_candidates, list)
                    and len(text_field_candidates) > 1
                ):
                    # Combine multiple text fields
                    text_parts = []
                    for field in text_field_candidates:
                        value = _get_field_value(row, [field])
                        if value:
                            text_parts.append(str(value))
                    text = " ".join(text_parts) if text_parts else None
                else:
                    text = _get_field_value(row, text_field_candidates)
                label = _get_field_value(row, label_field_candidates)

                if text and label is not None:
                    texts.append(text)
                    labels.append(label)
    else:
        # List of dictionaries
        combine_text = fields_config.get("combine_text_fields", False)
        for row in raw_data:
            if (
                combine_text
                and isinstance(text_field_candidates, list)
                and len(text_field_candidates) > 1
            ):
                # Combine multiple text fields
                text_parts = []
                for field in text_field_candidates:
                    value = _get_field_value(row, [field])
                    if value:
                        text_parts.append(str(value))
                text = " ".join(text_parts) if text_parts else None
            else:
                text = _get_field_value(row, text_field_candidates)
            label = _get_field_value(row, label_field_candidates)

            if text and label is not None:
                texts.append(text)
                labels.append(label)

    # Process labels based on config
    label_processing = config.get("label_processing", {})

    # Handle numeric to string conversion (for HuggingFace)
    if label_processing.get("convert_numeric_to_string"):
        # For HuggingFace datasets, we might need to look up label names
        if hasattr(raw_data, "keys") and "train" in raw_data.keys():
            # Try to get label names from features
            train_split = raw_data["train"]
            if hasattr(train_split, "features"):
                label_feature_name = label_processing.get(
                    "label_names_feature", label_field_candidates[0]
                )
                if label_feature_name in train_split.features:
                    feature = train_split.features[label_feature_name]
                    if hasattr(feature, "names"):
                        label_names = feature.names
                        labels = [
                            (
                                label_names[label]
                                if isinstance(label, int) and label < len(label_names)
                                else str(label)
                            )
                            for label in labels
                        ]
                    else:
                        labels = [str(label) for label in labels]
                else:
                    labels = [str(label) for label in labels]
            else:
                labels = [str(label) for label in labels]
        else:
            labels = [str(label) for label in labels]

    # Handle multilabel normalization
    if is_multilabel and label_processing.get("ensure_list"):
        labels = [
            label if isinstance(label, list) else [label] if label else [] for label in labels
        ]
        if label_processing.get("normalize_to_list"):
            labels = [
                [str(l)] if not isinstance(l, list) else [str(item) for item in l] for l in labels
            ]
        # Split comma-separated strings into separate labels
        if label_processing.get("split_comma_separated"):
            labels = [
                (
                    [tag.strip() for tag in str(label[0]).split(",") if tag.strip()]
                    if isinstance(label, list)
                    and len(label) == 1
                    and isinstance(label[0], str)
                    and "," in label[0]
                    else label
                )
                for label in labels
            ]

    return texts, labels


def _get_field_value(row: Dict[str, Any] | Any, field_candidates: List[str]) -> Any:
    """Get field value trying multiple candidate field names."""
    if isinstance(row, dict):
        for field in field_candidates:
            if field in row:
                return row[field]
        return None
    else:
        # For HuggingFace dataset rows, try attribute access
        for field in field_candidates:
            if hasattr(row, field):
                return getattr(row, field)
        return None


def _filter_by_min_samples(
    texts: List[str],
    labels: List[Any],
    min_samples: int,
    is_multilabel: bool,
) -> Tuple[List[str], List[Any]]:
    """Filter out labels with fewer than min_samples samples."""
    if is_multilabel:
        # Count all labels
        all_tags = []
        for label_list in labels:
            if isinstance(label_list, list):
                all_tags.extend(label_list)
            else:
                all_tags.append(label_list)
        label_counts = Counter(all_tags)
        valid_labels = {label for label, count in label_counts.items() if count >= min_samples}

        filtered_texts = []
        filtered_labels = []
        for text, label_list in zip(texts, labels, strict=False):
            if isinstance(label_list, list):
                filtered_label_list = [tag for tag in label_list if tag in valid_labels]
                if filtered_label_list:
                    filtered_texts.append(text)
                    filtered_labels.append(filtered_label_list)
            else:
                if label_list in valid_labels:
                    filtered_texts.append(text)
                    filtered_labels.append(label_list)
    else:
        label_counts = Counter(labels)
        valid_labels = {label for label, count in label_counts.items() if count >= min_samples}
        filtered_texts = [
            text for text, label in zip(texts, labels, strict=False) if label in valid_labels
        ]
        filtered_labels = [label for label in labels if label in valid_labels]

    return filtered_texts, filtered_labels


def _filter_oos(
    texts: List[str],
    labels: List[Any],
    oos_label: str,
    is_multilabel: bool,
) -> Tuple[List[str], List[Any]]:
    """Filter out out-of-scope examples."""
    if is_multilabel:
        filtered_texts = []
        filtered_labels = []
        for text, label_list in zip(texts, labels, strict=False):
            if isinstance(label_list, list):
                if oos_label not in label_list:
                    filtered_texts.append(text)
                    filtered_labels.append(label_list)
            else:
                if label_list != oos_label:
                    filtered_texts.append(text)
                    filtered_labels.append(label_list)
        return filtered_texts, filtered_labels
    else:
        filtered_texts = []
        filtered_labels = []
        for text, label in zip(texts, labels, strict=False):
            if label != oos_label:
                filtered_texts.append(text)
                filtered_labels.append(label)
        return filtered_texts, filtered_labels


def _use_predefined_splits(
    raw_data: Any,
    splits_config: Dict[str, Any],
    fields_config: Dict[str, Any],
    is_multilabel: bool,
    config: Dict[str, Any],
    use_oos: bool,
) -> Tuple[List[str], List[Any], List[str], List[Any], List[str], List[Any]]:
    """Extract data from predefined splits (e.g., HuggingFace dataset)."""
    train_split_name = splits_config["train_split"]
    val_split_name = splits_config["val_split"]
    test_split_name = splits_config["test_split"]

    text_field = fields_config["text"]
    label_field = fields_config["label"]

    if isinstance(text_field, list):
        text_field = text_field[0]
    if isinstance(label_field, list):
        label_field = label_field[0]

    def _extract_split(split_name: str):
        split = raw_data[split_name]
        texts = []
        labels = []

        for row in split:
            text = _get_field_value(row, [text_field])
            label = _get_field_value(row, [label_field])

            # Handle OOS filtering
            if config.get("has_oos") and not use_oos:
                oos_label = config.get("oos_label", "oos")
                if (is_multilabel and isinstance(label, list) and oos_label in label) or (
                    not is_multilabel and label == oos_label
                ):
                    continue

            if text and label is not None:
                texts.append(text)
                labels.append(label)

        # Handle label processing
        label_processing = config.get("label_processing", {})
        if label_processing.get("convert_numeric_to_string"):
            # Try to get label names from features
            if hasattr(split, "features") and label_field in split.features:
                feature = split.features[label_field]
                if hasattr(feature, "names"):
                    label_names = feature.names
                    labels = [
                        (
                            label_names[label]
                            if isinstance(label, int) and label < len(label_names)
                            else str(label)
                        )
                        for label in labels
                    ]
                else:
                    labels = [str(label) for label in labels]
            else:
                labels = [str(label) for label in labels]

        return texts, labels

    X_train, y_train = _extract_split(train_split_name)
    X_val, y_val = _extract_split(val_split_name)
    X_test, y_test = _extract_split(test_split_name)

    return X_train, y_train, X_val, y_val, X_test, y_test


def _create_splits(
    texts: List[str],
    labels: List[Any],
    splits_config: Dict[str, Any],
    seed: int,
    is_multilabel: bool,
) -> Tuple[List[str], List[Any], List[str], List[Any], List[str], List[Any]]:
    """Create train/val/test splits from combined data."""
    test_size = splits_config.get("test_size", 0.2)
    val_size = splits_config.get("val_size", 0.1)
    stratify = splits_config.get("stratify", False) and not is_multilabel

    # First split: test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        stratify=labels if stratify else None,
        random_state=seed,
    )

    # Second split: train/val
    val_size_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_relative,
        stratify=y_temp if stratify else None,
        random_state=seed,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def _apply_limits(
    X_train: List[str],
    y_train: List[Any],
    X_val: List[str],
    y_val: List[Any],
    X_test: List[str],
    y_test: List[Any],
    max_classes: int | None,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    seed: int,
    is_multilabel: bool,
) -> Tuple[List[str], List[Any], List[str], List[Any], List[str], List[Any], List[str]]:
    """Apply class and sample limits."""
    # Get all classes
    if is_multilabel:
        all_tags = []
        for label_list in y_train + y_val + y_test:
            if isinstance(label_list, list):
                all_tags.extend(label_list)
            else:
                all_tags.append(label_list)
        all_classes = sorted(set(all_tags))
    else:
        all_classes = sorted(set(y_train + y_val + y_test))

    # Limit classes
    if max_classes is not None and len(all_classes) > max_classes:
        random.seed(seed)
        selected_classes = sorted(random.sample(all_classes, max_classes))

        if is_multilabel:
            # Filter to keep samples with at least one selected class
            def _filter_multilabel(X, y):
                filtered_X = []
                filtered_y = []
                for text, label_list in zip(X, y, strict=False):
                    if isinstance(label_list, list):
                        filtered_labels = [l for l in label_list if l in selected_classes]
                        if filtered_labels:
                            filtered_X.append(text)
                            filtered_y.append(filtered_labels)
                    else:
                        if label_list in selected_classes:
                            filtered_X.append(text)
                            filtered_y.append([label_list])
                return filtered_X, filtered_y

            X_train, y_train = _filter_multilabel(X_train, y_train)
            X_val, y_val = _filter_multilabel(X_val, y_val)
            X_test, y_test = _filter_multilabel(X_test, y_test)
        else:
            # Filter single-label
            X_train = [
                text
                for text, label in zip(X_train, y_train, strict=False)
                if label in selected_classes
            ]
            y_train = [label for label in y_train if label in selected_classes]
            X_val = [
                text for text, label in zip(X_val, y_val, strict=False) if label in selected_classes
            ]
            y_val = [label for label in y_val if label in selected_classes]
            X_test = [
                text
                for text, label in zip(X_test, y_test, strict=False)
                if label in selected_classes
            ]
            y_test = [label for label in y_test if label in selected_classes]

        classes = selected_classes
    else:
        classes = all_classes

    # Limit samples
    if max_train_samples is not None and len(X_train) > max_train_samples:
        X_train, y_train = _sample_data(X_train, y_train, max_train_samples, seed, is_multilabel)
    if max_val_samples is not None and len(X_val) > max_val_samples:
        X_val, y_val = _sample_data(X_val, y_val, max_val_samples, seed + 1, is_multilabel)
    if max_test_samples is not None and len(X_test) > max_test_samples:
        X_test, y_test = _sample_data(X_test, y_test, max_test_samples, seed + 2, is_multilabel)

    # Recompute classes from final data
    if is_multilabel:
        all_tags = []
        for label_list in y_train + y_val + y_test:
            if isinstance(label_list, list):
                all_tags.extend(label_list)
            else:
                all_tags.append(label_list)
        classes = sorted(set(all_tags))
    else:
        classes = sorted(set(y_train + y_val + y_test))

    return X_train, y_train, X_val, y_val, X_test, y_test, classes


def _sample_data(
    texts: List[str],
    labels: List[Any],
    max_samples: int,
    seed: int,
    is_multilabel: bool,
) -> Tuple[List[str], List[Any]]:
    """Sample data (stratified for single-label, random for multilabel)."""
    if max_samples >= len(texts):
        return texts, labels

    if is_multilabel:
        # Random sampling for multilabel
        combined = list(zip(texts, labels, strict=False))
        random.seed(seed)
        random.shuffle(combined)
        sampled = combined[:max_samples]
        sampled_texts, sampled_labels = zip(*sampled, strict=False)
        return list(sampled_texts), list(sampled_labels)
    else:
        # Stratified sampling for single-label
        return _stratified_sample_singlelabel(texts, labels, max_samples, seed)


def _stratified_sample_singlelabel(
    texts: List[str], labels: List[str], max_samples: int, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Perform stratified sampling to ensure all classes are represented.

    This function:
    1. Groups samples by class
    2. Ensures at least 1 sample per class (if possible)
    3. Distributes remaining samples proportionally across classes
    4. Maintains class balance as much as possible
    """
    if max_samples >= len(texts):
        return texts, labels

    # Group samples by class
    class_to_samples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for text, label in zip(texts, labels, strict=False):
        class_to_samples[label].append((text, label))

    num_classes = len(class_to_samples)
    class_names = list(class_to_samples.keys())

    # If we need fewer samples than classes, we can't represent all classes
    if max_samples < num_classes:
        # Take at least 1 sample from each of the first max_samples classes
        sampled_texts = []
        sampled_labels = []
        for class_name in class_names[:max_samples]:
            if class_to_samples[class_name]:
                text, label = class_to_samples[class_name][0]
                sampled_texts.append(text)
                sampled_labels.append(label)
        return sampled_texts, sampled_labels

    # Strategy: Ensure at least 1 sample per class, then distribute the rest
    samples_per_class = max_samples // num_classes
    remainder = max_samples % num_classes

    sampled_texts = []
    sampled_labels = []

    # Set random seed for reproducibility
    random.seed(seed)

    # Shuffle each class's samples to randomize selection
    for class_name in class_names:
        class_samples = class_to_samples[class_name].copy()
        random.shuffle(class_samples)

        # Calculate how many samples to take from this class
        num_to_take = samples_per_class
        if class_names.index(class_name) < remainder:
            num_to_take += 1

        # Take samples (but don't exceed what's available)
        num_to_take = min(num_to_take, len(class_samples))
        for i in range(num_to_take):
            text, label = class_samples[i]
            sampled_texts.append(text)
            sampled_labels.append(label)

    # Shuffle the final result to mix classes
    combined = list(zip(sampled_texts, sampled_labels, strict=False))
    random.shuffle(combined)
    sampled_texts, sampled_labels = zip(*combined, strict=False)

    return list(sampled_texts), list(sampled_labels)
