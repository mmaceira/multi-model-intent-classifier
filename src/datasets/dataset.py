"""\
Reuters Dataset Module.

This module provides functions for loading and manipulating the Reuters-21578 
text classification dataset with various sampling strategies.

Functions:
    get_dataset: Common entry point for all dataset loading strategies
    _extract_fileids: Extract file IDs based on selection criteria
    _prepare_dataset: Convert file IDs to text and labels
    _load_standard_split: Load the standard train/test split
    _load_small_test_dataset: Load a small balanced test dataset

Created: 2025-05-03
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Callable, Literal
import nltk
from nltk.corpus import reuters
from collections import Counter
import warnings as _warnings
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type aliases for better code readability
DatasetSplit = Tuple[List[str], List[str], List[str], List[str], List[str]]
FileIDs = List[str]
SplitType = Literal["standard", "test"]


def get_dataset(
    split_type: SplitType = "standard", 
    n_classes: Optional[int] = None,
    n_samples_per_class: Optional[int] = None,
    random_seed: Optional[int] = None
) -> DatasetSplit:
    """
    Main entry point for loading Reuters dataset with different split strategies.
    
    Args:
        split_type: The type of dataset split to use:
            - "standard": Use NLTK's train/test split
            - "test": Small balanced dataset with equal samples per class
        n_classes: Number of most frequent classes to include (None for all)
        n_samples_per_class: Maximum samples per class (only used with "test" split)
        random_seed: Random seed for reproducibility in sampling
    
    Returns:
        Tuple containing:
        - X_train: List of training document texts
        - y_train: List of training document labels
        - X_test: List of test document texts
        - y_test: List of test document labels
        - classes: List of class labels
    """
    # Log dataset configuration
    logger.info(f"Loading Reuters dataset with configuration:")
    logger.info(f"  - Split type: {split_type}")
    logger.info(f"  - Number of classes: {n_classes if n_classes is not None else 'all'}")
    if split_type == "test":
        logger.info(f"  - Samples per class: {n_samples_per_class or 20}")
    logger.info(f"  - Random seed: {random_seed}")
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        logger.info(f"Random seed set to {random_seed}")
    
    # Download Reuters dataset if not already available
    nltk.download("reuters", quiet=True)
    
    # Route to appropriate dataset loading function
    if split_type == "standard":
        return _load_standard_split(n_classes)
    elif split_type == "test":
        return _load_small_test_dataset(
            n_samples_per_class or 20,
            n_classes or 7
        )
    else:
        raise ValueError(f"Invalid split_type: {split_type}")


def _extract_fileids(selection_criteria: Callable[[str], bool]) -> FileIDs:
    """
    Extract file IDs based on provided selection criteria.
    
    Args:
        selection_criteria: Function that takes a file ID and returns True if it should be included
    
    Returns:
        List of file IDs that match the criteria
    """
    return [fid for fid in reuters.fileids() if selection_criteria(fid)]


def _get_top_classes(fileids: FileIDs, n_classes: Optional[int] = None) -> List[str]:
    """
    Determine the most frequent classes in the given file IDs.
    
    Args:
        fileids: List of file IDs to analyze
        n_classes: Number of most frequent classes to return (None for all classes)
    
    Returns:
        List of the most frequent class labels
    """
    categories = [reuters.categories(fid)[0] for fid in fileids]
    if n_classes:
        return [cat for cat, _ in Counter(categories).most_common(n_classes)]
    return sorted(set(categories))


def _filter_by_classes(fileids: FileIDs, classes: List[str]) -> FileIDs:
    """
    Filter file IDs to only include documents with the given classes.
    
    Args:
        fileids: List of file IDs to filter
        classes: List of class labels to include
    
    Returns:
        Filtered list of file IDs
    """
    return [fid for fid in fileids if reuters.categories(fid)[0] in classes]


def _prepare_dataset(train_ids: FileIDs, test_ids: FileIDs) -> DatasetSplit:
    """
    Convert file IDs to document text and labels for train and test sets.
    
    Args:
        train_ids: List of file IDs for training set
        test_ids: List of file IDs for test set
    
    Returns:
        X_train, y_train, X_test, y_test, classes
    """
    X_train = [reuters.raw(fid) for fid in train_ids]
    y_train = [reuters.categories(fid)[0] for fid in train_ids]
    X_test = [reuters.raw(fid) for fid in test_ids]
    y_test = [reuters.categories(fid)[0] for fid in test_ids]
    
    # Get all unique class labels from both train and test
    classes = sorted(set(y_train + y_test))
    
    # Log dataset statistics
    logger.info(f"Dataset prepared with:")
    logger.info(f"  - Training samples: {len(X_train)}")
    logger.info(f"  - Test samples: {len(X_test)}")
    logger.info(f"  - Classes: {len(classes)}")
    
    return X_train, y_train, X_test, y_test, classes


def _load_standard_split(n_classes: Optional[int] = None) -> DatasetSplit:
    """
    Load Reuters dataset using NLTK's standard train/test split.
    
    Args:
        n_classes: Number of most frequent classes to include (None for all)
    
    Returns:
        X_train, y_train, X_test, y_test, classes
    """
    logger.info("Loading standard train/test split")
    
    # Get standard train/test split
    train_ids = _extract_fileids(lambda fid: fid.startswith("train"))
    test_ids = _extract_fileids(lambda fid: fid.startswith("test"))
    
    # Determine top classes from training set
    top_classes = _get_top_classes(train_ids, n_classes)
    logger.info(f"Selected {len(top_classes)} classes: {', '.join(top_classes[:5])}{' and more...' if len(top_classes) > 5 else ''}")
    
    # Filter documents to only include selected classes
    train_ids = _filter_by_classes(train_ids, top_classes)
    test_ids = _filter_by_classes(test_ids, top_classes)
    
    return _prepare_dataset(train_ids, test_ids)


def _load_small_test_dataset(n_samples_per_class: int = 20, n_classes: int = 7) -> DatasetSplit:
    """
    Load a small balanced dataset with equal samples per class for testing purposes.
    
    Args:
        n_samples_per_class: Maximum number of samples per class (default: 20)
        n_classes: Number of classes to include (default: 7)
    
    Returns:
        X_train, y_train, X_test, y_test, classes
    """
    logger.info(f"Loading small test dataset with {n_samples_per_class} samples per class across {n_classes} classes")
    
    # Get all file IDs
    all_ids = reuters.fileids()
    
    # Get most common classes
    all_categories = [reuters.categories(fid)[0] for fid in all_ids]
    top_classes = [cls for cls, _ in Counter(all_categories).most_common(n_classes)]
    logger.info(f"Selected classes: {', '.join(top_classes)}")
    
    # Group documents by class
    class_docs: Dict[str, List[str]] = {cls: [] for cls in top_classes}
    for fid in all_ids:
        category = reuters.categories(fid)[0]
        if category in top_classes:
            class_docs[category].append(fid)
    
    # Sample documents and create train/test split
    X_train, y_train, X_test, y_test = [], [], [], []
    for cls, docs in class_docs.items():
        # Shuffle documents
        random.shuffle(docs)
        
        # Limit to n_samples_per_class docs per class
        docs = docs[:n_samples_per_class]
        
        # Split into train/test (70/30 split)
        train_size = int(0.7 * len(docs))
        train_docs = docs[:train_size]
        test_docs = docs[train_size:]
        
        # Add to dataset
        X_train.extend([reuters.raw(fid) for fid in train_docs])
        y_train.extend([cls] * len(train_docs))
        X_test.extend([reuters.raw(fid) for fid in test_docs])
        y_test.extend([cls] * len(test_docs))
        
        logger.info(f"  - Class '{cls}': {len(train_docs)} train, {len(test_docs)} test")
    
    return X_train, y_train, X_test, y_test, top_classes


def test_dataset_loading():
    """
    Test function to debug dataset loading with current configuration.
    """
    # Test with current config values
    print("\nTesting dataset loading with current configuration:")
    print("=================================================")
    
    # Test standard split
    print("\n1. Testing standard split:")
    X_train, y_train, X_test, y_test, classes = get_dataset(
        split_type="standard",
        n_classes=10
    )
    print(f"Standard split results:")
    print(f"- Train samples: {len(X_train)}")
    print(f"- Test samples: {len(X_test)}")
    print(f"- Classes: {len(classes)}")
    print(f"- Class distribution in train: {Counter(y_train)}")
    print(f"- Class distribution in test: {Counter(y_test)}")
    
    # Test small test dataset
    print("\n2. Testing small test dataset:")
    X_train, y_train, X_test, y_test, classes = get_dataset(
        split_type="test",
        n_classes=10,
        n_samples_per_class=20
    )
    print(f"Small test dataset results:")
    print(f"- Train samples: {len(X_train)}")
    print(f"- Test samples: {len(X_test)}")
    print(f"- Classes: {len(classes)}")
    print(f"- Class distribution in train: {Counter(y_train)}")
    print(f"- Class distribution in test: {Counter(y_test)}")

if __name__ == "__main__":
    test_dataset_loading()

