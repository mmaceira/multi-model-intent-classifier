"""
Classifier Base Module

This module provides the base class for RAG (Retrieval-Augmented Generation)
classifiers. It defines the common interface and functionality shared by all
RAG classifier implementations.

Key Features:
- Abstract base class for RAG classifiers
- Common label management
- Majority voting utility
- Type hints for consistent interfaces

Classes:
- RagClassifierBase: Abstract base class for RAG classifiers

Functions:
- None (Class methods only)

Dependencies:
- abc
- typing
- numpy

Example Usage:
    >>> # Create a custom classifier by inheriting from RagClassifierBase
    >>> class CustomClassifier(RagClassifierBase):
    ...     def predict(self, docs: Sequence[str], **kwargs) -> list[str]:
    ...         # Implement prediction logic
    ...         return predictions
    ...     
    ...     def predict_proba(self, docs: Sequence[str], **kwargs) -> np.ndarray:
    ...         # Implement probability estimation
    ...         return probabilities
"""

from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

class RagClassifierBase(ABC):
    def __init__(self, labels: Sequence[str]):
        self.labels = list(labels)

    @abstractmethod
    def predict(self, docs: Sequence[str], **kwargs) -> list[str]:
        ...

    def _majority_vote(self, neighbor_labels: Sequence[str]) -> str:
        vals, counts = np.unique(neighbor_labels, return_counts=True)
        return vals[counts.argmax()]
