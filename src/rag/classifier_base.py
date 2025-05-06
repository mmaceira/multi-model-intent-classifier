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
    RagClassifierBase: Abstract base class for RAG classifiers
        - Implements common functionality for all RAG classifiers
        - Defines abstract methods that must be implemented by subclasses
        - Provides utility methods for label management and voting

Functions:
    None (Class methods only)

Dependencies:
    abc: For abstract base class functionality
    typing: For type hints
    numpy: For numerical operations

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
    """
    Abstract base class for RAG classifiers.
    
    This class defines the common interface and functionality that all RAG classifiers
    must implement. It provides basic label management and utility methods for
    majority voting.
    
    Attributes:
        labels (list[str]): List of possible class labels for classification
        
    Methods:
        predict: Abstract method for making predictions (must be implemented)
        predict_proba: Abstract method for probability estimates (must be implemented)
        _majority_vote: Utility method for majority voting
    """
    
    def __init__(self, labels: Sequence[str]) -> None:
        """
        Initialize the classifier with a set of possible labels.
        
        Args:
            labels (Sequence[str]): List of possible class labels
        """
        self.labels = list(labels)

    @abstractmethod
    def predict(self, docs: Sequence[str], **kwargs) -> list[str]:
        """
        Abstract method for making predictions on input documents.
        
        Args:
            docs (Sequence[str]): Sequence of documents to classify
            **kwargs: Additional arguments specific to the implementation
            
        Returns:
            list[str]: List of predicted labels for each document
        """
        pass

    @abstractmethod
    def predict_proba(self, docs: Sequence[str], **kwargs) -> np.ndarray:
        """
        Abstract method for estimating class probabilities.
        
        Args:
            docs (Sequence[str]): Sequence of documents to classify
            **kwargs: Additional arguments specific to the implementation
            
        Returns:
            np.ndarray: Array of probability estimates for each class
        """
        pass

    def _majority_vote(self, neighbor_labels: Sequence[str]) -> str:
        """
        Perform majority voting on a sequence of labels.
        
        Args:
            neighbor_labels (Sequence[str]): Sequence of labels to vote on
            
        Returns:
            str: The most frequent label in the sequence
            
        Example:
            >>> classifier = RagClassifierBase(['A', 'B', 'C'])
            >>> classifier._majority_vote(['A', 'B', 'A', 'C', 'A'])
            'A'
        """
        vals, counts = np.unique(neighbor_labels, return_counts=True)
        return vals[counts.argmax()]
