"""\
Classifier Base module.

Classes:
- RagClassifierBase

Functions:
- None

Created: 2025-05-03
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
