"""\
  Init   module.

Classes:
- None

Functions:
- None

Created: 2025-05-03
"""

import importlib
import pathlib

from .linear_svm import LinearSVMBigrams, LinearSVMClassifier  # noqa: F401
from .naive_bayes import NaiveBayesClassifier  # noqa: F401
from .openai_logreg import OpenAIEmbedLogReg  # noqa: F401
from .transformer_logreg import TransformerLogReg  # noqa: F401

__all__ = [
    "LinearSVMBigrams",
    "LinearSVMClassifier",
    "NaiveBayesClassifier",
    "OpenAIEmbedLogReg",
    "TransformerLogReg",
]

# Make submodules importable as algorithms.<sub>
_src_pkg = importlib.import_module("src.algorithms")
# expose all names
globals().update(_src_pkg.__dict__)

# Ensure pkg resources
__path__ = [str(pathlib.Path(_src_pkg.__file__).parent)]
