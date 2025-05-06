"""Text processing utilities shared across the project.

This module provides common text processing functions used for text analysis
and preprocessing. It includes functions for identifying numeric and financial
terms in text, which are useful for filtering and cleaning text data.

Functions:
- is_numeric_or_financial: Check if a token is numeric or financial
"""

import re
from typing import Set


# Regular expression for matching numeric values (integers or floats)
_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")

# Set of common financial terms and currency symbols
_FINANCE_TOKENS: Set[str] = {
    "usd", "eur", "gbp", "cad", "aud",
    "yen", "jpy", "inr", "cny", "rupee",
    "million", "billion", "thousand", "%",
}


def is_numeric_or_financial(token: str) -> bool:
    """Check if a token is numeric or related to finance.
    
    This function uses a simple heuristic to identify tokens that are either
    numeric values (integers or floats) or common financial terms. It is
    designed to be fast and efficient for use in text preprocessing.
    
    Parameters
    ----------
    token : str
        The token to check.
        
    Returns
    -------
    bool
        True if the token is numeric or a financial term, False otherwise.
        
    Examples
    --------
    >>> is_numeric_or_financial("123")
    True
    
    >>> is_numeric_or_financial("million")
    True
    
    >>> is_numeric_or_financial("company")
    False
    
    >>> is_numeric_or_financial("123.45")
    True
    
    >>> is_numeric_or_financial("USD")
    True
    """
    tk = token.lower()
    return bool(_NUM_RE.match(tk)) or tk in _FINANCE_TOKENS
