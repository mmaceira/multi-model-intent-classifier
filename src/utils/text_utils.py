"""Text‑processing helpers shared by analysis scripts."""

import re

_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")  # integer or float

_FINANCE_TOKENS = {
    "usd", "eur", "gbp", "cad", "aud",
    "yen", "jpy", "inr", "cny", "rupee",
    "million", "billion", "thousand", "%",
}

def is_numeric_or_financial(token: str) -> bool:
    """Return *True* if *token* is numeric or finance‑related.
    
    The heuristic is intentionally minimalist – it only needs to be good
    enough for stop‑word curation and vocabulary trimming.
    """
    tk = token.lower()
    return bool(_NUM_RE.match(tk)) or tk in _FINANCE_TOKENS
