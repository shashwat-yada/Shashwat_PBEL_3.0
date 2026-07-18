"""
utils.py
--------
Small shared text-cleaning helper. Kept dependency-free (no NLTK downloads
needed) so the project runs offline without extra setup steps.
"""

import re


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/punctuation/numbers/extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)               # keep letters only
    text = re.sub(r"\s+", " ", text).strip()             # collapse spaces
    return text
