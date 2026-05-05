"""
Length Metrics - Utilities for word count analysis and length-based evaluation.
"""

from typing import List

import numpy as np

from .config import SHORT_TEXT_WORD_THRESHOLD


def word_count(text: str) -> int:
   
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def word_counts(texts: List[str]) -> np.ndarray:
    
    return np.array([word_count(t) for t in texts])


def short_mask(texts: List[str], threshold: int = SHORT_TEXT_WORD_THRESHOLD) -> np.ndarray:
    """
    Create a boolean mask for short texts (word count < threshold).

    """
    counts = word_counts(texts)
    return counts < threshold


def long_mask(texts: List[str], threshold: int = SHORT_TEXT_WORD_THRESHOLD) -> np.ndarray:
    """
    Create a boolean mask for long texts (word count >= threshold).
    """
    counts = word_counts(texts)
    return counts >= threshold


def get_length_segment(text: str, threshold: int = SHORT_TEXT_WORD_THRESHOLD) -> str:
  
    from .config import LENGTH_SEGMENT_SHORT, LENGTH_SEGMENT_LONG
    
    if word_count(text) < threshold:
        return LENGTH_SEGMENT_SHORT
    return LENGTH_SEGMENT_LONG
