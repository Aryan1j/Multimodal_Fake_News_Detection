"""
Data Preprocessing - Build model input strings from title and text columns.
"""

import re
import pandas as pd


def build_input_text(title: str, text: str, sep_token: str = "[SEP]") -> str:
    """
    Build the input text string for the model from title and text.
    
    """
    title = str(title).strip() if title else ""
    text = str(text).strip() if text else ""
    
    if title and text:
        return f"{title} {sep_token} {text}"
    elif title:
        return title
    elif text:
        return text
    else:
        return ""


def add_model_input_column(
    df: pd.DataFrame,
    title_col: str = "title",
    text_col: str = "text",
    output_col: str = "input_text"
) -> pd.DataFrame:
    
    df = df.copy()
    df[output_col] = df.apply(
        lambda row: build_input_text(row[title_col], row[text_col]),
        axis=1
    )
    return df


def _normalize_user_input(text: str) -> str:
   
    # Remove structured label prefixes e.g. "Vaccine Development: "
    text = re.sub(r"^[A-Z][^:\n]{3,40}:\s*", "", text, flags=re.MULTILINE)

    # Remove bullet points and numbered list markers
    text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)

    # Collapse multiple newlines / whitespace into a single space
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def preprocess_user_input(raw_text: str) -> str:

    raw_text = str(raw_text).strip()
    
    if not raw_text:
        return ""

    # --- OOD normalization (new) ---
    raw_text = _normalize_user_input(raw_text)

    if not raw_text:
        return ""

    words = raw_text.split()
    
    # Short text: treat as title only
    if len(words) < 50:
        return raw_text
    
    # Long text: extract first sentence as synthetic title
    first_sentence_end = -1
    for punct in ['. ', '! ', '? ']:
        idx = raw_text.find(punct)
        if idx != -1:
            if first_sentence_end == -1 or idx < first_sentence_end:
                first_sentence_end = idx + 1
    
    if first_sentence_end > 0 and first_sentence_end < len(raw_text) - 10:
        synthetic_title = raw_text[:first_sentence_end].strip()
        remaining_text = raw_text[first_sentence_end:].strip()
        return build_input_text(synthetic_title, remaining_text)
    
    # No clear sentence boundary: use first 10 words as title
    synthetic_title = ' '.join(words[:10])
    remaining_text = ' '.join(words[10:])
    return build_input_text(synthetic_title, remaining_text)