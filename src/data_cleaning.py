
import re
from typing import Optional, Tuple

import pandas as pd

from .config import DATA_CSV


def clean_welfake_dataframe(
    df: pd.DataFrame,
    lowercase: bool = False
) -> pd.DataFrame:
    """
    Clean a WELFake-shaped DataFrame.
    
    Operations:
    - Drop unnamed index columns
    - Require title, text, label columns
    - Strip and normalize whitespace
    - Optional lowercase conversion
    - Drop rows with both title and text empty
    - Coerce labels to numeric
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Drop unnamed index columns (common in WELFake exports)
    unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    
    # Check required columns
    required_cols = {'title', 'text', 'label'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Fill NaN with empty string for text columns
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    
    # Convert to string and strip whitespace
    df['title'] = df['title'].astype(str).str.strip()
    df['text'] = df['text'].astype(str).str.strip()
    
    # Normalize whitespace (collapse multiple spaces)
    df['title'] = df['title'].apply(lambda x: re.sub(r'\s+', ' ', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    # Optional lowercase conversion
    if lowercase:
        df['title'] = df['title'].str.lower()
        df['text'] = df['text'].str.lower()
    
    # Drop rows where both title and text are empty
    empty_mask = (df['title'] == '') & (df['text'] == '')
    if empty_mask.any():
        n_dropped = empty_mask.sum()
        print(f"Dropping {n_dropped} rows with empty title and text")
        df = df[~empty_mask].reset_index(drop=True)
    
    # Coerce labels to numeric
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    
    # Drop rows with invalid labels
    invalid_labels = df['label'].isna()
    if invalid_labels.any():
        n_dropped = invalid_labels.sum()
        print(f"Dropping {n_dropped} rows with invalid labels")
        df = df[~invalid_labels].reset_index(drop=True)
    
    # Convert label to int
    df['label'] = df['label'].astype(int)
    
    return df


def drop_duplicate_articles(
    df: pd.DataFrame,
    report_conflicts: bool = True
) -> pd.DataFrame:
   
    original_len = len(df)
    
    if report_conflicts:
        # Find duplicates and check for label conflicts
        dup_mask = df.duplicated(subset=['title', 'text'], keep=False)
        if dup_mask.any():
            dup_groups = df[dup_mask].groupby(['title', 'text'])['label']
            conflicts = 0
            for (title, text), labels in dup_groups:
                unique_labels = labels.unique()
                if len(unique_labels) > 1:
                    conflicts += 1
                    if conflicts <= 5:  # Only print first 5
                        print(f"Label conflict on duplicate: '{title[:50]}...' has labels {list(unique_labels)}")
            
            if conflicts > 5:
                print(f"... and {conflicts - 5} more label conflicts")
    
    # Drop duplicates keeping first
    df = df.drop_duplicates(subset=['title', 'text'], keep='first').reset_index(drop=True)
    
    n_dropped = original_len - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} duplicate articles")
    
    return df


def load_and_clean_csv(
    path: Optional[str] = None,
    lowercase: bool = False,
    dedupe: bool = False
) -> pd.DataFrame:
   
    csv_path = path or str(DATA_CSV)
    
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Clean the dataframe
    df = clean_welfake_dataframe(df, lowercase=lowercase)
    print(f"After cleaning: {len(df)} rows")
    
    # Optional deduplication
    if dedupe:
        df = drop_duplicate_articles(df)
        print(f"After deduplication: {len(df)} rows")
    
    return df


def save_cleaned_sample(
    df: pd.DataFrame,
    output_path: str,
    n_samples: Optional[int] = None,
    random_state: int = 42
) -> None:
    
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_state)
    
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")


def load_prepared_data(
    data_path: Optional[str] = None,
    lowercase: bool = False,
    dedupe: bool = False,
    max_samples: Optional[int] = None
) -> Tuple[pd.DataFrame, str]:
   
    csv_path = data_path or str(DATA_CSV)
    
    # Load and clean
    df = load_and_clean_csv(csv_path, lowercase=lowercase, dedupe=dedupe)
    
    # Optional sampling
    if max_samples is not None and max_samples < len(df):
        print(f"Sampling {max_samples} rows from {len(df)} total")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    return df, csv_path
