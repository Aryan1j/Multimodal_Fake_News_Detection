"""
Splits - Stratified train/val/test index management with JSON persistence.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION,
    RANDOM_SEED, SPLIT_JSON_PATH
)


# Split file version for compatibility checking
SPLIT_VERSION = 2


def compute_stratified_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRACTION,
    val_frac: float = VAL_FRACTION,
    test_frac: float = TEST_FRACTION,
    random_seed: int = RANDOM_SEED
) -> Tuple[List[int], List[int], List[int]]:
    """
    Compute stratified train/val/test split indices.
    
    Uses sklearn's train_test_split twice to achieve 70/10/20 split:
    1. Split into train+val (80%) vs test (20%)
    2. Split train+val into train (87.5% of 80% = 70%) vs val (12.5% of 80% = 10%)
    
    Args:
        df: DataFrame with 'label' column for stratification
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    indices = np.arange(len(df))
    labels = df['label'].values
    
    # First split: train+val vs test
    train_val_size = train_frac + val_frac
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_frac,
        stratify=labels,
        random_state=random_seed
    )
    
    # Second split: train vs val from train_val
    val_relative_frac = val_frac / train_val_size
    train_val_labels = labels[train_val_indices]
    
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_relative_frac,
        stratify=train_val_labels,
        random_state=random_seed
    )
    
    return (
    [int(i) for i in train_indices],
    [int(i) for i in val_indices],
    [int(i) for i in test_indices]
)


def create_split_metadata(
    csv_path: str,
    n_rows: int,
    lowercase: bool,
    dedupe: bool,
    max_samples: Optional[int]
) -> Dict[str, Any]:
    """
    Create metadata dictionary for split file.
    
    Args:
        csv_path: Path to the source CSV
        n_rows: Number of rows in the cleaned DataFrame
        lowercase: Whether lowercase cleaning was applied
        dedupe: Whether deduplication was applied
        max_samples: Max samples limit (None if not limited)
        
    Returns:
        Metadata dictionary
    """
    return {
        'version': SPLIT_VERSION,
        'csv_path': str(csv_path),
        'n_rows': n_rows,
        'lowercase': lowercase,
        'dedupe': dedupe,
        'max_samples': max_samples
    }


def split_metadata_matches_current(
    saved_metadata: Dict[str, Any],
    csv_path: str,
    n_rows: int,
    lowercase: bool,
    dedupe: bool,
    max_samples: Optional[int]
) -> bool:
    """
    Check if saved split metadata matches current data configuration.
    
    Args:
        saved_metadata: Metadata from saved split file
        csv_path: Current CSV path
        n_rows: Current row count
        lowercase: Current lowercase setting
        dedupe: Current dedupe setting
        max_samples: Current max_samples setting
        
    Returns:
        True if metadata matches, False otherwise
    """
    if saved_metadata.get('version') != SPLIT_VERSION:
        return False
    
    # Normalize paths for comparison
    saved_csv = Path(saved_metadata.get('csv_path', '')).resolve()
    current_csv = Path(csv_path).resolve()
    
    return (
        saved_csv == current_csv and
        saved_metadata.get('n_rows') == n_rows and
        saved_metadata.get('lowercase') == lowercase and
        saved_metadata.get('dedupe') == dedupe and
        saved_metadata.get('max_samples') == max_samples
    )


def save_split(
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    metadata: Dict[str, Any],
    path: Path = SPLIT_JSON_PATH
) -> None:
    """
    Save split indices and metadata to JSON file.
    
    Args:
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Test set indices
        metadata: Split metadata dictionary
        path: Path to save the split file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    split_data = {
        'metadata': metadata,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }
    
    with open(path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"Saved split to {path}")
    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")


def load_split(path: Path = SPLIT_JSON_PATH) -> Optional[Dict[str, Any]]:
    """
    Load split data from JSON file.
    
    Args:
        path: Path to the split file
        
    Returns:
        Split data dictionary or None if file doesn't exist
    """
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        return json.load(f)


def apply_saved_indices(
    df: pd.DataFrame,
    split_data: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply saved indices to split a DataFrame.
    
    Args:
        df: Full DataFrame to split
        split_data: Split data with train/val/test indices
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = df.iloc[split_data['train_indices']].reset_index(drop=True)
    val_df = df.iloc[split_data['val_indices']].reset_index(drop=True)
    test_df = df.iloc[split_data['test_indices']].reset_index(drop=True)
    
    return train_df, val_df, test_df


def ensure_data_split(
    df: pd.DataFrame,
    csv_path: str,
    lowercase: bool = False,
    dedupe: bool = False,
    max_samples: Optional[int] = None,
    force_new: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ensure a valid data split exists, creating one if needed.
    
    This is the main entry point for getting train/val/test DataFrames.
    It loads an existing split if metadata matches, otherwise creates a new one.
    
    Args:
        df: Cleaned DataFrame
        csv_path: Path to source CSV
        lowercase: Whether lowercase cleaning was applied
        dedupe: Whether deduplication was applied
        max_samples: Max samples limit
        force_new: Force creation of new split even if one exists
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n_rows = len(df)
    
    if not force_new:
        # Try to load existing split
        split_data = load_split()
        
        if split_data is not None:
            metadata = split_data.get('metadata', {})
            
            if split_metadata_matches_current(
                metadata, csv_path, n_rows, lowercase, dedupe, max_samples
            ):
                print("Using existing data split from", SPLIT_JSON_PATH)
                return apply_saved_indices(df, split_data)
            else:
                print("Split metadata mismatch - creating new split")
    
    # Create new split
    print("Computing new stratified split...")
    train_indices, val_indices, test_indices = compute_stratified_split(df)
    
    # Create and save metadata
    metadata = create_split_metadata(csv_path, n_rows, lowercase, dedupe, max_samples)
    save_split(train_indices, val_indices, test_indices, metadata)
    
    # Apply indices
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    return train_df, val_df, test_df


def remove_stale_split_before_training(
    csv_path: str,
    n_rows: int,
    lowercase: bool,
    dedupe: bool,
    max_samples: Optional[int]
) -> None:
    """
    Remove stale split file if training args have changed.
    
    Call this before ensure_data_split during training to ensure
    splits are regenerated when training configuration changes.
    
    Args:
        csv_path: Path to source CSV
        n_rows: Number of rows in cleaned DataFrame
        lowercase: Whether lowercase cleaning is applied
        dedupe: Whether deduplication is applied
        max_samples: Max samples limit
    """
    split_data = load_split()
    
    if split_data is None:
        return
    
    metadata = split_data.get('metadata', {})
    
    if not split_metadata_matches_current(
        metadata, csv_path, n_rows, lowercase, dedupe, max_samples
    ):
        print(f"Removing stale split file: {SPLIT_JSON_PATH}")
        os.remove(SPLIT_JSON_PATH)
