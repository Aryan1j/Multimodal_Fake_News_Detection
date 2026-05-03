"""
Evaluate - Model evaluation script with metrics by text length.

CLI: python -m src.evaluate
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from tqdm import tqdm

# Apply SSL fixes before importing transformers
from . import ssl_setup  # noqa: F401

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    MODEL_KEY_TO_DIR, RESULTS_DIR, DATA_CSV,
    LENGTH_SEGMENT_SHORT, LENGTH_SEGMENT_LONG, SHORT_TEXT_WORD_THRESHOLD,
    SPLIT_JSON_PATH
)
from .data_cleaning import load_prepared_data
from .data_preprocessing import add_model_input_column
from .dataset_utils import create_dataloader
from .splits import load_split, apply_saved_indices
from .length_metrics import short_mask, long_mask


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained fake news classification models"
    )
    
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=["bert", "roberta", "distilbert"],
        default=None,
        help="Models to evaluate (default: all trained models)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=f"Path to dataset CSV (default: from split metadata)"
    )
    
    return parser.parse_args()


def get_trained_models() -> List[str]:
    """Get list of model keys that have been trained (have saved weights)."""
    trained = []
    for key, path in MODEL_KEY_TO_DIR.items():
        if (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists():
            trained.append(key)
    return trained


def evaluate_model(
    model_key: str,
    test_texts: List[str],
    test_labels: List[int],
    batch_size: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a single model and return predictions.
    
    Args:
        model_key: Model key (bert, roberta, distilbert)
        test_texts: Test texts
        test_labels: Test labels
        batch_size: Batch size
        device: Device for inference
        
    Returns:
        Tuple of (predictions array, labels array)
    """
    model_dir = MODEL_KEY_TO_DIR[model_key]
    
    print(f"\nLoading {model_key.upper()} from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Create dataloader
    test_loader = create_dataloader(
        test_texts, test_labels, tokenizer,
        batch_size=batch_size, shuffle=False
    )
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_key}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def save_results(
    metrics_by_model: Dict[str, Dict[str, float]],
    metrics_by_length: Dict[str, Dict[str, Dict[str, Any]]],
    confusion_matrices: Dict[str, List[List[int]]],
    confusion_by_length: Dict[str, Dict[str, List[List[int]]]],
    length_context: Dict[str, Any]
) -> None:
    """
    Save all evaluation results to the results directory.
    
    Args:
        metrics_by_model: Overall metrics per model
        metrics_by_length: Metrics per model and length segment
        confusion_matrices: Confusion matrices per model
        confusion_by_length: Confusion matrices per model and length segment
        length_context: Length evaluation context information
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save overall metrics
    metrics_df = pd.DataFrame(metrics_by_model).T
    metrics_df.index.name = 'model'
    metrics_path = RESULTS_DIR / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path)
    print(f"\nSaved metrics to {metrics_path}")
    print(metrics_df.to_string())
    
    # Save metrics by length
    rows = []
    for model, segments in metrics_by_length.items():
        for segment, data in segments.items():
            row = {'model': model, 'segment': segment, **data}
            rows.append(row)
    
    if rows:
        length_df = pd.DataFrame(rows)
        length_path = RESULTS_DIR / "evaluation_metrics_by_length.csv"
        length_df.to_csv(length_path, index=False)
        print(f"\nSaved length metrics to {length_path}")
        print(length_df.to_string(index=False))
    
    # Save confusion matrices
    cm_path = RESULTS_DIR / "confusion_matrices.json"
    with open(cm_path, 'w') as f:
        json.dump(confusion_matrices, f, indent=2)
    print(f"\nSaved confusion matrices to {cm_path}")
    
    # Save confusion matrices by length
    cm_length_path = RESULTS_DIR / "confusion_matrices_by_length.json"
    with open(cm_length_path, 'w') as f:
        json.dump(confusion_by_length, f, indent=2)
    print(f"Saved confusion matrices by length to {cm_length_path}")
    
    # Save length context
    context_json_path = RESULTS_DIR / "evaluation_length_context.json"
    with open(context_json_path, 'w') as f:
        json.dump(length_context, f, indent=2)
    
    context_txt_path = RESULTS_DIR / "evaluation_length_context.txt"
    with open(context_txt_path, 'w') as f:
        f.write("Evaluation Length Context\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Short text threshold: <{length_context['threshold']} words\n")
        f.write(f"Long text threshold: >={length_context['threshold']} words\n\n")
        f.write(f"Short segment name: {length_context['short_segment']}\n")
        f.write(f"Long segment name: {length_context['long_segment']}\n\n")
        f.write(f"Total test samples: {length_context['total_samples']}\n")
        f.write(f"Short samples: {length_context['short_count']}\n")
        f.write(f"Long samples: {length_context['long_count']}\n")
    
    print(f"Saved length context to {context_json_path} and {context_txt_path}")


def main():
    """Main evaluation entry point."""
    args = parse_args()
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load split metadata
    split_data = load_split()
    if split_data is None:
        raise RuntimeError(
            f"No split file found at {SPLIT_JSON_PATH}. "
            "Run training first: python -m src.train"
        )
    
    metadata = split_data.get('metadata', {})
    csv_path = metadata.get('csv_path', str(DATA_CSV))
    lowercase = metadata.get('lowercase', False)
    dedupe = metadata.get('dedupe', False)
    max_samples = metadata.get('max_samples')
    
    if args.data:
        csv_path = args.data
    
    print("\n" + "="*60)
    print("Loading data using saved split metadata...")
    print("="*60)
    print(f"CSV: {csv_path}")
    print(f"Lowercase: {lowercase}, Dedupe: {dedupe}, Max samples: {max_samples}")
    
    # Load and prepare data
    df, _ = load_prepared_data(
        data_path=csv_path,
        lowercase=lowercase,
        dedupe=dedupe,
        max_samples=max_samples
    )
    
    # Add model input column
    df = add_model_input_column(df)
    
    # Apply saved split — use ONLY the saved test indices (fixes full-dataset evaluation bug)
    test_indices = split_data.get('test_indices', [])
    if not test_indices:
        raise RuntimeError(
            "No test_indices found in split file. "
            "Re-run training to regenerate the split: python -m src.train"
        )
    valid_test_indices = [i for i in test_indices if i < len(df)]
    test_df = df.iloc[valid_test_indices].reset_index(drop=True)
    
    print(f"\nTest set size: {len(test_df)} (out of {len(df)} total rows)")
    
    # Extract texts and labels
    test_texts = test_df['input_text'].tolist()
    test_labels = test_df['label'].tolist()
    
    # Determine which models to evaluate
    if args.models:
        models_to_eval = args.models
    else:
        models_to_eval = get_trained_models()
    
    if not models_to_eval:
        raise RuntimeError(
            "No trained models found. Run training first: python -m src.train"
        )
    
    print(f"\nEvaluating models: {models_to_eval}")
    
    # Evaluate each model
    metrics_by_model = {}
    metrics_by_length = {}
    confusion_matrices = {}
    confusion_by_length = {}
    
    # Create length masks
    short = short_mask(test_texts)
    long = long_mask(test_texts)
    
    for model_key in models_to_eval:
        try:
            preds, labels = evaluate_model(
                model_key, test_texts, test_labels,
                args.batch_size, device
            )
            
            # Overall metrics
            metrics = compute_metrics(labels, preds)
            metrics_by_model[model_key] = metrics
            
            # Confusion matrix
            cm = confusion_matrix(labels, preds).tolist()
            confusion_matrices[model_key] = cm
            
            # Metrics by length
            metrics_by_length[model_key] = {}
            confusion_by_length[model_key] = {}
            
            # Short texts
            if short.any():
                short_labels = labels[short]
                short_preds = preds[short]
                short_metrics = compute_metrics(short_labels, short_preds)
                short_metrics['n_samples'] = int(short.sum())
                metrics_by_length[model_key][LENGTH_SEGMENT_SHORT] = short_metrics
                
                short_cm = confusion_matrix(short_labels, short_preds).tolist()
                confusion_by_length[model_key][LENGTH_SEGMENT_SHORT] = short_cm
            
            # Long texts
            if long.any():
                long_labels = labels[long]
                long_preds = preds[long]
                long_metrics = compute_metrics(long_labels, long_preds)
                long_metrics['n_samples'] = int(long.sum())
                metrics_by_length[model_key][LENGTH_SEGMENT_LONG] = long_metrics
                
                long_cm = confusion_matrix(long_labels, long_preds).tolist()
                confusion_by_length[model_key][LENGTH_SEGMENT_LONG] = long_cm
            
            print(f"\n{model_key.upper()} Results:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"\nError evaluating {model_key}: {e}")
            continue
    
    # Length context
    length_context = {
        'threshold': SHORT_TEXT_WORD_THRESHOLD,
        'short_segment': LENGTH_SEGMENT_SHORT,
        'long_segment': LENGTH_SEGMENT_LONG,
        'total_samples': len(test_texts),
        'short_count': int(short.sum()),
        'long_count': int(long.sum())
    }
    
    # Save all results
    if metrics_by_model:
        save_results(
            metrics_by_model,
            metrics_by_length,
            confusion_matrices,
            confusion_by_length,
            length_context
        )
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)
    print("\nNext steps:")
    print("  python -m src.visualization  # Generate plots")
    print("  python -m app.app            # Run Flask web app")


if __name__ == "__main__":
    main()