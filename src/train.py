"""
Train - Main training script for fake news classification models.

CLI: python -m src.train --models bert roberta distilbert --epochs 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

# Apply SSL fixes before importing transformers
from . import ssl_setup  # noqa: F401

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    MODEL_IDS, MODEL_KEY_TO_DIR, MAX_LENGTH, RANDOM_SEED,
    DATA_CSV
)
from .data_cleaning import load_prepared_data
from .data_preprocessing import add_model_input_column
from .dataset_utils import create_dataloader
from .splits import ensure_data_split, remove_stale_split_before_training


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train fake news classification models"
    )
    
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=["bert", "roberta", "distilbert"],
        default=["bert"],
        help="Models to train (default: bert)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (default: 0.0)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=f"Path to dataset CSV (default: {DATA_CSV})"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use (default: all)"
    )
    parser.add_argument(
        "--lowercase-clean",
        action="store_true",
        help="Convert text to lowercase during cleaning"
    )
    parser.add_argument(
        "--dedupe-data",
        action="store_true",
        help="Remove duplicate articles"
    )
    parser.add_argument(
        "--force-new-split",
        action="store_true",
        help="Force creation of new train/val/test split"
    )
    
    return parser.parse_args()


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    label_smoothing: float = 0.0
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: HuggingFace model
        dataloader: Training DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        label_smoothing: Label smoothing factor
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Compute loss
        loss = loss_fn(outputs.logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(
    model,
    dataloader,
    device: torch.device
) -> float:
    """
    Validate model on validation set.
    
    Args:
        model: HuggingFace model
        dataloader: Validation DataLoader
        device: Device for inference
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def train_one_model(
    model_key: str,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Train a single model with validation-based checkpointing.
    
    Args:
        model_key: Model key (bert, roberta, distilbert)
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        label_smoothing: Label smoothing factor
        device: Training device
        
    Returns:
        Training history dictionary with train_loss and val_loss lists
    """
    model_id = MODEL_IDS[model_key]
    save_dir = MODEL_KEY_TO_DIR[model_key]
    
    print(f"\n{'='*60}")
    print(f"Training {model_key.upper()} ({model_id})")
    print(f"{'='*60}")
    
    # Load tokenizer and model
    print(f"Loading tokenizer and model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2
    )
    model.to(device)
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_texts, train_labels, tokenizer,
        batch_size=batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        val_texts, val_labels, tokenizer,
        batch_size=batch_size, shuffle=False
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop with best model tracking
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, label_smoothing
        )
        history['train_loss'].append(train_loss)
        
        # Validation
        val_loss = validate(model, val_loader, device)
        history['val_loss'].append(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best val loss! Saving to {save_dir}")
            
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
    
    # Save training history
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")
    
    return history


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
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
    
    # Load and prepare data
    print("\n" + "="*60)
    print("Loading and preparing data...")
    print("="*60)
    
    df, csv_path = load_prepared_data(
        data_path=args.data,
        lowercase=args.lowercase_clean,
        dedupe=args.dedupe_data,
        max_samples=args.max_samples
    )
    
    # Remove stale split if configuration changed
    remove_stale_split_before_training(
        csv_path=csv_path,
        n_rows=len(df),
        lowercase=args.lowercase_clean,
        dedupe=args.dedupe_data,
        max_samples=args.max_samples
    )
    
    # Add model input column
    df = add_model_input_column(df)
    
    # Get train/val/test splits
    train_df, val_df, test_df = ensure_data_split(
        df=df,
        csv_path=csv_path,
        lowercase=args.lowercase_clean,
        dedupe=args.dedupe_data,
        max_samples=args.max_samples,
        force_new=args.force_new_split
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")
    
    # Extract texts and labels
    train_texts = train_df['input_text'].tolist()
    train_labels = train_df['label'].tolist()
    val_texts = val_df['input_text'].tolist()
    val_labels = val_df['label'].tolist()
    
    # Train each model
    all_histories = {}
    
    for model_key in args.models:
        history = train_one_model(
            model_key=model_key,
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            device=device
        )
        all_histories[model_key] = history
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nTrained models saved to:")
    for model_key in args.models:
        print(f"  {model_key}: {MODEL_KEY_TO_DIR[model_key]}")
    
    print("\nNext steps:")
    print("  python -m src.evaluate    # Evaluate models on test set")
    print("  python -m src.visualization  # Generate plots")
    print("  python -m app.app         # Run Flask web app")


if __name__ == "__main__":
    main()
