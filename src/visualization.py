"""
Visualization - Generate plots and charts from evaluation results.

CLI: python -m src.visualization
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import RESULTS_DIR, GRAPHS_DIR, LABEL_NAMES


def load_results():
    """Load all evaluation results from the results directory."""
    results = {}
    
    # Load overall metrics
    metrics_path = RESULTS_DIR / "evaluation_metrics.csv"
    if metrics_path.exists():
        results['metrics'] = pd.read_csv(metrics_path, index_col='model')
    
    # Load metrics by length
    length_path = RESULTS_DIR / "evaluation_metrics_by_length.csv"
    if length_path.exists():
        results['metrics_by_length'] = pd.read_csv(length_path)
    
    # Load confusion matrices
    cm_path = RESULTS_DIR / "confusion_matrices.json"
    if cm_path.exists():
        with open(cm_path) as f:
            results['confusion_matrices'] = json.load(f)
    
    # Load confusion matrices by length
    cm_length_path = RESULTS_DIR / "confusion_matrices_by_length.json"
    if cm_length_path.exists():
        with open(cm_length_path) as f:
            results['confusion_by_length'] = json.load(f)
    
    # Load length context
    context_path = RESULTS_DIR / "evaluation_length_context.json"
    if context_path.exists():
        with open(context_path) as f:
            results['length_context'] = json.load(f)
    
    return results


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_path: Path):
    """
    Create a bar chart comparing metrics across models.
    
    Args:
        metrics_df: DataFrame with models as index, metrics as columns
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    available_metrics = [m for m in metrics if m in metrics_df.columns]
    
    x = np.arange(len(metrics_df.index))
    width = 0.2
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, metrics_df[metric], width, label=metric.title())
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_df.index])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison to {output_path}")


def plot_confusion_matrix(cm: list, model_name: str, output_path: Path):
    """
    Create a heatmap for a confusion matrix.
    
    Args:
        cm: Confusion matrix as nested list
        model_name: Name of the model
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm_array = np.array(cm)
    
    # Create heatmap
    sns.heatmap(
        cm_array,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[LABEL_NAMES[0], LABEL_NAMES[1]],
        yticklabels=[LABEL_NAMES[0], LABEL_NAMES[1]],
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name.upper()}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_metrics_by_length(length_df: pd.DataFrame, output_path: Path):
    """
    Create a grouped bar chart comparing metrics by text length.
    
    Args:
        length_df: DataFrame with model, segment, and metrics columns
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        if metric not in length_df.columns:
            continue
            
        ax = axes[idx]
        
        # Pivot for grouped bars
        pivot = length_df.pivot(index='model', columns='segment', values=metric)
        
        pivot.plot(kind='bar', ax=ax, width=0.7)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title(f'{metric.title()} by Text Length')
        ax.set_xticklabels([m.upper() for m in pivot.index], rotation=0)
        ax.legend(title='Segment', fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics by length to {output_path}")


def plot_training_history(model_key: str, history: dict, output_path: Path):
    """
    Plot training and validation loss over epochs.
    
    Args:
        model_key: Model name
        history: Dictionary with train_loss and val_loss lists
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    ax.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training History - {model_key.upper()}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history to {output_path}")


def main():
    """Generate all visualization plots."""
    print("="*60)
    print("Generating visualizations...")
    print("="*60)
    
    # Ensure output directory exists
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results()
    
    if not results:
        print("No evaluation results found. Run evaluation first:")
        print("  python -m src.evaluate")
        return
    
    # Plot metrics comparison
    if 'metrics' in results:
        plot_metrics_comparison(
            results['metrics'],
            GRAPHS_DIR / "metrics_comparison.png"
        )
    
    # Plot confusion matrices
    if 'confusion_matrices' in results:
        for model, cm in results['confusion_matrices'].items():
            plot_confusion_matrix(
                cm, model,
                GRAPHS_DIR / f"confusion_matrix_{model}.png"
            )
    
    # Plot metrics by length
    if 'metrics_by_length' in results:
        plot_metrics_by_length(
            results['metrics_by_length'],
            GRAPHS_DIR / "metrics_by_length.png"
        )
    
    # Plot training histories
    from .config import MODEL_KEY_TO_DIR
    
    for model_key, model_dir in MODEL_KEY_TO_DIR.items():
        history_path = model_dir / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            plot_training_history(
                model_key, history,
                GRAPHS_DIR / f"training_history_{model_key}.png"
            )
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print(f"\nPlots saved to: {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
