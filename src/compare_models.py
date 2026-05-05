
# Compare Models - Detailed comparison of models on short vs long form texts.


import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from .config import (
    RESULTS_DIR, GRAPHS_DIR, LABEL_NAMES,
    SHORT_TEXT_WORD_THRESHOLD, LENGTH_SEGMENT_SHORT, LENGTH_SEGMENT_LONG
)


def load_comparison_data() -> Dict[str, Any]:
    """Load all data needed for model comparison."""
    data = {}
    
    # Overall metrics
    metrics_path = RESULTS_DIR / "evaluation_metrics.csv"
    if metrics_path.exists():
        data['overall'] = pd.read_csv(metrics_path, index_col='model')
    
    # Length-based metrics
    length_path = RESULTS_DIR / "evaluation_metrics_by_length.csv"
    if length_path.exists():
        data['by_length'] = pd.read_csv(length_path)
    
    # Confusion matrices by length
    cm_path = RESULTS_DIR / "confusion_matrices_by_length.json"
    if cm_path.exists():
        with open(cm_path) as f:
            data['confusion_by_length'] = json.load(f)
    
    # Length context
    context_path = RESULTS_DIR / "evaluation_length_context.json"
    if context_path.exists():
        with open(context_path) as f:
            data['context'] = json.load(f)
    
    return data


def print_summary_table(data: Dict[str, Any]) -> None:
    """Print a summary table comparing models on short vs long texts."""
    if 'by_length' not in data:
        print("No length-based metrics found.")
        return
    
    df = data['by_length']
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: SHORT vs LONG FORM FAKE NEWS")
    print("=" * 80)
    
    if 'context' in data:
        ctx = data['context']
        print(f"\nThreshold: {ctx['threshold']} words")
        print(f"Short texts (<{ctx['threshold']} words): {ctx['short_count']} samples")
        print(f"Long texts (>={ctx['threshold']} words): {ctx['long_count']} samples")
    
    # Create comparison table
    print("\n" + "-" * 80)
    print("ACCURACY COMPARISON")
    print("-" * 80)
    
    models = df['model'].unique()
    segments = df['segment'].unique()
    
    # Pivot for comparison
    accuracy_pivot = df.pivot(index='model', columns='segment', values='accuracy')
    accuracy_pivot['difference'] = accuracy_pivot.iloc[:, 1] - accuracy_pivot.iloc[:, 0]
    accuracy_pivot.columns = [c.replace('_', ' ').title() for c in accuracy_pivot.columns]
    
    print(tabulate(
        accuracy_pivot.round(4),
        headers='keys',
        tablefmt='pretty',
        showindex=True
    ))
    
    # F1 comparison
    print("\n" + "-" * 80)
    print("F1 SCORE COMPARISON")
    print("-" * 80)
    
    f1_pivot = df.pivot(index='model', columns='segment', values='f1')
    f1_pivot['difference'] = f1_pivot.iloc[:, 1] - f1_pivot.iloc[:, 0]
    f1_pivot.columns = [c.replace('_', ' ').title() for c in f1_pivot.columns]
    
    print(tabulate(
        f1_pivot.round(4),
        headers='keys',
        tablefmt='pretty',
        showindex=True
    ))
    
    # Best model per segment
    print("\n" + "-" * 80)
    print("BEST MODEL PER SEGMENT")
    print("-" * 80)
    
    for segment in segments:
        segment_df = df[df['segment'] == segment]
        best_acc_model = segment_df.loc[segment_df['accuracy'].idxmax(), 'model']
        best_f1_model = segment_df.loc[segment_df['f1'].idxmax(), 'model']
        best_acc = segment_df['accuracy'].max()
        best_f1 = segment_df['f1'].max()
        
        segment_name = segment.replace('_', ' ').title()
        print(f"\n{segment_name}:")
        print(f"  Best Accuracy: {best_acc_model.upper()} ({best_acc:.4f})")
        print(f"  Best F1 Score: {best_f1_model.upper()} ({best_f1:.4f})")


def plot_short_vs_long_comparison(data: Dict[str, Any]) -> None:
    """Create detailed comparison plots for short vs long texts."""
    if 'by_length' not in data:
        print("No length-based metrics found for plotting.")
        return
    
    df = data['by_length']
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Side-by-side bar chart for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = {'short': '#3498db', 'long': '#e74c3c'}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Get data for each segment
        models = df['model'].unique()
        x = np.arange(len(models))
        width = 0.35
        
        for i, segment in enumerate(df['segment'].unique()):
            segment_data = df[df['segment'] == segment]
            values = [segment_data[segment_data['model'] == m][metric].values[0] 
                     for m in models]
            
            label = 'Short' if 'short' in segment else 'Long'
            color = colors['short'] if 'short' in segment else colors['long']
            
            bars = ax.bar(x + (i - 0.5) * width, values, width, 
                         label=f'{label} Form', color=color, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel(metric.title(), fontsize=11)
        ax.set_title(f'{metric.title()} by Text Length', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models], fontsize=10)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Model Comparison: Short vs Long Form Fake News Detection', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "comparison_short_vs_long.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {GRAPHS_DIR / 'comparison_short_vs_long.png'}")
    
    # 2. Performance gap analysis (heatmap)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model'].unique()
    gap_data = []
    
    for model in models:
        model_df = df[df['model'] == model]
        short_row = model_df[model_df['segment'].str.contains('short')].iloc[0]
        long_row = model_df[model_df['segment'].str.contains('long')].iloc[0]
        
        gaps = {
            'Model': model.upper(),
            'Accuracy Gap': long_row['accuracy'] - short_row['accuracy'],
            'Precision Gap': long_row['precision'] - short_row['precision'],
            'Recall Gap': long_row['recall'] - short_row['recall'],
            'F1 Gap': long_row['f1'] - short_row['f1']
        }
        gap_data.append(gaps)
    
    gap_df = pd.DataFrame(gap_data).set_index('Model')
    
    # Create heatmap
    sns.heatmap(gap_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Performance Gap (Long - Short)'})
    
    ax.set_title('Performance Gap: Long Form vs Short Form\n(Positive = Better on Long Texts)',
                fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "comparison_performance_gap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved gap analysis to {GRAPHS_DIR / 'comparison_performance_gap.png'}")
    
    # 3. Radar chart for each model
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = df[df['model'] == model]
        
        for segment in model_df['segment'].unique():
            row = model_df[model_df['segment'] == segment].iloc[0]
            values = [row['accuracy'], row['precision'], row['recall'], row['f1']]
            values += values[:1]
            
            label = 'Short' if 'short' in segment else 'Long'
            color = colors['short'] if 'short' in segment else colors['long']
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'{label} Form', color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model.upper()}', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.suptitle('Model Performance Profiles: Short vs Long Form', 
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "comparison_radar_charts.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved radar charts to {GRAPHS_DIR / 'comparison_radar_charts.png'}")
    
    # 4. Sample count bar chart
    if 'n_samples' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        sample_df = df.groupby('segment')['n_samples'].first().reset_index()
        sample_df['label'] = sample_df['segment'].apply(
            lambda x: 'Short Form' if 'short' in x else 'Long Form'
        )
        
        bars = ax.bar(sample_df['label'], sample_df['n_samples'], 
                     color=[colors['short'], colors['long']], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title('Test Set Distribution by Text Length', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / "comparison_sample_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved sample distribution to {GRAPHS_DIR / 'comparison_sample_distribution.png'}")


def generate_comparison_report(data: Dict[str, Any]) -> None:
    """Generate a detailed text report for the comparison."""
    if 'by_length' not in data:
        return
    
    df = data['by_length']
    report_path = RESULTS_DIR / "model_comparison_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FAKE NEWS CLASSIFICATION MODEL COMPARISON REPORT\n")
        f.write("Short Form vs Long Form Text Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        if 'context' in data:
            ctx = data['context']
            f.write("DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Short text threshold: <{ctx['threshold']} words\n")
            f.write(f"Long text threshold: >={ctx['threshold']} words\n")
            f.write(f"Total test samples: {ctx['total_samples']}\n")
            f.write(f"Short form samples: {ctx['short_count']}\n")
            f.write(f"Long form samples: {ctx['long_count']}\n\n")
        
        models = df['model'].unique()
        
        for model in models:
            f.write(f"\n{'='*40}\n")
            f.write(f"{model.upper()} MODEL\n")
            f.write(f"{'='*40}\n\n")
            
            model_df = df[df['model'] == model]
            
            for _, row in model_df.iterrows():
                segment = 'SHORT FORM' if 'short' in row['segment'] else 'LONG FORM'
                f.write(f"{segment}:\n")
                f.write(f"  Accuracy:  {row['accuracy']:.4f}\n")
                f.write(f"  Precision: {row['precision']:.4f}\n")
                f.write(f"  Recall:    {row['recall']:.4f}\n")
                f.write(f"  F1 Score:  {row['f1']:.4f}\n")
                if 'n_samples' in row:
                    f.write(f"  Samples:   {int(row['n_samples'])}\n")
                f.write("\n")
        
        # Summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY & RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        # Find best models
        short_df = df[df['segment'].str.contains('short')]
        long_df = df[df['segment'].str.contains('long')]
        
        best_short_acc = short_df.loc[short_df['accuracy'].idxmax()]
        best_long_acc = long_df.loc[long_df['accuracy'].idxmax()]
        best_short_f1 = short_df.loc[short_df['f1'].idxmax()]
        best_long_f1 = long_df.loc[long_df['f1'].idxmax()]
        
        f.write("Best Model for SHORT FORM texts:\n")
        f.write(f"  By Accuracy: {best_short_acc['model'].upper()} ({best_short_acc['accuracy']:.4f})\n")
        f.write(f"  By F1 Score: {best_short_f1['model'].upper()} ({best_short_f1['f1']:.4f})\n\n")
        
        f.write("Best Model for LONG FORM texts:\n")
        f.write(f"  By Accuracy: {best_long_acc['model'].upper()} ({best_long_acc['accuracy']:.4f})\n")
        f.write(f"  By F1 Score: {best_long_f1['model'].upper()} ({best_long_f1['f1']:.4f})\n\n")
        
        # Overall best
        if 'overall' in data:
            overall = data['overall']
            best_overall = overall['accuracy'].idxmax()
            f.write(f"Best OVERALL Model: {best_overall.upper()} (Accuracy: {overall.loc[best_overall, 'accuracy']:.4f})\n")
    
    print(f"\nSaved comparison report to {report_path}")


def main():
    """Main entry point for model comparison."""
    print("\n" + "=" * 80)
    print("FAKE NEWS CLASSIFICATION - MODEL COMPARISON")
    print("Comparing BERT, RoBERTa, and DistilBERT on Short vs Long Form Texts")
    print("=" * 80)
    
    # Load data
    data = load_comparison_data()
    
    if not data:
        print("\nNo evaluation results found!")
        print("Please run evaluation first:")
        print("  python -m src.evaluate")
        return
    
    # Print summary table
    print_summary_table(data)
    
    # Generate plots
    print("\n" + "-" * 80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("-" * 80)
    plot_short_vs_long_comparison(data)
    
    # Generate report
    generate_comparison_report(data)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {GRAPHS_DIR / 'comparison_short_vs_long.png'}")
    print(f"  - {GRAPHS_DIR / 'comparison_performance_gap.png'}")
    print(f"  - {GRAPHS_DIR / 'comparison_radar_charts.png'}")
    print(f"  - {GRAPHS_DIR / 'comparison_sample_distribution.png'}")
    print(f"  - {RESULTS_DIR / 'model_comparison_report.txt'}")


if __name__ == "__main__":
    main()
