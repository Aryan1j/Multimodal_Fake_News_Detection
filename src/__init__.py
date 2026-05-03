"""
Fake News Classification System - Core Package

Main modules:
- config: Configuration constants and paths
- ssl_setup: SSL certificate handling for HuggingFace downloads
- data_cleaning: CSV loading and data cleaning
- data_preprocessing: Text preprocessing for model input
- dataset_utils: PyTorch Dataset and DataLoader utilities
- splits: Stratified train/val/test splitting
- length_metrics: Text length analysis utilities
- train: Model training script
- train_quick: Quick training for demos
- evaluate: Model evaluation script
- inference: Single prediction utilities
- api_check: Optional news API evidence checking
- visualization: Results visualization
- compare_models: Model comparison on short vs long form texts
"""

__version__ = "1.0.0"
