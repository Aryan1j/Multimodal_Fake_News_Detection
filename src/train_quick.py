"""
Quick Train - Faster training script for demos and smoke testing.

CLI: python -m src.train_quick --models bert --epochs 1
"""

import sys
from .config import QUICK_TRAIN_MAX_SAMPLES, QUICK_TRAIN_DEFAULT_EPOCHS


def main():
    """
    Quick training entry point.
    
    Wraps the main train module with defaults for quick testing:
    - Limited samples (default 3000)
    - Single epoch by default
    """
    # Import here to avoid circular imports and ensure ssl_setup runs
    from . import train
    
    # Inject quick-train defaults into sys.argv if not specified
    args = sys.argv[1:]
    
    # Add default max-samples if not specified
    if "--max-samples" not in args:
        args.extend(["--max-samples", str(QUICK_TRAIN_MAX_SAMPLES)])
    
    # Add default epochs if not specified
    if "--epochs" not in args and "-e" not in args:
        args.extend(["--epochs", str(QUICK_TRAIN_DEFAULT_EPOCHS)])
    
    # Replace sys.argv for the train module
    sys.argv = [sys.argv[0]] + args
    
    print("="*60)
    print("QUICK TRAINING MODE")
    print(f"Default max samples: {QUICK_TRAIN_MAX_SAMPLES}")
    print(f"Default epochs: {QUICK_TRAIN_DEFAULT_EPOCHS}")
    print("="*60)
    
    # Run the main training
    train.main()


if __name__ == "__main__":
    main()
