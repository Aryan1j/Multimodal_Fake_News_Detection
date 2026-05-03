"""
Configuration - Single source of truth for all paths, constants, and model settings.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Project root is the parent of the src/ directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
DATA_CSV = DATA_DIR / "WELFake_Dataset.csv"

# Models directory and per-model subdirectories
MODELS_DIR = PROJECT_ROOT / "models"
BERT_MODEL_DIR = MODELS_DIR / "bert_model"
ROBERTA_MODEL_DIR = MODELS_DIR / "roberta_model"
DISTILBERT_MODEL_DIR = MODELS_DIR / "distilbert_model"

# Results and graphs directories
RESULTS_DIR = PROJECT_ROOT / "results"
GRAPHS_DIR = PROJECT_ROOT / "graphs"

# Split file path
SPLIT_JSON_PATH = RESULTS_DIR / "data_split.json"

# =============================================================================
# HUGGINGFACE MODEL IDS
# =============================================================================

MODEL_IDS = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
}

# Mapping from model key to save directory
MODEL_KEY_TO_DIR = {
    "bert": BERT_MODEL_DIR,
    "roberta": ROBERTA_MODEL_DIR,
    "distilbert": DISTILBERT_MODEL_DIR,
}

# =============================================================================
# TOKENIZER SETTINGS
# =============================================================================

MAX_LENGTH = 256 # Maximum sequence length for tokenization

# =============================================================================
# RANDOM SEED
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# TRAIN/VAL/TEST SPLIT FRACTIONS
# =============================================================================

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10
TEST_FRACTION = 0.20

# =============================================================================
# QUICK TRAINING DEFAULTS
# =============================================================================

QUICK_TRAIN_MAX_SAMPLES = 3000
QUICK_TRAIN_DEFAULT_EPOCHS = 1

# =============================================================================
# LABELS
# =============================================================================

LABEL_REAL = 0
LABEL_FAKE = 1

LABEL_NAMES = {
    LABEL_REAL: "Real",
    LABEL_FAKE: "Fake",
}

# =============================================================================
# INFERENCE SETTINGS
# =============================================================================

# Confidence threshold below which predictions are marked as uncertain
CONFIDENCE_THRESHOLD = 0.6

# =============================================================================
# LENGTH-BASED EVALUATION
# =============================================================================

# Word count threshold for short vs long text classification
SHORT_TEXT_WORD_THRESHOLD = 100

# Segment names for evaluation reports
LENGTH_SEGMENT_SHORT = f"short_lt{SHORT_TEXT_WORD_THRESHOLD}_words"
LENGTH_SEGMENT_LONG = f"long_ge{SHORT_TEXT_WORD_THRESHOLD}_words"

# =============================================================================
# ENSURE DIRECTORIES EXIST
# =============================================================================

def ensure_directories():
    """Create all required directories if they don't exist."""
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, GRAPHS_DIR,
                      BERT_MODEL_DIR, ROBERTA_MODEL_DIR, DISTILBERT_MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HYBRID LAYER PATHS
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

def get_model_artifact_dir(model_key: str) -> Path:
    return ARTIFACTS_DIR / model_key

EVIDENCE_INDEX_PATH = ARTIFACTS_DIR / "evidence_index.joblib"
EVIDENCE_CORPUS_PATH = ARTIFACTS_DIR / "evidence_corpus.csv"
# Create directories on import
def ensure_directories():
    for directory in [
        DATA_DIR, MODELS_DIR, RESULTS_DIR, GRAPHS_DIR,
        BERT_MODEL_DIR, ROBERTA_MODEL_DIR, DISTILBERT_MODEL_DIR,
        ARTIFACTS_DIR,
        ARTIFACTS_DIR / "bert",
        ARTIFACTS_DIR / "roberta",
        ARTIFACTS_DIR / "distilbert",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
