"""
Inference - Single prediction utilities for the Flask app and CLI.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

# Apply SSL fixes before importing transformers
from . import ssl_setup  # noqa: F401

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import (
    MODEL_KEY_TO_DIR, MAX_LENGTH, CONFIDENCE_THRESHOLD, LABEL_NAMES
)
from .data_preprocessing import preprocess_user_input


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    label: str  # "Real" or "Fake"
    label_id: int  # 0 or 1
    confidence: float  # Probability of predicted class
    model_key: str  # Model used
    model_display_name: str  # Human-readable model name
    note: Optional[str] = None  # Uncertainty message if confidence is low
    evidence_check: Optional[str] = None  # News API evidence check result


# Cache for loaded models and tokenizers
_model_cache: Dict[str, Tuple] = {}


def get_model_display_name(model_key: str) -> str:
    """Get human-readable display name for a model."""
    names = {
        'bert': 'BERT',
        'roberta': 'RoBERTa',
        'distilbert': 'DistilBERT'
    }
    return names.get(model_key, model_key.upper())


def load_model(model_key: str) -> Tuple:
    """
    Load model and tokenizer, using cache if available.
    
    Args:
        model_key: Model key (bert, roberta, distilbert)
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    if model_key in _model_cache:
        return _model_cache[model_key]
    
    model_dir = MODEL_KEY_TO_DIR[model_key]
    
    if not model_dir.exists():
        raise RuntimeError(
            f"Model not found at {model_dir}. "
            f"Run training first: python -m src.train --models {model_key}"
        )
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Loading {model_key} from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    _model_cache[model_key] = (tokenizer, model, device)
    return tokenizer, model, device


def predict(
    raw_text: str,
    model_key: str = "bert",
    check_evidence: bool = False
) -> PredictionResult:
    """
    Make a prediction on raw user input text.
    
    Args:
        raw_text: User-provided text (may not have separate title)
        model_key: Model to use (bert, roberta, distilbert)
        check_evidence: Whether to call news API for evidence checking
        
    Returns:
        PredictionResult with label, confidence, and optional evidence
    """
    # Load model (uses cache)
    tokenizer, model, device = load_model(model_key)
    
    # Preprocess text
    input_text = preprocess_user_input(raw_text)
    
    if not input_text:
        return PredictionResult(
            label="Unknown",
            label_id=-1,
            confidence=0.0,
            model_key=model_key,
            model_display_name=get_model_display_name(model_key),
            note="Empty input text"
        )
    
    # Tokenize
    encoding = tokenizer(
        input_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Get probabilities and prediction
    probs = torch.softmax(outputs.logits, dim=1)[0]
    pred_label_id = torch.argmax(probs).item()
    confidence = probs[pred_label_id].item()
    
    # Get label name
    label = LABEL_NAMES.get(pred_label_id, "Unknown")
    
    # Check confidence threshold
    note = None
    if confidence < CONFIDENCE_THRESHOLD:
        note = f"Low confidence ({confidence:.1%}). Prediction may be uncertain."
    
    # Optional evidence check
    evidence_check = None
    if check_evidence:
        try:
            from .api_check import check_news
            evidence_check = check_news(raw_text)
        except Exception as e:
            evidence_check = f"Evidence check failed: {str(e)}"
    
    return PredictionResult(
        label=label,
        label_id=pred_label_id,
        confidence=confidence,
        model_key=model_key,
        model_display_name=get_model_display_name(model_key),
        note=note,
        evidence_check=evidence_check
    )


def predict_batch(
    texts: list,
    model_key: str = "bert"
) -> list:
    """
    Make predictions on a batch of texts.
    
    Args:
        texts: List of raw text strings
        model_key: Model to use
        
    Returns:
        List of PredictionResult objects
    """
    return [predict(text, model_key) for text in texts]


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference 'Your news text here' [model_key]")
        sys.exit(1)
    
    text = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "bert"
    
    result = predict(text, model, check_evidence=True)
    
    print(f"\nPrediction: {result.label}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Model: {result.model_display_name}")
    
    if result.note:
        print(f"Note: {result.note}")
    
    if result.evidence_check:
        print(f"Evidence: {result.evidence_check}")
