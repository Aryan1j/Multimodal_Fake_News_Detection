
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import ARTIFACTS_DIR, CONFIDENCE_THRESHOLD, MAX_LENGTH
from .inference import get_model_display_name, load_model

from context.linguistic_features import build_context_score
from fact_check.retriever import LocalEvidenceRetriever
from fusion.combine import DEFAULT_WEIGHTS, build_explanation, combine_scores


@dataclass
class HybridPredictionResult:
    label: str
    confidence: float
    model_key: str
    model_display_name: str
    breakdown: dict = field(default_factory=dict)
    explanation: list = field(default_factory=list)
    evidence_snippets: list = field(default_factory=list)
    details: dict = field(default_factory=dict)
    note: Optional[str] = None


_per_model_cache: dict = {}
_retriever = None

# All available model keys for ensemble
ALL_MODEL_KEYS = ["bert", "roberta", "distilbert"]


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = LocalEvidenceRetriever()
    return _retriever


def _load_calibrator(model_key: str):
    try:
        import joblib
        path = ARTIFACTS_DIR / model_key / "calibrator.joblib"
        if path.exists():
            return joblib.load(path)
    except Exception:
        pass
    return None


def _load_fusion_weights(model_key: str) -> dict:
    path = ARTIFACTS_DIR / model_key / "fusion_weights.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return DEFAULT_WEIGHTS.copy()


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / exp_v.sum(axis=1, keepdims=True)


def _get_model_score(text: str, model_key: str, calibrator) -> float:
    import torch
    tokenizer, model, device = load_model(model_key)
    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                    padding="max_length", return_tensors="pt")
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits.cpu().numpy()
    prob_fake = float(_softmax(logits)[0, 1])
    if calibrator is not None:
        prob_fake = float(
            calibrator.predict_proba(np.array([[prob_fake]]))[0, 1]
        )
    return prob_fake


def _evidence_score(snippets: list) -> float:
    """
    Compute an evidence score from retrieved snippets.
    """
    if not snippets:
        return 0.5

    quality_snippets = [s for s in snippets if s["similarity"] > 0.35]
    if not quality_snippets:
        return 0.5

    support = 0.0
    contradiction = 0.0
    related = 0.0
    total_weight = 0.0

    for item in quality_snippets:
        w = item["similarity"]
        total_weight += w
        if item["stance"] == "support":
            support += w
        elif item["stance"] == "contradict":
            contradiction += w
        else:
            related += w * 0.3

    if total_weight == 0:
        return 0.5

    support_ratio = support / total_weight
    contradict_ratio = contradiction / total_weight

    score = 0.5 + (contradict_ratio * 0.4) - (support_ratio * 0.4)
    return max(0.0, min(1.0, score))


def _build_note(model_score: float, confidence: float, word_count: int) -> Optional[str]:
    """
    Build a user-facing note for uncertain or OOD predictions.
    """
    notes = []

    # OOD uncertainty: model score very close to decision boundary
    if abs(model_score - 0.5) < 0.15:
        notes.append("⚠️ Model is uncertain on this input — result may be unreliable for non-article text.")

    # Low overall confidence after fusion
    if confidence < CONFIDENCE_THRESHOLD:
        notes.append(f"⚠️ Low confidence ({confidence:.1%}). Result may not be reliable.")

    # Short text warning
    if word_count < 50:
        notes.append(f"⚠️ Short input ({word_count} words). Models perform best on full articles (100+ words).")

    return " ".join(notes) if notes else None


def _ensure_model_cache(model_key: str):
    if model_key not in _per_model_cache:
        _per_model_cache[model_key] = {
            "calibrator": _load_calibrator(model_key),
            "weights":    _load_fusion_weights(model_key),
        }


def predict_hybrid(text: str, model_key: str = "roberta") -> HybridPredictionResult:
    from .data_preprocessing import preprocess_user_input

    input_text = preprocess_user_input(text)
    word_count = len(input_text.split())

    if not input_text:
        return HybridPredictionResult(
            label="Unknown", confidence=0.0,
            model_key=model_key,
            model_display_name=get_model_display_name(model_key),
            note="Empty input after preprocessing.",
        )

    _ensure_model_cache(model_key)
    calibrator = _per_model_cache[model_key]["calibrator"]
    weights    = _per_model_cache[model_key]["weights"]

    model_score     = _get_model_score(input_text, model_key, calibrator)
    context_details = build_context_score(input_text)
    snippets        = _get_retriever().retrieve(input_text)
    ev_score        = _evidence_score(snippets)

    final_score = combine_scores(
        transformer_score=model_score,
        context_score=context_details["context_score"],
        evidence_score=ev_score,
        model_key=model_key,
        weights=weights,
    )
    

    label      = "FAKE" if final_score >= 0.5 else "REAL"
    confidence = final_score if label == "FAKE" else 1.0 - final_score
    display    = get_model_display_name(model_key)

    explanation = build_explanation(
        label, model_score, context_details, snippets,
        model_name=display,
    )

    note = _build_note(model_score, confidence, word_count)

    return HybridPredictionResult(
        label=label,
        confidence=round(confidence, 4),
        model_key=model_key,
        model_display_name=display,
        breakdown={
            "model_score":    round(model_score, 4),
            "context_score":  round(context_details["context_score"], 4),
            "evidence_score": round(ev_score, 4),
        },
        explanation=explanation,
        evidence_snippets=snippets,
        details={
            "sentiment":  context_details["sentiment"],
            "linguistic": context_details["linguistic"],
        },
        note=note,
    )


def predict_ensemble(text: str) -> HybridPredictionResult:
    """
    Ensemble prediction: averages model scores from all 3 models (BERT, RoBERTa, DistilBERT).

    More robust than a single model — smooths out OOD bias from any one model.
    Falls back gracefully if a model fails to load.
    """
    from .data_preprocessing import preprocess_user_input

    input_text = preprocess_user_input(text)
    word_count = len(input_text.split())

    if not input_text:
        return HybridPredictionResult(
            label="Unknown", confidence=0.0,
            model_key="ensemble",
            model_display_name="Ensemble (BERT + RoBERTa + DistilBERT)",
            note="Empty input after preprocessing.",
        )

    # Collect scores from all models
    model_scores = []
    failed_models = []
    for key in ALL_MODEL_KEYS:
        try:
            _ensure_model_cache(key)
            calibrator = _per_model_cache[key]["calibrator"]
            score = _get_model_score(input_text, key, calibrator)
            model_scores.append((key, score))
        except Exception as e:
            failed_models.append(key)

    if not model_scores:
        return HybridPredictionResult(
            label="Unknown", confidence=0.0,
            model_key="ensemble",
            model_display_name="Ensemble (BERT + RoBERTa + DistilBERT)",
            note="All models failed to load.",
        )

    # Average model scores
    avg_model_score = float(np.mean([s for _, s in model_scores]))

    # Score spread — high spread = models disagree = OOD signal
    score_spread = float(np.max([s for _, s in model_scores]) - np.min([s for _, s in model_scores]))

    context_details = build_context_score(input_text)
    snippets        = _get_retriever().retrieve(input_text)
    ev_score        = _evidence_score(snippets)

   
    final_score = combine_scores(
    transformer_score=avg_model_score,
    context_score=context_details["context_score"],
    evidence_score=ev_score,
    model_key="roberta",   # or "ensemble" — any key in DEFAULT_WEIGHTS
    weights=DEFAULT_WEIGHTS,
)

    label      = "FAKE" if final_score >= 0.5 else "REAL"
    confidence = final_score if label == "FAKE" else 1.0 - final_score

    explanation = build_explanation(
        label, avg_model_score, context_details, snippets,
        model_name="Ensemble",
    )

    # Build note — flag disagreement between models
    note_parts = []
    if score_spread > 0.3:
        scores_str = ", ".join(f"{k.upper()}: {s:.2f}" for k, s in model_scores)
        note_parts.append(f"⚠️ Models disagree significantly ({scores_str}). Input may be out-of-distribution.")
    base_note = _build_note(avg_model_score, confidence, word_count)
    if base_note:
        note_parts.append(base_note)
    if failed_models:
        note_parts.append(f"⚠️ Could not load: {', '.join(failed_models)}.")

    note = " ".join(note_parts) if note_parts else None

    return HybridPredictionResult(
        label=label,
        confidence=round(confidence, 4),
        model_key="ensemble",
        model_display_name="Ensemble (BERT + RoBERTa + DistilBERT)",
        breakdown={
            "model_score":    round(avg_model_score, 4),
            "context_score":  round(context_details["context_score"], 4),
            "evidence_score": round(ev_score, 4),
            "score_spread":   round(score_spread, 4),
            "per_model":      {k: round(s, 4) for k, s in model_scores},
        },
        explanation=explanation,
        evidence_snippets=snippets,
        details={
            "sentiment":  context_details["sentiment"],
            "linguistic": context_details["linguistic"],
        },
        note=note,
    )