"""
Improved Score Fusion for Hybrid Fake News Detector.

Key improvements:
1. Adaptive weights: if context/evidence are weak signals, trust transformer more
2. Confidence-gated context: only use context score when it's decisive (>0.65 or <0.35)
3. Evidence quality check: only apply evidence when similarity is high enough to trust
4. Better explanation messages tied to actual scores
"""

from typing import Dict, List


DEFAULT_WEIGHTS = {
    "roberta": 0.70,   # Transformer is the strongest signal — raised from 0.60
    "context": 0.15,   # Context is supplementary — lowered from 0.20
    "evidence": 0.15,  # Evidence is supplementary — lowered from 0.20
}

# When context or evidence are in the "uncertain middle" zone,
# their contribution is reduced to avoid pulling away a correct transformer score
UNCERTAIN_ZONE_LOW = 0.35
UNCERTAIN_ZONE_HIGH = 0.65


def _is_uncertain(score: float) -> bool:
    """Returns True if a score is in the uncertain middle zone."""
    return UNCERTAIN_ZONE_LOW <= score <= UNCERTAIN_ZONE_HIGH


def combine_scores(
    roberta_score: float,
    context_score: float,
    evidence_score: float,
    weights: Dict[str, float] | None = None,
) -> float:
    """
    Combine transformer, context, and evidence scores with adaptive weighting.
    
    If context or evidence are uncertain (near 0.5), their weight is
    redistributed to the transformer score to avoid noise-driven errors.
    """
    active_weights = dict(weights or DEFAULT_WEIGHTS)

    # Adaptive weight redistribution
    context_uncertain = _is_uncertain(context_score)
    evidence_uncertain = _is_uncertain(evidence_score)

    if context_uncertain and evidence_uncertain:
        # Both supplementary signals are uncertain — trust transformer almost fully
        w_model = active_weights["roberta"] + active_weights["context"] * 0.7 + active_weights["evidence"] * 0.7
        w_context = active_weights["context"] * 0.3
        w_evidence = active_weights["evidence"] * 0.3
    elif context_uncertain:
        w_model = active_weights["roberta"] + active_weights["context"] * 0.6
        w_context = active_weights["context"] * 0.4
        w_evidence = active_weights["evidence"]
    elif evidence_uncertain:
        w_model = active_weights["roberta"] + active_weights["evidence"] * 0.6
        w_context = active_weights["context"]
        w_evidence = active_weights["evidence"] * 0.4
    else:
        w_model = active_weights["roberta"]
        w_context = active_weights["context"]
        w_evidence = active_weights["evidence"]

    # Normalise weights to sum to 1.0
    total = w_model + w_context + w_evidence
    w_model /= total
    w_context /= total
    w_evidence /= total

    final_score = (
        w_model * roberta_score
        + w_context * context_score
        + w_evidence * evidence_score
    )
    return max(0.0, min(1.0, final_score))


def build_explanation(
    final_label: str,
    roberta_score: float,
    context_details: dict,
    evidence_snippets: List[dict],
    model_name: str = "The classifier",
) -> List[str]:
    """Build human-readable explanation lines for the prediction."""
    lines = []
    linguistic = context_details.get("linguistic", {})
    sentiment = context_details.get("sentiment", {})

    # Lead with transformer verdict and confidence
    model_pct = int(roberta_score * 100)
    if roberta_score >= 0.75:
        lines.append(f"{model_name} is strongly confident this is FAKE ({model_pct}% fake score).")
    elif roberta_score >= 0.55:
        lines.append(f"{model_name} leans toward FAKE with a score of {model_pct}%.")
    elif roberta_score <= 0.25:
        lines.append(f"{model_name} is strongly confident this is REAL ({100 - model_pct}% real score).")
    elif roberta_score <= 0.45:
        lines.append(f"{model_name} leans toward REAL with a score of {100 - model_pct}% real.")
    else:
        lines.append(f"{model_name} is uncertain (score near 50%) — treat result with caution.")

    # Linguistic signals
    if linguistic.get("clickbait_score", 0) > 0.25:
        lines.append("Clickbait or sensational language patterns were detected in the text.")
    if linguistic.get("capitalization_ratio", 0) > 0.10:
        lines.append("Excessive capitalisation suggests sensational or emotionally charged writing.")
    if linguistic.get("credibility_score", 0) > 0.3:
        lines.append("The text contains credibility markers (source attribution, statistics, quotes).")

    # Sentiment signals
    if sentiment.get("amplifier_density", 0) > 0.2:
        lines.append("Emotional amplifier words (e.g. 'horrific', 'bombshell') were found — a common fake news pattern.")
    elif sentiment.get("subjectivity", 0) > 0.60:
        lines.append("The text is highly subjective, which increases suspicion of bias or fabrication.")
    if sentiment.get("hedge_suspicion", 0) > 0.7:
        lines.append("The text lacks journalistic hedging (e.g. 'reportedly', 'according to'), which is unusual for real news.")

    # Evidence signals
    if evidence_snippets:
        high_sim = [s for s in evidence_snippets if s["similarity"] > 0.5]
        support_count = sum(1 for s in high_sim if s["stance"] == "support")
        contradict_count = sum(1 for s in high_sim if s["stance"] == "contradict")
        if contradict_count > support_count:
            lines.append("Retrieved evidence from the knowledge base contradicts the main claims.")
        elif support_count > 0:
            lines.append("Retrieved evidence partially supports the claims in this article.")
        else:
            lines.append("Evidence retrieved is related but not strongly conclusive either way.")
    else:
        lines.append("No matching evidence was found in the local knowledge base.")

    return lines[:5]