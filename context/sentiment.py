"""
Improved Sentiment Analysis for Fake News Detection.

Key insight: Raw polarity is NOT a reliable fake news signal.
What matters is:
- HIGH subjectivity (fake news is opinionated, not factual)
- Emotional intensity / extreme language density
- Absence of journalistic hedging (real news hedges claims)
- Sentence-level tone variance (fake news mixes calm + explosive sentences)
"""

from textblob import TextBlob


# Emotionally charged / fear-mongering words common in fake news
EMOTIONAL_AMPLIFIERS = {
    "horrific", "terrifying", "devastating", "outrageous", "disgusting",
    "shocking", "explosive", "bombshell", "alarming", "catastrophic",
    "urgent", "crisis", "chaos", "panic", "threat", "danger", "evil",
    "destroy", "collapse", "corrupt", "traitor", "fraud", "hoax",
    "exposed", "scandal", "lie", "liar", "cheat", "rigged", "stolen",
    "invasion", "attack", "conspiracy", "coverup", "cover-up",
    "mainstream media", "deep state", "elite", "globalist", "radical",
    "insane", "lunatic", "monster", "regime", "puppet",
}

# Hedge words — real journalism uses these; fake news avoids them
HEDGE_WORDS = {
    "allegedly", "reportedly", "according to", "sources say",
    "officials said", "confirmed", "unconfirmed", "claims",
    "appears to", "suggests", "may", "might", "could",
    "spokesperson", "statement", "press release",
}


def analyze_sentiment(text: str) -> dict:
    """
    Analyze text for sentiment features relevant to fake news detection.
    """
    content = (text or "").strip()
    blob = TextBlob(content)
    sentiment = blob.sentiment

    polarity = float(sentiment.polarity)
    subjectivity = float(sentiment.subjectivity)
    polarity_norm = float((polarity + 1.0) / 2.0)
    polarity_extremity = abs(polarity)

    lower = content.lower()
    words = lower.split()
    word_count = max(1, len(words))

    # Emotional amplifier density
    amplifier_hits = sum(1 for word in EMOTIONAL_AMPLIFIERS if word in lower)
    amplifier_density = min(1.0, amplifier_hits / max(1, word_count / 20))

    # Hedge word absence: lack of hedging in long text is suspicious
    hedge_hits = sum(1 for phrase in HEDGE_WORDS if phrase in lower)
    hedge_ratio = min(1.0, hedge_hits / max(1, word_count / 50))
    hedge_suspicion = 1.0 - min(1.0, hedge_ratio * 3.0) if word_count > 50 else 0.0

    # Sentence-level subjectivity variance
    sentences = [str(s) for s in blob.sentences]
    if len(sentences) > 2:
        sent_subjectivities = [TextBlob(s).sentiment.subjectivity for s in sentences]
        subjectivity_variance = min(
            1.0,
            sum((x - subjectivity) ** 2 for x in sent_subjectivities) / len(sent_subjectivities) * 4.0
        )
    else:
        subjectivity_variance = 0.0

    suspicion = min(
        1.0,
        (0.35 * subjectivity)
        + (0.25 * amplifier_density)
        + (0.20 * hedge_suspicion)
        + (0.12 * polarity_extremity)
        + (0.08 * subjectivity_variance)
    )

    return {
        "polarity": polarity_norm,
        "subjectivity": subjectivity,
        "polarity_extremity": polarity_extremity,
        "amplifier_density": amplifier_density,
        "hedge_suspicion": hedge_suspicion,
        "subjectivity_variance": subjectivity_variance,
        "sentiment_suspicion": suspicion,
    }