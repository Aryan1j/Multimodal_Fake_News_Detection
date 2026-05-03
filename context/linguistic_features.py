"""
Improved Linguistic Features for Fake News Detection.

Key improvements:
- Expanded clickbait/fake-news phrase list (50+ patterns vs 12)
- Added fake-news structural markers (ALL CAPS titles, excessive punctuation)
- Added credibility signals (named sources, statistics, quotes)
- Better suspicion formula with credibility REDUCING suspicion
"""

import re
import math
import textstat


# Expanded fake-news / clickbait patterns
CLICKBAIT_PATTERNS = [
    # Classic clickbait
    "you won't believe", "what happened next", "this is why",
    "the truth about", "they don't want you to know",
    "what they're hiding", "the real story",
    "jaw-dropping", "mind-blowing", "mind blowing",
    "must see", "must read", "you need to know",
    "here's what", "this is what",
    # Sensationalism
    "shocking", "unbelievable", "incredible", "insane",
    "bombshell", "explosive", "breaking", "urgent", "alert",
    "viral", "secret revealed", "leaked", "exposed", "uncovered",
    "banned", "censored", "silenced", "suppressed",
    # Fear / outrage bait
    "terrifying", "disturbing", "horrifying", "outrageous",
    "destroying", "collapsing", "invasion", "crisis",
    "threat to", "end of", "goodbye to",
    # Conspiracy language
    "deep state", "mainstream media", "fake news" , "false flag",
    "cover-up", "coverup", "cover up", "they lied",
    "globalist", "new world order", "elite agenda",
    "wake up", "sheeple", "do your research",
    # Pseudo-medical / pseudo-science
    "doctors hate", "big pharma", "cure they don't want",
    "natural remedy", "miracle cure", "banned cure",
    # Political rage bait
    "radical left", "far-left", "far-right", "regime",
    "communist", "socialist agenda", "tyranny",
    "traitor", "treason", "puppet", "rigged",
]

# Credibility markers — REAL journalism tends to have these
CREDIBILITY_MARKERS = [
    # Source attribution
    r"\baccording to\b", r"\bsaid in a statement\b", r"\btold reporters\b",
    r"\bconfirmed by\b", r"\bsources say\b", r"\breported by\b",
    r"\bofficial statement\b", r"\bpress conference\b",
    # Named organisations
    r"\b(reuters|ap|associated press|bbc|cnn|nyt|washington post)\b",
    # Statistical language
    r"\bpercent\b", r"\bstatistics\b", r"\bdata shows\b",
    r"\bstudy found\b", r"\bresearch shows\b", r"\bsurvey\b",
    # Direct quotes
    r'"[^"]{10,}"',   # quoted speech of 10+ chars
    # Date / time anchoring (real news is time-specific)
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b20\d\d\b",    # year like 2024
]


def _safe_divide(n: float, d: float) -> float:
    return n / d if d else 0.0


def extract_linguistic_features(text: str) -> dict:
    content = (text or "").strip()
    tokens = re.findall(r"\b\w+\b", content)
    token_count = max(1, len(tokens))

    # ALL-CAPS tokens (excludes short acronyms like "US", "UK")
    uppercase_tokens = sum(1 for t in tokens if len(t) > 2 and t.isupper())
    capitalization_ratio = _safe_divide(uppercase_tokens, token_count)

    # Punctuation intensity
    exclamation_count = content.count("!")
    question_count = content.count("?")
    punctuation_intensity = min(
        1.0, _safe_divide(exclamation_count + question_count, max(1, token_count / 5))
    )

    # Digit ratio
    digit_ratio = _safe_divide(sum(c.isdigit() for c in content), max(1, len(content)))

    # Clickbait score — expanded list
    lower = content.lower()
    clickbait_hits = sum(1 for phrase in CLICKBAIT_PATTERNS if phrase in lower)
    clickbait_score = min(1.0, clickbait_hits / 4.0)  # 4 hits = max score

    # Credibility score — regex-based
    credibility_hits = sum(
        1 for pattern in CREDIBILITY_MARKERS if re.search(pattern, lower)
    )
    credibility_score = min(1.0, credibility_hits / 5.0)  # 5 hits = max credibility

    # Readability
    try:
        readability = float(textstat.flesch_reading_ease(content)) if content else 50.0
    except Exception:
        readability = 50.0
    # Very low readability (hard to read) OR very high (oversimplified) can be suspicious
    # Normal journalism: 30-70 Flesch score
    readability_suspicion = max(0.0, min(1.0, (abs(readability - 50.0) - 15.0) / 35.0))

    # Repetition suspicion: fake news often repeats the same phrases
    if token_count > 20:
        unique_ratio = len(set(t.lower() for t in tokens)) / token_count
        repetition_suspicion = max(0.0, 1.0 - unique_ratio * 1.5)
    else:
        repetition_suspicion = 0.0

    # Base suspicion from fake signals
    raw_suspicion = min(
        1.0,
        (0.35 * clickbait_score)
        + (0.25 * min(1.0, capitalization_ratio * 5.0))
        + (0.20 * punctuation_intensity)
        + (0.10 * readability_suspicion)
        + (0.05 * repetition_suspicion)
        + (0.05 * min(1.0, digit_ratio * 8.0))
    )

    # Credibility REDUCES suspicion (real signals push it down)
    suspicion = max(0.0, raw_suspicion - credibility_score * 0.25)

    return {
        "clickbait_score": clickbait_score,
        "capitalization_ratio": capitalization_ratio,
        "punctuation_intensity": punctuation_intensity,
        "readability_score": readability,
        "readability_suspicion": readability_suspicion,
        "digit_ratio": digit_ratio,
        "credibility_score": credibility_score,
        "repetition_suspicion": repetition_suspicion,
        "linguistic_suspicion": suspicion,
    }


def build_context_score(text: str) -> dict:
    from context.sentiment import analyze_sentiment

    sentiment_features = analyze_sentiment(text)
    linguistic_features = extract_linguistic_features(text)

    # Credibility from linguistic reduces context score
    credibility_bonus = linguistic_features["credibility_score"] * 0.15

    context_score = max(
        0.0,
        min(
            1.0,
            (0.40 * linguistic_features["linguistic_suspicion"])
            + (0.35 * sentiment_features["sentiment_suspicion"])
            + (0.15 * linguistic_features["clickbait_score"])
            + (0.10 * sentiment_features.get("amplifier_density", 0.0))
            - credibility_bonus
        )
    )

    return {
        "context_score": context_score,
        "sentiment": sentiment_features,
        "linguistic": linguistic_features,
    }