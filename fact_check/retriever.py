import json
import re
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from fact_check.similarity import SentenceSimilarity, split_into_sentences


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_EVIDENCE_DATA = ARTIFACT_DIR / "evidence_corpus.csv"
DEFAULT_INDEX_PATH = ARTIFACT_DIR / "evidence_index.joblib"
DEFAULT_META_PATH = ARTIFACT_DIR / "evidence_meta.json"


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_key_claims(text: str, max_claims: int = 3) -> List[str]:
    sentences = split_into_sentences(text)
    ranked = sorted(
        sentences,
        key=lambda sentence: len(re.findall(r"\b[A-Z][a-z]+\b", sentence)) + len(sentence.split()),
        reverse=True,
    )
    return ranked[:max_claims] if ranked else [text[:300]]


class LocalEvidenceRetriever:
    def __init__(
        self,
        similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        corpus_path: Path = DEFAULT_EVIDENCE_DATA,
        index_path: Path = DEFAULT_INDEX_PATH,
        meta_path: Path = DEFAULT_META_PATH,
    ):
        self.similarity = SentenceSimilarity(similarity_model)
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.neighbors = None
        self.embeddings = None
        self.corpus = []
        self._load()

    def _load(self):
        if self.index_path.exists() and self.meta_path.exists():
            payload = joblib.load(self.index_path)
            self.neighbors = payload["neighbors"]
            self.embeddings = payload["embeddings"]
            self.corpus = payload["corpus"]
            self.similarity.vectorizer = payload.get("vectorizer")
            return

        if self.corpus_path.exists():
            df = pd.read_csv(self.corpus_path)
            self.build_index(df["text"].dropna().astype(str).tolist())

    def build_index(self, snippets: List[str]):
        clean_snippets = [_normalize_whitespace(item) for item in snippets if _normalize_whitespace(item)]
        self.corpus = clean_snippets[:5000]
        if not self.corpus:
            return

        self.embeddings = self.similarity.encode(self.corpus)
        self.neighbors = NearestNeighbors(metric="cosine", n_neighbors=min(10, len(self.corpus)))
        self.neighbors.fit(self.embeddings)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "neighbors": self.neighbors,
                "embeddings": self.embeddings,
                "corpus": self.corpus,
                "vectorizer": self.similarity.vectorizer,
            },
            self.index_path,
        )
        self.meta_path.write_text(json.dumps({"size": len(self.corpus)}, indent=2), encoding="utf-8")

    def retrieve(self, text: str, top_k: int = 3) -> List[dict]:
        if self.neighbors is None or not self.corpus:
            return []

        claims = extract_key_claims(text)
        query = " ".join(claims)
        query_embedding = self.similarity.encode([query])
        distances, indices = self.neighbors.kneighbors(query_embedding, n_neighbors=min(top_k, len(self.corpus)))

        snippets = []
        for distance, index in zip(distances[0], indices[0]):
            similarity = float(1.0 - distance)
            snippet = self.corpus[int(index)]
            stance = infer_stance(query, snippet, similarity)
            snippets.append(
                {
                    "text": snippet,
                    "similarity": max(0.0, min(1.0, similarity)),
                    "stance": stance,
                }
            )
        return snippets


def infer_stance(claim: str, evidence: str, similarity: float) -> str:
   
    claim_tokens = set(re.findall(r"\b\w+\b", claim.lower()))
    evidence_tokens = set(re.findall(r"\b\w+\b", evidence.lower()))

    # Strong negation words
    negations = {"not", "no", "never", "fake", "false", "deny", "denied",
                 "debunked", "untrue", "incorrect", "wrong", "refuted",
                 "misleading", "misinformation", "disinformation", "hoax"}

    # Factual agreement words
    confirmations = {"confirmed", "verified", "true", "correct", "accurate",
                     "official", "reported", "announced", "published"}

    claim_has_negation = bool(claim_tokens & negations)
    evidence_has_negation = bool(evidence_tokens & negations)
    evidence_has_confirmation = bool(evidence_tokens & confirmations)

    # Token overlap (excluding stopwords)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "has", "have", "had", "do", "did", "will", "would", "could",
                 "in", "on", "at", "to", "for", "of", "and", "or", "but"}
    claim_content = claim_tokens - stopwords
    evidence_content = evidence_tokens - stopwords
    overlap = len(claim_content & evidence_content) / max(1, len(claim_content))

    # Decision logic
    if similarity < 0.25:
        return "unrelated"

    # High similarity + confirmation language = support
    if similarity > 0.65 and evidence_has_confirmation and not evidence_has_negation:
        return "support"

    # High similarity + no negation mismatch + good overlap = support
    if similarity > 0.60 and claim_has_negation == evidence_has_negation and overlap > 0.3:
        return "support"

    # Negation mismatch on related content = contradict
    if similarity > 0.45 and (claim_has_negation != evidence_has_negation) and overlap > 0.2:
        return "contradict"

    # Evidence debunks or labels something as false
    if evidence_has_negation and overlap > 0.25:
        return "contradict"

    return "related"