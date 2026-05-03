import re
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - import fallback
    SentenceTransformer = None


class SentenceSimilarity:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = None
        self.vectorizer = None
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # Fallback mode for offline environments without embedding downloads.
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            return self.vectorizer.fit_transform(texts).toarray()
        return self.vectorizer.transform(texts).toarray()

    def compare(self, query: str, candidates: List[str]) -> np.ndarray:
        if not candidates:
            return np.array([])
        if self.model is None:
            matrix = TfidfVectorizer(max_features=5000, stop_words="english").fit_transform([query] + candidates)
            query_embedding = matrix[0:1].toarray()
            candidate_embeddings = matrix[1:].toarray()
        else:
            query_embedding = self.encode([query])
            candidate_embeddings = self.encode(candidates)
        scores = cosine_similarity(query_embedding, candidate_embeddings)[0]
        return scores


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]
