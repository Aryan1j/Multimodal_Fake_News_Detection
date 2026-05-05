
# Run this ONCE after your models are trained.
# It builds the calibration files and evidence index that the hybrid layer needs.


import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ARTIFACTS_DIR, DATA_CSV, LABEL_REAL, MAX_LENGTH,
    MODEL_KEY_TO_DIR, SPLIT_JSON_PATH,
    EVIDENCE_CORPUS_PATH, EVIDENCE_INDEX_PATH,
    get_model_artifact_dir,
)
from src.inference import load_model

FUSION_DEFAULTS = {"roberta": 0.6, "context": 0.2, "evidence": 0.2}


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / exp_v.sum(axis=1, keepdims=True)


def get_raw_probs(model_key, texts):
    tokenizer, model, device = load_model(model_key)
    probs = []
    for text in tqdm(texts, desc=f"  Scoring with {model_key}"):
        enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH,
                        padding="max_length", return_tensors="pt")
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits.cpu().numpy()
        probs.append(float(softmax(logits)[0, 1]))
    return np.array(probs)


def build_calibrator(model_key, val_texts, val_labels):
    print(f"\n[{model_key}] Fitting calibrator ({len(val_texts)} samples)...")
    raw = get_raw_probs(model_key, val_texts)
    cal = LogisticRegression(max_iter=1000)
    cal.fit(raw.reshape(-1, 1), val_labels)

    out_dir = get_model_artifact_dir(model_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, out_dir / "calibrator.joblib")
    (out_dir / "fusion_weights.json").write_text(
        json.dumps(FUSION_DEFAULTS, indent=2), encoding="utf-8"
    )
    print(f"  Saved to artifacts/{model_key}/")


def build_evidence_index(train_texts, train_labels):
    print("\n[shared] Building evidence index...")
    real = [t for t, l in zip(train_texts, train_labels) if l == LABEL_REAL][:5000]
    EVIDENCE_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": real}).to_csv(EVIDENCE_CORPUS_PATH, index=False)
    print(f"  Saved {len(real)} real-news articles as evidence corpus")

    from fact_check.retriever import LocalEvidenceRetriever
    LocalEvidenceRetriever()
    print(f"  Evidence index built at artifacts/evidence_index.joblib")


def main():
    if not SPLIT_JSON_PATH.exists():
        print("ERROR: No data split found.")
        print("Run this first:  python -m src.evaluate")
        sys.exit(1)

    split = json.loads(SPLIT_JSON_PATH.read_text())
    df    = pd.read_csv(DATA_CSV).dropna(subset=["text", "label"]).reset_index(drop=True)

    val_idx   = [i for i in split["val_indices"]   if i < len(df)]
    train_idx = [i for i in split["train_indices"] if i < len(df)]

    val_texts    = df.iloc[val_idx]["text"].tolist()
    val_labels   = df.iloc[val_idx]["label"].astype(int).tolist()
    train_texts  = df.iloc[train_idx]["text"].tolist()
    train_labels = df.iloc[train_idx]["label"].astype(int).tolist()

    for key in ["bert", "roberta", "distilbert"]:
        if not MODEL_KEY_TO_DIR[key].exists():
            print(f"\n[{key}] Skipping — model not trained yet")
            continue
        build_calibrator(key, val_texts, val_labels)

    build_evidence_index(train_texts, train_labels)
    print("\nAll done! Run the app:  python -m app.app")


if __name__ == "__main__":
    main()