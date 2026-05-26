import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from .database import Dataset, ModelMetadata
from .database import db as _db


_MODEL_BUNDLE_FILENAME = "model_bundle.joblib"


def _ensure_nltk():
    """
    Download NLTK corpora if missing.
    This is best-effort; failures are handled by using minimal stopwords.
    """
    try:
        import nltk

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
    except Exception:
        return


def _get_stopwords() -> set:
    _ensure_nltk()
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        # Minimal fallback list.
        return {
            "the",
            "and",
            "a",
            "an",
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "or",
            "as",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "it",
            "this",
            "that",
            "these",
            "those",
        }


_STOPWORDS = _get_stopwords()
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("\n", " ")
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    if not text:
        return ""
    tokens = [t for t in text.split(" ") if t and t not in _STOPWORDS]
    return " ".join(tokens).strip()


def _user_model_dir(app_root: str, user_id: int) -> str:
    models_root = os.path.join(app_root, "models")
    return os.path.join(models_root, f"user_{user_id}")


def _bundle_paths(app_root: str, user_id: int) -> Tuple[str, str]:
    model_dir = _user_model_dir(app_root, user_id)
    return (
        os.path.join(model_dir, _MODEL_BUNDLE_FILENAME),
        model_dir,
    )


@dataclass
class TrainResult:
    accuracy: float
    trained_docs: int
    categories: List[str]
    status: str


def train_user_model(app_root: str, user_id: int, datasets: List[Dataset]) -> TrainResult:
    """
    Train a per-user TF-IDF + Logistic Regression classifier.
    """
    extracted_texts: List[str] = []
    labels: List[str] = []
    for d in datasets:
        if not d.extracted_text or not d.extracted_text.strip():
            continue
        processed = preprocess_text(d.extracted_text)
        if not processed.strip():
            continue
        extracted_texts.append(processed)
        labels.append(d.category)

    trained_docs = len(labels)
    categories = sorted(list({l for l in labels}))

    if trained_docs < 2 or len(categories) < 2:
        return TrainResult(
            accuracy=0.0,
            trained_docs=trained_docs,
            categories=categories,
            status="not_enough_data",
        )

    # Ensure stratified split possible.
    # If some classes have 1 sample, train_test_split may fail; handle it gracefully.
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            extracted_texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
    except Exception:
        # Fallback to non-stratified split.
        x_train, x_test, y_train, y_test = train_test_split(
            extracted_texts, labels, test_size=0.2, random_state=42
        )

    vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(
        max_iter=500,
        n_jobs=1,
    )
    model.fit(x_train_vec, y_train)
    accuracy = float(model.score(x_test_vec, y_test)) if len(y_test) else 0.0

    bundle = {
        "vectorizer": vectorizer,
        "model": model,
        "categories": categories,
    }

    bundle_path, model_dir = _bundle_paths(app_root, user_id)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(bundle, bundle_path)

    # Persist metadata in DB for analytics.
    meta = ModelMetadata.query.filter_by(user_id=user_id).first()
    if meta is None:
        meta = ModelMetadata(user_id=user_id, model_path=bundle_path, vectorizer_path=bundle_path)
        _db.session.add(meta)
    meta.accuracy = accuracy
    meta.trained_docs = trained_docs
    meta.categories = json.dumps(categories)
    meta.labels = json.dumps(model.classes_.tolist() if hasattr(model, "classes_") else categories)
    meta.model_path = bundle_path
    meta.vectorizer_path = bundle_path
    _db.session.commit()

    return TrainResult(
        accuracy=accuracy,
        trained_docs=trained_docs,
        categories=categories,
        status="trained",
    )


def load_user_model(app_root: str, user_id: int) -> Optional[Dict]:
    bundle_path, _ = _bundle_paths(app_root, user_id)
    if not os.path.exists(bundle_path):
        return None
    data = joblib.load(bundle_path)
    if not isinstance(data, dict) or "vectorizer" not in data or "model" not in data:
        return None
    return data


def _get_dataset_texts_for_duplicates(datasets: List[Dataset]) -> List[str]:
    processed = []
    for d in datasets:
        p = preprocess_text(d.extracted_text)
        if p.strip():
            processed.append(p)
    return processed


def generate_tags_from_vector(tfidf_vector, vectorizer, top_n: int = 8) -> List[str]:
    """
    tfidf_vector: 1 x vocab sparse matrix
    """
    if tfidf_vector is None:
        return []
    try:
        scores = tfidf_vector.toarray().ravel()
        if scores.size == 0:
            return []
        top_idx = np.argsort(scores)[::-1][:top_n]
        feature_names = np.array(vectorizer.get_feature_names_out())
        tags = []
        for idx in top_idx:
            term = str(feature_names[idx])
            if not term or term in _STOPWORDS:
                continue
            # Filter obvious noise.
            if len(term) < 3:
                continue
            tags.append(term)
        # De-duplicate while preserving order.
        seen = set()
        result = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result[:top_n]
    except Exception:
        return []


def predict_user_document(
    app_root: str, user_id: int, extracted_text: str, datasets: List[Dataset]
) -> Dict:
    """
    Returns prediction + duplicate similarity + tags + keywords.
    """
    extracted_text = (extracted_text or "").strip()
    processed = preprocess_text(extracted_text)
    if not processed:
        raise ValueError("Could not extract meaningful text from the document")

    bundle = load_user_model(app_root, user_id)
    vectorizer = None
    model = None

    if bundle is not None:
        vectorizer = bundle["vectorizer"]
        model = bundle["model"]

    if vectorizer is None or model is None:
        raise ValueError("Model not trained yet. Please train your model first.")

    vec = vectorizer.transform([processed])
    probs = model.predict_proba(vec)
    max_idx = int(np.argmax(probs))
    confidence = float(probs[0, max_idx]) if probs.size else 0.0
    predicted_category = str(model.classes_[max_idx]) if hasattr(model, "classes_") else ""

    # Duplicate detection using cosine similarity with existing dataset vectors.
    dataset_texts_processed = _get_dataset_texts_for_duplicates(datasets)
    duplicate_similarity = 0.0
    if dataset_texts_processed:
        dataset_vecs = vectorizer.transform(dataset_texts_processed)
        sims = cosine_similarity(vec, dataset_vecs).ravel()
        duplicate_similarity = float(np.max(sims)) * 100.0 if sims.size else 0.0

    tags = generate_tags_from_vector(vec, vectorizer, top_n=8)
    keywords = tags  # For the UI highlight we can reuse tags.

    # Extract text preview (raw) is handled by caller; we return only derived fields here.
    return {
        "predicted_category": predicted_category,
        "confidence": confidence,
        "duplicate_similarity": duplicate_similarity,
        "tags": tags,
        "keywords": keywords,
    }

