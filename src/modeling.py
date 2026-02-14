from typing import Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def make_vectorizer(kind: str = "TF-IDF", max_features: int = 1500):
    kind = (kind or "TF-IDF").strip().lower()
    if kind in ["tf-idf", "tfidf", "tf idf", "tf-idf vectorizer", "tfidfvectorizer"]:
        return TfidfVectorizer(max_features=max_features)
    return CountVectorizer(max_features=max_features)


def train_logreg(
    corpus,
    y,
    *,
    vectorizer_kind: str = "TF-IDF",
    max_features: int = 1500,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Train a Logistic Regression model from corpus + labels.
    Returns a dict with model, vectorizer, splits.
    """
    vectorizer = make_vectorizer(vectorizer_kind, max_features=max_features)
    X = vectorizer.fit_transform(corpus).toarray()

    # stratify keeps class balance stable
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    return {
        "vectorizer": vectorizer,
        "model": model,
        "X": X,
        "y": y,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def evaluate(model, x_test, y_test) -> Tuple[float, np.ndarray, np.ndarray]:
    y_pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm, y_pred


def predict_with_probability(model, vectorizer, cleaned_text: str):
    """
    Returns:
      pred (0/1),
      pred_prob (probability for predicted class),
      full_proba = [P(class0), P(class1)]
    """
    X_new = vectorizer.transform([cleaned_text]).toarray()
    proba = model.predict_proba(X_new)[0]
    pred = int(model.predict(X_new)[0])
    pred_prob = float(proba[pred])
    return pred, pred_prob, proba
