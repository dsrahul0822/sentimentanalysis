import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

st.title("🧠 Model Training (Logistic Regression + CountVectorizer)")

# --- Guardrails: data must be loaded first ---
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Please load data first from the **Data Load** page.")
    st.stop()

df = st.session_state.df
text_col = st.session_state.text_col
label_col = st.session_state.label_col if st.session_state.label_col and st.session_state.label_col != "<None>" else None

st.write(f"Using text column: **{text_col}**")
if label_col is None:
    st.error("No label column selected. Please go back to Data Load and select a binary label column (0/1).")
    st.stop()
else:
    st.write(f"Using label column: **{label_col}**")

# --- Validate binary labels ---
labels = df[label_col].dropna().unique()
if len(labels) != 2:
    st.error(f"Label column appears to have {len(labels)} unique values: {labels}. Please provide a binary 0/1 label.")
    st.stop()

# --- Hyperparameters/UI controls ---
test_size = st.slider("Test size (fraction)", 0.1, 0.4, 0.2, 0.05)
min_df = st.number_input("min_df (min docs a term must appear in)", min_value=1, value=2, step=1)
max_features = st.number_input("max_features (0 = no cap)", min_value=0, value=30000, step=1000)
use_binary = st.checkbox("Use binary counts (presence/absence instead of term counts)", value=False)
filename = st.text_input("Filename for model (.pkl)", value="model_pipeline.pkl")

# --- Data split ---
X = df[text_col].astype(str).values
y = df[label_col].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# --- Build pipeline (CountVectorizer with unigrams only) ---
vec = CountVectorizer(
    ngram_range=(1, 1),          # unigrams only (no n-grams)
    min_df=min_df,
    max_features=None if max_features == 0 else max_features,
    binary=use_binary
)
clf = LogisticRegression(max_iter=1000)
pipe = Pipeline([("bow", vec), ("logreg", clf)])

# --- Single-step train & download ---
if st.button("Train & Download Model"):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Probabilities (or scaled decision scores)
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        from sklearn.preprocessing import MinMaxScaler
        scores = pipe.decision_function(X_test).reshape(-1, 1)
        y_proba = MinMaxScaler().fit_transform(scores).ravel()

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float("nan")

    st.success(
        f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | "
        f"F1: {f1:.4f} | ROC-AUC: {auc if not np.isnan(auc) else 'nan'}"
    )

    # Serialize to in-memory buffer so download works immediately
    buffer = io.BytesIO()
    pickle.dump(pipe, buffer)
    buffer.seek(0)

    st.download_button(
        label="⬇️ Download model .pkl",
        data=buffer,
        file_name=filename if filename.strip() else "model_pipeline.pkl",
        mime="application/octet-stream",
    )

    # Optional: also allow saving server-side if user wants a copy
    with st.expander("Optional: Save a copy on the server"):
        save_server = st.checkbox("Also save to server filesystem", value=False)
        if save_server:
            try:
                with open(filename if filename.strip() else "model_pipeline.pkl", "wb") as f:
                    pickle.dump(pipe, f)
                st.success(f"Saved on server as: {filename}")
            except Exception as e:
                st.error(f"Could not save on server: {e}")
