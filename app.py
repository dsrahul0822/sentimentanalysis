import streamlit as st

st.set_page_config(
    page_title="Restaurant Sentiment Demo",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Restaurant Sentiment (Demo App)")

st.write(
    """
Multi-page demo app to:
- Upload data
- Choose text cleaning options
- Train a Logistic Regression sentiment model
- Evaluate it (accuracy + confusion matrix)
- Predict sentiment for a new review with probability
"""
)

# ---------- Initialize session state ----------
DEFAULT_KEYS = {
    "df": None,
    "text_col": "Review",
    "label_col": "Liked",
    "cleaning_opts": {
        "lowercase": True,
        "remove_non_letters": True,
        "remove_stopwords": True,
        "keep_not": True,
        "stemming": True,
    },
    "max_rows": 1000,
    "corpus": None,
    "vectorizer_kind": "TF-IDF",
    "max_features": 1500,
    "test_size": 0.2,
    "random_state": 42,
    "vectorizer": None,
    "model": None,
    "X": None,
    "y": None,
    "x_train": None,
    "x_test": None,
    "y_train": None,
    "y_test": None,
    "y_pred": None,
    "accuracy": None,
    "conf_matrix": None,
    "trained": False,
}

for k, v in DEFAULT_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.success("Use the sidebar to open pages: Data Upload → Cleaning → Model Building → Evaluation → Predict")

st.markdown("### Quick Flow")
st.markdown(
    """
1. **📤 Data Upload** → Upload TSV and select columns  
2. **🧹 Data Cleaning** → Choose preprocessing options + build corpus  
3. **🏗️ Model Building** → Choose test size + vectorizer + train  
4. **📊 Model Evaluation** → View accuracy + confusion matrix  
5. **🔮 Predict Feedback** → Enter new review → get probability
"""
)
