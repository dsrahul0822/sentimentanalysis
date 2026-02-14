import streamlit as st
from src.app_state import init_state
from src.modeling import train_logreg

init_state()

st.title("🏗️ Model Building")

if st.session_state.df is None:
    st.warning("Upload data first in **📤 Data Upload**.")
    st.stop()

if st.session_state.corpus is None:
    st.warning("Build corpus first in **🧹 Data Cleaning**.")
    st.stop()

df = st.session_state.df
label_col = st.session_state.label_col

if label_col not in df.columns:
    st.error("Label column not found. Please re-select columns in **📤 Data Upload**.")
    st.stop()

st.subheader("Training Options")

col1, col2, col3 = st.columns(3)

with col1:
    st.session_state.vectorizer_kind = st.selectbox(
        "Vectorizer",
        ["TF-IDF", "CountVectorizer"],
        index=0 if st.session_state.vectorizer_kind == "TF-IDF" else 1
    )

with col2:
    st.session_state.max_features = st.number_input(
        "max_features",
        min_value=500,
        max_value=5000,
        value=int(st.session_state.max_features),
        step=100,
    )

with col3:
    st.session_state.test_size = st.slider(
        "Test size",
        min_value=0.1,
        max_value=0.4,
        value=float(st.session_state.test_size),
        step=0.05,
    )

st.session_state.random_state = st.number_input(
    "Random state",
    min_value=0,
    max_value=9999,
    value=int(st.session_state.random_state),
    step=1,
)

y = df[label_col].values
try:
    y = y.astype(int)
except Exception:
    pass

if st.button("🚀 Train Model"):
    out = train_logreg(
        st.session_state.corpus,
        y[:len(st.session_state.corpus)],
        vectorizer_kind=st.session_state.vectorizer_kind,
        max_features=int(st.session_state.max_features),
        test_size=float(st.session_state.test_size),
        random_state=int(st.session_state.random_state),
    )

    st.session_state.vectorizer = out["vectorizer"]
    st.session_state.model = out["model"]
    st.session_state.X = out["X"]
    st.session_state.y = out["y"]
    st.session_state.x_train = out["x_train"]
    st.session_state.x_test = out["x_test"]
    st.session_state.y_train = out["y_train"]
    st.session_state.y_test = out["y_test"]
    st.session_state.trained = True

    st.success("✅ Model trained successfully!")
    st.info("Next: go to **📊 Model Evaluation** page.")
