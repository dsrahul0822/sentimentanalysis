import streamlit as st
import pandas as pd
from src.app_state import init_state
from src.modeling import evaluate

init_state()

st.title("📊 Model Evaluation")

if not st.session_state.trained or st.session_state.model is None:
    st.warning("Train the model first in **🏗️ Model Building**.")
    st.stop()

if st.button("✅ Evaluate on Test Set"):
    acc, cm, y_pred = evaluate(st.session_state.model, st.session_state.x_test, st.session_state.y_test)
    st.session_state.accuracy = acc
    st.session_state.conf_matrix = cm
    st.session_state.y_pred = y_pred

if st.session_state.accuracy is not None:
    st.metric("Accuracy", f"{st.session_state.accuracy:.4f}")

if st.session_state.conf_matrix is not None:
    cm = st.session_state.conf_matrix
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0 (Negative)", "Actual 1 (Positive)"],
        columns=["Pred 0 (Negative)", "Pred 1 (Positive)"],
    )
    st.subheader("Confusion Matrix")
    st.dataframe(cm_df, use_container_width=True)
