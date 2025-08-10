import streamlit as st
import pickle
import numpy as np

st.title("🔮 Predict Sentiment")

# --- Upload model ---
uploaded_model = st.file_uploader("Upload your trained model (.pkl)", type=["pkl"])
model = None
if uploaded_model is not None:
    try:
        model = pickle.load(uploaded_model)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# --- Input text ---
txt = st.text_area("Enter new feedback text", height=150, placeholder="Type a restaurant review here...")

# Single button with a unique key
predict_clicked = st.button("Predict", key="predict_btn")

if predict_clicked:
    if model is None:
        st.warning("Please upload a trained model first.")
    elif not txt.strip():
        st.warning("Please enter some text to predict.")
    else:
        try:
            # Prefer predict_proba; fallback to decision_function
            if hasattr(model, "predict_proba"):
                p = float(model.predict_proba([txt])[0, 1])
            else:
                from sklearn.preprocessing import MinMaxScaler
                score = model.decision_function([txt]).reshape(-1, 1)
                p = float(MinMaxScaler().fit_transform(score).ravel()[0])

            label = "Positive" if p >= 0.5 else "Negative"
            st.info(f"Predicted probability of positive: {p:.4f} → **{label}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
