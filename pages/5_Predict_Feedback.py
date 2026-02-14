import streamlit as st
from src.app_state import init_state
from src.text_preprocessing import clean_text
from src.modeling import predict_with_probability

init_state()

st.title("🔮 Predict Feedback")

if not st.session_state.trained or st.session_state.model is None or st.session_state.vectorizer is None:
    st.warning("Train the model first in **🏗️ Model Building**.")
    st.stop()

user_text = st.text_area("Enter new feedback", placeholder="e.g. The food was not good.")

if st.button("🔍 Predict"):
    opts = st.session_state.cleaning_opts

    cleaned = clean_text(
        user_text,
        lowercase=opts["lowercase"],
        remove_non_letters=opts["remove_non_letters"],
        remove_stopwords=opts["remove_stopwords"],
        keep_not=opts["keep_not"],
        stemming=opts["stemming"],
    )

    pred, pred_prob, full_proba = predict_with_probability(
        st.session_state.model,
        st.session_state.vectorizer,
        cleaned,
    )

    pct = int(round(pred_prob * 100))

    if pred == 1:
        st.success(f"✅ {pct}% probable **Positive**")
    else:
        st.error(f"❌ {pct}% probable **Negative**")

    st.caption(f"P(Negative)={full_proba[0]:.4f}, P(Positive)={full_proba[1]:.4f}")
