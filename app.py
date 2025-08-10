import streamlit as st

st.set_page_config(page_title="Restaurant Review Sentiment App", page_icon="🍽️", layout="wide")

st.title("🍽️ Restaurant Review Sentiment – End-to-End App")

st.markdown(
    """
**Pages**
- **Data Load:** Upload CSV/TSV/Excel, pick text & label columns.
- **Data Visualization:** WordCloud + word frequency chart.
- **Model Training:** Train Logistic Regression (TF-IDF) and download a `.pkl` pipeline.
- **Predict:** Load a saved `.pkl` and predict sentiment probabilities for new feedback.
"""
)

st.success("Tip: Use the sidebar to navigate pages. Start with **Data Load**.")
