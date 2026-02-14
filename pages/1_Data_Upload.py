import streamlit as st
import pandas as pd
from src.app_state import init_state

init_state()

st.title("📤 Data Upload")

uploaded = st.file_uploader("Upload Restaurant_Reviews.tsv", type=["tsv"])

if uploaded is None:
    st.warning("Upload a TSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded, delimiter="\t")
st.session_state.df = df

cols = df.columns.tolist()

# Safe defaults
text_guess = "Review" if "Review" in cols else cols[0]
label_guess = "Liked" if "Liked" in cols else cols[-1]

st.session_state.text_col = st.selectbox("Text column", options=cols, index=cols.index(text_guess))
st.session_state.label_col = st.selectbox("Label column (0/1)", options=cols, index=cols.index(label_guess))

st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head(20), use_container_width=True)

st.info("Next: go to **🧹 Data Cleaning** page.")
