import streamlit as st
import pandas as pd
from utils import read_any, basic_clean

st.title("📥 Data Load")

if "df" not in st.session_state:
    st.session_state.df = None
if "text_col" not in st.session_state:
    st.session_state.text_col = None
if "label_col" not in st.session_state:
    st.session_state.label_col = None

uploaded = st.file_uploader("Upload your dataset (CSV, TSV, or Excel)", type=["csv","tsv","txt","xlsx","xls"])
if uploaded is not None:
    df = read_any(uploaded, uploaded.name)
    st.session_state.df = df.copy()
    st.success(f"Loaded shape: {df.shape}")
    st.dataframe(df.head(20))

    cols = list(df.columns)

    # decent defaults if present
    default_text = 0
    for i, c in enumerate(cols):
        if c.lower() in {"review","text","comment","feedback"}:
            default_text = i
            break
    default_label = 0
    for i, c in enumerate(cols):
        if c.lower() in {"liked","label","sentiment","target","y"}:
            default_label = i + 1  # because of "<None>" at index 0
            break

    st.session_state.text_col = st.selectbox("Select text column", options=cols, index=default_text)
    st.session_state.label_col = st.selectbox("Select label column (optional; binary 0/1)", options=["<None>"] + cols, index=default_label)

    apply_clean = st.checkbox("Apply basic cleaning (lowercase, remove URLs & punctuation)", value=True)
    if st.button("Save selection"):
        if apply_clean:
            st.session_state.df[st.session_state.text_col] = (
                st.session_state.df[st.session_state.text_col].astype(str).apply(basic_clean)
            )
        st.success("Selections saved. Proceed to **Data Visualization**.")
