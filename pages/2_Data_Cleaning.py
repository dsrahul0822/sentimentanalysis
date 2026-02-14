import streamlit as st
from src.app_state import init_state
from src.text_preprocessing import build_corpus

init_state()

st.title("🧹 Data Cleaning Options")

if st.session_state.df is None:
    st.warning("No dataset found. Please upload data in **📤 Data Upload** first.")
    st.stop()

df = st.session_state.df

# Ensure correct text_col exists
if st.session_state.text_col not in df.columns:
    st.session_state.text_col = "Review" if "Review" in df.columns else df.columns[0]

text_col = st.session_state.text_col

st.subheader("Choose cleaning steps")

c1, c2, c3 = st.columns(3)

with c1:
    lowercase = st.checkbox("Lowercase", value=st.session_state.cleaning_opts["lowercase"])
    remove_non_letters = st.checkbox("Remove non-letters (keep A-Z only)", value=st.session_state.cleaning_opts["remove_non_letters"])

with c2:
    remove_stopwords = st.checkbox("Remove stopwords", value=st.session_state.cleaning_opts["remove_stopwords"])
    keep_not = st.checkbox('Keep the word "not"', value=st.session_state.cleaning_opts["keep_not"])

with c3:
    stemming = st.checkbox("Stemming (PorterStemmer)", value=st.session_state.cleaning_opts["stemming"])

max_rows = st.number_input(
    "Max rows to process (demo)",
    min_value=100,
    max_value=int(len(df)),
    value=min(int(st.session_state.max_rows), int(len(df))),
    step=100,
)

st.session_state.cleaning_opts = {
    "lowercase": lowercase,
    "remove_non_letters": remove_non_letters,
    "remove_stopwords": remove_stopwords,
    "keep_not": keep_not,
    "stemming": stemming,
}
st.session_state.max_rows = int(max_rows)

st.divider()
st.subheader("Preview (Before → After)")

max_idx = min(int(st.session_state.max_rows) - 1, len(df) - 1)
idx = st.slider("Pick row index", 0, max_idx, 0)

st.write("**Before**")
st.write(df[text_col].iloc[idx])

if st.button("✅ Build / Rebuild Corpus"):
    texts = df[text_col].astype(str).tolist()
    st.session_state.corpus = build_corpus(texts, st.session_state.cleaning_opts, st.session_state.max_rows)
    st.success(f"Corpus built with {len(st.session_state.corpus)} reviews.")
    st.info("Next: go to **🏗️ Model Building** page.")

if st.session_state.corpus is not None:
    st.write("**After**")
    st.write(st.session_state.corpus[idx])
