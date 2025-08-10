import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import re

st.title("📊 Data Visualization")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Please load data first from the **Data Load** page.")
    st.stop()

df = st.session_state.df
text_col = st.session_state.text_col

st.subheader("WordCloud")
all_text = " ".join(df[text_col].astype(str).tolist())
stopwords = set(STOPWORDS)
extra_stops = st.text_area("Add extra stopwords (comma-separated)", value="food,restaurant,place").strip()
if extra_stops:
    for w in [w.strip().lower() for w in extra_stops.split(",") if w.strip()]:
        stopwords.add(w)

wc = WordCloud(width=1000, height=500, background_color="white", stopwords=stopwords).generate(all_text)
fig_wc, ax_wc = plt.subplots(figsize=(10,5))
ax_wc.imshow(wc, interpolation="bilinear")
ax_wc.axis("off")
st.pyplot(fig_wc)

st.subheader("Top-N Word Frequency")
n = st.slider("Top N words", 10, 50, 20)

# Very basic tokenization
tokens = re.findall(r"[a-zA-Z']+", all_text.lower())
tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
freq = Counter(tokens)
most_common = freq.most_common(n)
if most_common:
    words, counts = zip(*most_common)
else:
    words, counts = [], []

fig_bar, ax_bar = plt.subplots(figsize=(10,5))
ax_bar.bar(words, counts)
ax_bar.set_xticklabels(words, rotation=45, ha="right")
ax_bar.set_ylabel("Frequency")
ax_bar.set_title("Top Word Frequencies")
st.pyplot(fig_bar)
