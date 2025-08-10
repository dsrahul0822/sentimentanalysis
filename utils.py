import re
import string
import pandas as pd

def read_any(filepath_or_buffer, filename: str):
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(filepath_or_buffer)
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(filepath_or_buffer, sep="\t")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(filepath_or_buffer)
    # fallback: try csv, then excel
    try:
        return pd.read_csv(filepath_or_buffer)
    except Exception:
        return pd.read_excel(filepath_or_buffer)

def basic_clean(text: str):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
