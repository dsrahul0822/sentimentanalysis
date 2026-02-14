import streamlit as st

def init_state():
    defaults = {
        "df": None,
        "text_col": "Review",
        "label_col": "Liked",
        "cleaning_opts": {
            "lowercase": True,
            "remove_non_letters": True,
            "remove_stopwords": True,
            "keep_not": True,
            "stemming": True,
        },
        "max_rows": 1000,
        "corpus": None,
        "vectorizer_kind": "TF-IDF",
        "max_features": 1500,
        "test_size": 0.2,
        "random_state": 42,
        "vectorizer": None,
        "model": None,
        "X": None,
        "y": None,
        "x_train": None,
        "x_test": None,
        "y_train": None,
        "y_test": None,
        "y_pred": None,
        "accuracy": None,
        "conf_matrix": None,
        "trained": False,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
