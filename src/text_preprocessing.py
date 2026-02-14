import re
from typing import Dict, List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def _ensure_stopwords() -> None:
    """Ensure NLTK stopwords are available. Download if missing."""
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_non_letters: bool = True,
    remove_stopwords: bool = True,
    keep_not: bool = True,
    stemming: bool = True,
) -> str:
    """
    Clean a single text review using options similar to your notebook code.
    Returns a cleaned string.
    """
    _ensure_stopwords()

    if text is None:
        return ""

    review = str(text)

    if remove_non_letters:
        review = re.sub(r"[^a-zA-Z]", " ", review)

    if lowercase:
        review = review.lower()

    tokens = review.split()

    if remove_stopwords:
        all_stopwords = stopwords.words("english")
        if keep_not and "not" in all_stopwords:
            all_stopwords.remove("not")
        sw_set = set(all_stopwords)
        tokens = [t for t in tokens if t not in sw_set]

    if stemming:
        ps = PorterStemmer()
        tokens = [ps.stem(t) for t in tokens]

    return " ".join(tokens)


def build_corpus(
    texts: List[str],
    opts: Dict,
    max_rows: Optional[int] = None,
) -> List[str]:
    """
    Build a corpus (list of cleaned strings) from a list/series of texts.
    """
    if max_rows is None:
        max_rows = len(texts)

    corpus = []
    limit = min(int(max_rows), len(texts))

    for i in range(limit):
        corpus.append(
            clean_text(
                texts[i],
                lowercase=opts.get("lowercase", True),
                remove_non_letters=opts.get("remove_non_letters", True),
                remove_stopwords=opts.get("remove_stopwords", True),
                keep_not=opts.get("keep_not", True),
                stemming=opts.get("stemming", True),
            )
        )

    return corpus
