from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_WORD = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text or "")]


def document_frequencies(documents: Iterable[str]) -> dict[str, int]:
    df: dict[str, int] = {}
    for doc in documents:
        seen = set(tokenize(doc))
        for t in seen:
            df[t] = df.get(t, 0) + 1
    return df


def bm25_score(query: str, doc: str, avgdl: float, doc_freq: dict[str, int], ndocs: int, k1: float = 1.5, b: float = 0.75) -> float:
    qt = tokenize(query)
    if not qt:
        return 0.0
    dtoks = tokenize(doc)
    dl = len(dtoks) or 1
    tf = Counter(dtoks)
    score = 0.0
    for term in qt:
        if term not in tf:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log(1.0 + (ndocs - df + 0.5) / (df + 0.5))
        f = tf[term]
        denom = f + k1 * (1 - b + b * (dl / avgdl))
        score += idf * (f * (k1 + 1)) / denom
    return score


def rank_documents(query: str, documents: list[str]) -> list[float]:
    """Return BM25 scores aligned with `documents` indices."""
    ndocs = len(documents)
    if ndocs == 0:
        return []
    doc_freq = document_frequencies(documents)
    lengths = [max(1, len(tokenize(d))) for d in documents]
    avgdl = sum(lengths) / ndocs
    return [bm25_score(query, documents[i], avgdl, doc_freq, ndocs) for i in range(ndocs)]
