"""Hybrid retrieval: BM25 (lexical) fused with semantic (dense) similarity.

Fusion uses reciprocal-rank fusion of both ranked lists. Results carry title + page
so callers and the prompt always have exact citations.
"""

from __future__ import annotations

import math
from collections import Counter

from ..ingest.chunk import Chunk


def bm25_score(query_terms: list[str], chunk: Chunk, corpus_stats) -> float:
    """Simplified BM25. ``corpus_stats`` = (avg_len, doc_freq[C])"""
    avg_len, doc_freq = corpus_stats
    tf = Counter(chunk.text.split())
    k1, b = 1.5, 0.75
    doc_len = len(chunk.text.split())
    score = 0.0
    N = max(len(doc_freq), 1)
    for term in query_terms:
        if term not in tf:
            continue
        idf = math.log(1 + (N - doc_freq.get(term, 0) + 0.5) / (doc_freq.get(term, 0) + 0.5))
        denom = tf[term] + k1 * (1 - b + b * doc_len / avg_len)
        score += idf * (tf[term] * (k1 + 1)) / denom
    return score


def _corpus_stats(chunks: list[Chunk]):
    lengths = [len(c.text.split()) for c in chunks]
    avg = sum(lengths) / len(lengths) if lengths else 1.0
    doc_freq: Counter = Counter()
    for c in chunks:
        for term in set(c.text.split()):
            doc_freq[term] += 1
    return avg, doc_freq


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], semantic_weight: float = 0.5, k: int = 60) -> None:
        self.chunks = chunks
        self.semantic_weight = semantic_weight
        self.k = k
        self._stats = _corpus_stats(chunks)

    def query(self, question: str, top_k: int = 5) -> list[Chunk]:
        terms = [t for t in question.lower().split() if len(t) > 2]
        if not terms:
            return self.chunks[:top_k]

        # Lexical ranking.
        lexical = sorted(self.chunks, key=lambda c: bm25_score(terms, c, self._stats), reverse=True)
        lexical_rank = {id(c): i for i, c in enumerate(lexical)}

        # Semantic ranking (embedding cosine, when embeddings exist).
        dense_rank: dict[int, int] = {}
        with_embeds = [c for c in self.chunks if c.embedding]
        if with_embeds:
            qtok = _query_embedding(with_embeds)
            dense = sorted(with_embeds, key=lambda c: cosine(c.embedding, qtok), reverse=True)
            dense_rank = {id(c): i for i, c in enumerate(dense)}

        scored = []
        for c in self.chunks:
            lr = lexical_rank.get(id(c), len(self.chunks))
            dr = dense_rank.get(id(c), len(self.chunks))
            rrf = (1.0 / (self.k + lr)) + self.semantic_weight * (1.0 / (self.k + dr))
            scored.append((rrf, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored[:top_k]]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _query_embedding(_chunks: list[Chunk]) -> list[float]:
    """Query embedding hook -- override with a real dense encoder. Returns zero vector."""
    dim = len(_chunks[0].embedding) if _chunks and _chunks[0].embedding else 1
    return [0.0] * dim