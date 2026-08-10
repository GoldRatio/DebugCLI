"""Chunk documents with overlap, and embed into a vector store.

500-token chunks / 100-token overlap (spec). Metadata carries title + page so the
RAG layer can cite exact pages. The vector backend is abstracted so it can run
on-prem (proprietary-safe) without any cloud dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .pdf_parser import PageText


@dataclass
class Chunk:
    text: str
    title: str
    page: int
    index: int
    embedding: list[float] | None = field(default=None)

    @property
    def meta(self) -> dict:
        return {"title": self.title, "page": self.page, "idx": self.index}


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]: ...
    @abstractmethod
    def decode(self, tokens: list[int]) -> str: ...


class CharTokenizer(Tokenizer):
    """Deterministic, invertible fallback tokenizer (whitespace words).

    Maps each word to a stable per-instance id so ``decode(encode(text))`` round-trips
    the original text exactly. Swap for tiktoken/dense embeddings without changing
    the chunker contract.
    """

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._words: list[str] = []

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            idx = self._ids.get(word)
            if idx is None:
                idx = len(self._words)
                self._ids[word] = idx
                self._words.append(word)
            ids.append(idx)
        return ids

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self._words[idx] for idx in tokens)


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunk: Chunk) -> None: ...
    @abstractmethod
    def search(self, query: str, top_k: int) -> list[Chunk]: ...


class Chunker:
    def __init__(self, tokenizer: Tokenizer, chunk_tokens: int = 500, overlap: int = 100) -> None:
        self.tokenizer = tokenizer
        self.chunk_tokens = chunk_tokens
        self.overlap = overlap

    def chunk_pages(self, pages: list[PageText], title: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        idx = 0
        for page in pages:
            tokens = self.tokenizer.encode(page.text)
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_tokens, len(tokens))
                text = self.tokenizer.decode(tokens[start:end])
                chunks.append(Chunk(text=text, title=title, page=page.page, index=idx))
                idx += 1
                if end >= len(tokens):
                    break
                start = end - self.overlap
        return chunks