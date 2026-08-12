"""RAG: top-k sections with exact page citations.

Every returned snippet carries (title, page) so the diagnostic prompt can cite exact
pages and the scorer can validate retrieval_citation_support.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hybrid_search import HybridRetriever


@dataclass(frozen=True)
class CitedSnippet:
    text: str
    title: str
    page: int

    def as_line(self) -> str:
        return f"[{self.title} p.{self.page}] {self.text}"


class RagPipeline:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 5,
                 platform: str | None = None) -> list[CitedSnippet]:
        chunks = self.retriever.query(query, top_k=top_k, platform=platform)
        return [CitedSnippet(text=c.text, title=c.title, page=c.page) for c in chunks]

    def lines(self, query: str, top_k: int = 5,
              platform: str | None = None) -> list[str]:
        return [s.as_line() for s in self.retrieve(query, top_k, platform)]