"""Structured model of a test-harness run log.

``TestLogReport`` is the single shape every log source produces: the pipeline
consumes this shape only, so a local file and a future website fetcher feed the
same evidence path. Renderers cap their output so a huge log never blows the
prompt budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Max failure-context lines rendered per failure in the prompt.
_CONTEXT_RENDER_CAP = 12


@dataclass
class TestLogFailure:
    """One harness failure, deduped across repeated runs in the same log."""

    code: str = ""                # e.g. "P02002001"
    description: str = ""         # e.g. "PCIe Test Fail"
    test_name: str | None = None  # e.g. "pcie_cmp_chk"
    context: list[str] = field(default_factory=list)  # surrounding ERROR lines (redacted)
    occurrences: int = 1
    first_seen: str | None = None  # timestamp of the first occurrence, if any

    @property
    def signature(self) -> str:
        """Compact identity used by prompts, retrieval, and the case store."""
        if self.code and self.description:
            return f"{self.code}@{self.description}"
        return self.description or self.code or "harness failure"

    def render(self) -> str:
        return self.signature


@dataclass
class TestLogReport:
    """A parsed test-harness run log: metadata + deduped failures.

    ``raw_excerpt`` is the fallback when no structured failure could be parsed
    (so a foreign/one-off log format still feeds the agent as evidence).
    """

    source: str
    model: str | None = None
    serial: str | None = None
    station: str | None = None
    test_stage: str | None = None
    build_site: str | None = None
    program_version: str | None = None
    failures: list[TestLogFailure] = field(default_factory=list)
    raw_excerpt: str | None = None

    def summary_lines(self) -> list[str]:
        """Prompt-ready evidence lines (capped) for the log evidence section."""
        lines = [
            f"source={self.source}",
            f"model={self.model or 'unknown'}",
            f"serial={self.serial or 'unknown'}",
            f"station={self.station or 'unknown'}",
            f"stage={self.test_stage or 'unknown'}",
            f"build_site={self.build_site or 'unknown'}",
        ]
        if self.failures:
            lines.append(f"failures ({len(self.failures)}):")
            for failure in self.failures:
                lines.append(f"- {failure.render()}")
                if failure.test_name:
                    lines.append(f"    test={failure.test_name}")
                if failure.occurrences > 1:
                    lines.append(f"    occurrences={failure.occurrences}")
                for line in failure.context[:_CONTEXT_RENDER_CAP]:
                    lines.append(f"    {line}")
        else:
            lines.append("(no structured failure parsed; raw excerpt below)")
            if self.raw_excerpt:
                lines.append(self.raw_excerpt)
        return lines

    def rag_queries(self, limit: int = 3) -> list[str]:
        """Doc-retrieval queries derived from the failures (capped)."""
        queries: list[str] = []
        for failure in self.failures:
            query = failure.description
            if failure.code:
                query = f"{query} {failure.code}"
            if query and query not in queries:
                queries.append(query)
        for failure in self.failures:
            if failure.test_name and failure.test_name not in queries:
                queries.append(f"{failure.test_name} harness test failure")
        return queries[:limit]

    def case_terms(self) -> list[str]:
        """Failure identity terms recorded on / matched against the case store.

        A future run whose log shares any of these (e.g. the same error code)
        surfaces this verified case in its Prior Verified Cases section.
        """
        terms: list[str] = []
        for failure in self.failures:
            signature = failure.signature
            if signature not in terms:
                terms.append(signature)
            if failure.test_name and failure.test_name not in terms:
                terms.append(failure.test_name)
        return terms