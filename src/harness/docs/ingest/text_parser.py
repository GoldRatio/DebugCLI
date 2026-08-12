"""Text-format ingestion: Markdown, plain text, CSV -> text + section metadata.

Lets the doc library accept more than PDFs. Markdown is split on headings so
each section becomes a "page" and chunks keep an accurate citeable section
number; txt/csv are treated as single pages. CSV rows are rendered tab-joined
so cell values are directly searchable by the hybrid retriever.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .pdf_parser import PageText

HEADING_RE = re.compile(r"^#{1,3}\s+.+$", re.MULTILINE)
FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_front_matter(text: str) -> str:
    return FRONT_MATTER_RE.sub("", text, count=1)


@dataclass(frozen=True)
class _MarkdownParser:
    def parse(self, path: str | Path) -> list[PageText]:
        text = _strip_front_matter(Path(path).read_text(encoding="utf-8", errors="replace"))
        headings = list(HEADING_RE.finditer(text))
        if not headings:
            return [PageText(page=1, text=text.strip())]
        pages = []
        preamble = text[: headings[0].start()].strip()
        if preamble:
            pages.append(PageText(page=1, text=preamble))
        for i, m in enumerate(headings, start=1):
            end = headings[i].start() if i < len(headings) else len(text)
            section = text[m.start(): end].strip()
            if section:
                pages.append(PageText(page=i, text=section))
        return pages or [PageText(page=1, text=text.strip())]


@dataclass(frozen=True)
class _PlainTextParser:
    def parse(self, path: str | Path) -> list[PageText]:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return [PageText(page=1, text=text.strip())]


@dataclass(frozen=True)
class _CsvParser:
    def parse(self, path: str | Path) -> list[PageText]:
        with Path(path).open("r", newline="", encoding="utf-8", errors="replace") as fh:
            rows = list(csv.reader(fh))
        lines = ["\t".join(cell.strip() for cell in row) for row in rows if any(row)]
        text = "\n".join(lines).strip()
        return [PageText(page=1, text=text)] if text else []
