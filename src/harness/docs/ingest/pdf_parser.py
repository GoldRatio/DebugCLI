"""Layout-aware PDF -> text + page metadata.

Uses pymupdf (importlib-resource) to extract per-page text with the source page.
For scanned-only PDFs an OCR hook can be supplied; text is stored alongside the
page ref so the RAG layer can always cite exact pages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

try:  # optional dependency (`pip install harness[docs]`)
    import pymupdf
except Exception:  # noqa: BLE001 - optional dependency, degrade gracefully
    pymupdf = None


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    ocr: bool = False


class PdfParser:
    def __init__(self, ocr: Callable[[bytes], str] | None = None) -> None:
        """``ocr`` is an optional callable mapping a page's raw image bytes to text."""
        self.ocr = ocr

    def parse(self, path: str | Path) -> list[PageText]:
        if pymupdf is None:
            raise RuntimeError("pymupdf not installed; install harness[docs]")
        doc = pymupdf.open(str(path))
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            ocr = not text and bool(page.get_images()) and self.ocr is not None
            if ocr:
                pix = page.get_pixmap()
                text = self.ocr(pix.tobytes("png")).strip()
            pages.append(PageText(page=i, text=text, ocr=ocr))
        doc.close()
        return pages