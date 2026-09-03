"""Layout-aware PDF -> text + page metadata.

Uses pymupdf (importlib-resource) to extract per-page text with the source page.
Image content is not lost: an injectable ``ocr`` callable (PNG bytes -> text,
typically :func:`harness.docs.ingest.vision.vision_captioner`) is applied to
full-page renders of scanned pages AND to meaningful embedded images on text
pages (diagrams, tables, callouts). Caption text is merged into the page text
so chunks stay page-citeable; logos/icons and failed captions degrade silently
to text-only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

try:  # optional dependency (`pip install harness[docs]`)
    import pymupdf
except Exception:  # noqa: BLE001 - optional dependency, degrade gracefully
    pymupdf = None

# Images below either threshold are logos/icons/decorations, not content worth
# a vision round-trip.
MIN_IMAGE_BYTES = 2048
MIN_IMAGE_DIM = 64


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    ocr: bool = False


class PdfParser:
    def __init__(self, ocr: Callable[[bytes], str] | None = None) -> None:
        """``ocr`` is an optional callable mapping a page image's PNG bytes to text."""
        self.ocr = ocr

    def parse(self, path: str | Path) -> list[PageText]:
        if pymupdf is None:
            raise RuntimeError("pymupdf not installed; install harness[docs]")
        doc = pymupdf.open(str(path))
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            ocr = False
            if self.ocr is not None:
                if not text:
                    # Scanned page: no text layer, caption the full-page render.
                    # (Blank pages with no images at all skip the vision call.)
                    if page.get_images():
                        caption = self._caption(page.get_pixmap().tobytes("png"))
                        if caption:
                            text, ocr = caption, True
                else:
                    # Text page with embedded images: caption the meaningful ones
                    # so diagram labels/tables/callouts are searchable too.
                    captions = [c for c in (self._embedded_caption(doc, img)
                                            for img in page.get_images(full=True)) if c]
                    if captions:
                        text = "\n".join([text, *captions]).strip()
                        ocr = True
            pages.append(PageText(page=i, text=text, ocr=ocr))
        doc.close()
        return pages

    def _embedded_caption(self, doc, img: tuple) -> str | None:
        """Caption one embedded image; None when skipped (tiny/broken/failed)."""
        xref = img[0]
        if not xref:  # inline image, no xref: covered by the full-page path
            return None
        try:
            info = doc.extract_image(xref)
            if len(info["image"]) < MIN_IMAGE_BYTES:
                return None
            if max(info.get("width") or 0, info.get("height") or 0) < MIN_IMAGE_DIM:
                return None
            # Re-encode via Pixmap so the hook always receives PNG bytes,
            # whatever the embedded format (jpeg/jpx/jbig2).
            pix = pymupdf.Pixmap(doc, xref)
            if pix.colorspace is None:  # stencil/mask: not captionable content
                return None
            return self._caption(pix.tobytes("png"))
        except Exception:  # noqa: BLE001 - one broken image never kills a page
            return None

    def _caption(self, png: bytes) -> str:
        try:
            return self.ocr(png).strip()
        except Exception:  # noqa: BLE001 - captioner failure degrades to text-only
            return ""
