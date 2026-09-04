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
# Caption payload discipline: base64 image parts above ~1MB make vision
# endpoints choke (an oversized 2880x1620 page render took down a rack-served
# vLLM mid-batch). JPEGs pass through as-is when already small enough.
MAX_CAPTION_BYTES = 1_000_000


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    ocr: bool = False


class PdfParser:
    def __init__(self, ocr: Callable[[bytes], str] | None = None,
                 progress: Callable[[str], None] | None = None) -> None:
        """``ocr`` is an optional callable mapping page-image bytes (PNG, or
        native JPEG for small embedded JPEGs) to text; ``progress`` receives
        one line per caption attempt (vision batches over a tunnel can take
        minutes -- operators need a pulse)."""
        self.ocr = ocr
        self.progress = progress

    def parse(self, path: str | Path) -> list[PageText]:
        if pymupdf is None:
            raise RuntimeError("pymupdf not installed; install harness[docs]")
        doc = pymupdf.open(str(path))
        total = doc.page_count
        doc_captions: dict[int, str] = {}  # xref -> caption: header logos recur
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            ocr = False
            if self.ocr is not None:
                if not text:
                    # Scanned page: no text layer, caption the full-page render.
                    # (Blank pages with no images at all skip the vision call.)
                    if page.get_images():
                        if self.progress:
                            self.progress(f"captioning {Path(path).name} "
                                          f"page {i}/{total} (scanned render)")
                        caption = self._caption(page.get_pixmap().tobytes("png"))
                        if caption:
                            text, ocr = caption, True
                else:
                    # Text page with embedded images: caption the meaningful ones
                    # so diagram labels/tables/callouts are searchable too.
                    new_xrefs = [img[0] for img in page.get_images(full=True)
                                 if img[0] and img[0] not in doc_captions]
                    if new_xrefs and self.progress:
                        self.progress(f"captioning {Path(path).name} "
                                      f"page {i}/{total} ({len(new_xrefs)} new image(s))")
                    captions = [c for c in (self._embedded_caption(doc, img, doc_captions)
                                            for img in page.get_images(full=True)) if c]
                    if captions:
                        text = "\n".join([text, *captions]).strip()
                        ocr = True
            pages.append(PageText(page=i, text=text, ocr=ocr))
        doc.close()
        return pages

    def _embedded_caption(self, doc, img: tuple, cache: dict[int, str]) -> str | None:
        """Caption one embedded image; None when skipped (tiny/broken/failed).

        Captions are cached per document xref -- the same header/logo image
        recurs on many pages and must cost exactly one vision round-trip."""
        xref = img[0]
        if not xref:  # inline image, no xref: covered by the full-page path
            return None
        if xref in cache:
            return cache[xref] or None
        try:
            info = doc.extract_image(xref)
            if len(info["image"]) < MIN_IMAGE_BYTES:
                cache[xref] = ""
                return None
            if max(info.get("width") or 0, info.get("height") or 0) < MIN_IMAGE_DIM:
                cache[xref] = ""
                return None
            # Small JPEGs pass through untouched (native resolution, fraction
            # of a re-encoded PNG). Everything else renders via Pixmap so the
            # hook always receives PNG bytes, then shrinks until the payload
            # fits the caption cap.
            if info.get("ext") in ("jpeg", "jpg") and len(info["image"]) <= MAX_CAPTION_BYTES:
                caption = self._caption_jpeg(info["image"])
                cache[xref] = caption
                return caption or None
            pix = pymupdf.Pixmap(doc, xref)
            if pix.colorspace is None:  # stencil/mask: not captionable content
                cache[xref] = ""
                return None
            png = pix.tobytes("png")
            shrinks = 0
            while len(png) > MAX_CAPTION_BYTES and shrinks < 4:
                pix.shrink(1)  # halves both dimensions
                png = pix.tobytes("png")
                shrinks += 1
            caption = self._caption(png)
            cache[xref] = caption
            return caption or None
        except Exception:  # noqa: BLE001 - one broken image never kills a page
            cache[xref] = ""
            return None

    def _caption_jpeg(self, jpeg: bytes) -> str:
        try:
            return self.ocr(jpeg).strip()
        except Exception:  # noqa: BLE001 - captioner failure degrades to text-only
            return ""

    def _caption(self, png: bytes) -> str:
        try:
            return self.ocr(png).strip()
        except Exception:  # noqa: BLE001 - captioner failure degrades to text-only
            return ""


def render_page_png(path: str | Path, page: int, *, max_px: int = 1200,
                    max_bytes: int = 1_500_000) -> bytes | None:
    """Render one page of a PDF as a PNG for query-time vision RAG.

    Scales the page to ``max_px`` on the long edge, shrinking up to twice
    more while the PNG exceeds ``max_bytes`` (payload discipline for
    multimodal turns). Dense raster content may still exceed the cap -- the
    smallest render is returned regardless (an oversized page image beats no
    image). None when pymupdf is missing, the file/page is invalid, or
    rendering fails -- callers treat that as text-only.
    """
    if pymupdf is None:
        return None
    try:
        doc = pymupdf.open(str(path))
        try:
            if not 1 <= page <= doc.page_count:
                return None
            rect = doc[page - 1].rect
            zoom = min(max_px / max(rect.width, rect.height), 2.0)
            png = b""
            for _ in range(3):
                pix = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
                png = pix.tobytes("png")
                if len(png) <= max_bytes:
                    return png
                zoom *= 0.7
            return png
        finally:
            doc.close()
    except Exception:  # noqa: BLE001 - rendering failure degrades to text-only
        return None
