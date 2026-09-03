"""PDF image ingest: scanned pages + embedded diagrams -> searchable text.

Builds real single-page PDFs with pymupdf (installed for the docs extra) and
exercises the ``PdfParser`` caption hook paths: full-page render for scanned
pages, embedded-image extraction with logo/icon filtering, caption-failure
degradation to text-only, plus the ``VisionCaptioner`` wiring (env selection,
failure latch) and an end-to-end ingest through ``DocLibrary``.
"""

import random

import pytest

from harness.docs.ingest.pdf_parser import PdfParser

pymupdf = pytest.importorskip("pymupdf")


class _Captioner:
    """Fake vision captioner: canned text, records calls, counts failures."""

    def __init__(self, text="DIAGRAM: memory channel topology", fail=False):
        self.text = text
        self.fail = fail
        self.calls = []
        self.failures = 0

    def __call__(self, png: bytes) -> str:
        if self.fail:
            self.failures += 1
            raise RuntimeError("endpoint down")
        self.calls.append(png)
        return self.text


def _noise_png(width=200, height=100) -> bytes:
    """PNG with enough entropy that the embedded stream exceeds MIN_IMAGE_BYTES
    (seeded random samples compress about as poorly as real diagrams)."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height))
    pix.samples_mv[:] = random.Random(42).randbytes(len(pix.samples_mv))
    return pix.tobytes("png")


def _solid_png(width=100, height=100) -> bytes:
    """Solid-color PNG: large dimensions but tiny embedded bytes (logo-like)."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height))
    pix.clear_with(90)
    return pix.tobytes("png")


def _write_pdf(path, *, with_text, images):
    doc = pymupdf.open()
    page = doc.new_page()
    if with_text:
        page.insert_text((72, 72), "DIMM population rules for the compute tray")
    for png in images:
        page.insert_image(pymupdf.Rect(72, 100, 372, 250), stream=png)
    doc.save(str(path))
    doc.close()
    return path


# ---- PdfParser ----

def test_scanned_page_captioned(tmp_path):
    pdf = _write_pdf(tmp_path / "scan.pdf", with_text=False, images=[_noise_png()])
    cap = _Captioner("PSU LED map: amber = FRU fault")
    pages = PdfParser(ocr=cap).parse(pdf)
    assert len(pages) == 1
    assert pages[0].text == "PSU LED map: amber = FRU fault"
    assert pages[0].ocr is True
    assert len(cap.calls) == 1
    assert cap.calls[0].startswith(b"\x89PNG")  # hook contract: PNG bytes


def test_text_page_diagram_caption_appended(tmp_path):
    pdf = _write_pdf(tmp_path / "arch.pdf", with_text=True, images=[_noise_png()])
    cap = _Captioner("DIAGRAM: DIMM slot map")
    pages = PdfParser(ocr=cap).parse(pdf)
    assert pages[0].text.startswith("DIMM population rules")
    assert "DIAGRAM: DIMM slot map" in pages[0].text  # caption merged, not replacing
    assert pages[0].ocr is True


def test_logo_and_icon_images_skipped(tmp_path):
    dim_small = _noise_png(40, 40)      # >2KB but <64px -> dimension filter
    bytes_small = _solid_png(100, 100)  # >=64px but <2KB -> bytes filter
    pdf = _write_pdf(tmp_path / "logo.pdf", with_text=True,
                     images=[dim_small, bytes_small])
    cap = _Captioner()
    pages = PdfParser(ocr=cap).parse(pdf)
    assert cap.calls == []  # neither image worth a vision round-trip
    assert pages[0].text == "DIMM population rules for the compute tray"
    assert pages[0].ocr is False


def test_caption_failure_degrades_to_text_only(tmp_path):
    pdf = _write_pdf(tmp_path / "arch.pdf", with_text=True, images=[_noise_png()])
    cap = _Captioner(fail=True)
    pages = PdfParser(ocr=cap).parse(pdf)
    assert pages[0].text == "DIMM population rules for the compute tray"
    assert pages[0].ocr is False


def test_blank_page_never_calls_captioner(tmp_path):
    doc = pymupdf.open()
    doc.new_page()  # no text, no images
    doc.save(str(tmp_path / "blank.pdf"))
    doc.close()
    cap = _Captioner()
    pages = PdfParser(ocr=cap).parse(tmp_path / "blank.pdf")
    assert cap.calls == []
    assert pages[0].text == ""


def test_without_ocr_hook_text_only(tmp_path):
    pdf = _write_pdf(tmp_path / "scan.pdf", with_text=False, images=[_noise_png()])
    pages = PdfParser().parse(pdf)
    assert pages[0].text == ""
    assert pages[0].ocr is False


# ---- VisionCaptioner ----

def test_vision_captioner_none_when_disabled(monkeypatch):
    from harness.docs.ingest import vision
    monkeypatch.setenv("HARNESS_LLM_PROVIDER", "none")
    assert vision.vision_captioner() is None


def test_vision_captioner_stub_end_to_end(monkeypatch):
    from harness.docs.ingest import vision
    monkeypatch.setenv("HARNESS_LLM_PROVIDER", "stub")
    cap = vision.vision_captioner()
    assert isinstance(cap, vision.VisionCaptioner)
    assert "(stub caption)" in cap(b"\x89PNG")
    assert cap.failures == 0


def test_vision_captioner_latches_after_failure():
    from harness.diagnosis.llm import LLMError
    from harness.docs.ingest import vision

    class _Broken:
        def __init__(self):
            self.attempts = 0

        def caption_image(self, png, prompt=None):
            self.attempts += 1
            raise LLMError("endpoint down")

    broken = _Broken()
    cap = vision.VisionCaptioner(broken)
    with pytest.raises(LLMError):
        cap(b"png1")
    with pytest.raises(LLMError, match="disabled"):
        cap(b"png2")
    assert broken.attempts == 1  # latched: one attempt, not one per image
    assert cap.failures == 1


# ---- DocLibrary end-to-end ----

def test_library_end_to_end_vision_ingest(tmp_path):
    """A scanned PDF added with a working captioner is retrievable by its caption."""
    from harness.docs.ingest.library import DocLibrary
    pdf = _write_pdf(tmp_path / "scan.pdf", with_text=False, images=[_noise_png()])
    lib = DocLibrary(tmp_path / "lib", ocr=_Captioner("PSU LED map amber FRU fault"))
    status = lib.add([pdf])
    assert any("indexed scan.pdf" in s for s in status)
    rag = lib.build_retriever()
    assert rag is not None
    lines = rag.lines("PSU LED fault", top_k=1)
    assert "[scan.pdf p.1]" in lines[0]


def test_library_warns_when_captioner_fails(tmp_path):
    """Captioning failure degrades to text-only with a warn line, never a failure."""
    from harness.docs.ingest.library import DocLibrary
    pdf = _write_pdf(tmp_path / "arch.pdf", with_text=True, images=[_noise_png()])
    lib = DocLibrary(tmp_path / "lib", ocr=_Captioner(fail=True))
    status = lib.add([pdf])
    assert any("warn: image captioning failed" in s for s in status)
    entry = lib.entries()[0]
    assert entry.error is None
    assert entry.chunks >= 1  # text page still indexed
