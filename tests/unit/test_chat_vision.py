"""Vision RAG in chat: page-image attach, vision gating, text-only retry.

Covers the query-time half of image access (``_tool_docs`` queues rendered
page images for the next agent turn; ``build_messages`` renders them as
content parts; a vision-hostile endpoint triggers the one-shot text-only
retry), plus ingest-side caption stats and the ingest model resolution chain.
"""

import base64
import json
import random
import threading

import pytest

pymupdf = pytest.importorskip("pymupdf")

from harness.docs.ingest.library import DocLibrary
from harness.docs.ingest.pdf_parser import PageText
from harness.operator import chat_agent
from harness.operator.chat_agent import (
    build_messages,
    messages_have_images,
    strip_images,
)

# ---- fakes ----

class _FakeVisionLLM:
    """OpenAI-compatible-ish adapter whose caption behavior is configurable."""

    def __init__(self, caption="ok", fail=False):
        self.caption = caption
        self.fail = fail
        self.calls = []

    def caption_image(self, png, prompt=None):
        self.calls.append(png)
        if self.fail:
            raise RuntimeError("endpoint rejected image parts")
        return self.caption


class _FakeRouter:
    """Router LLM that refuses image-part messages (vision-hostile endpoint)."""

    def __init__(self):
        self.saw_parts = []

    def chat_json(self, messages):
        parts = [m for m in messages if isinstance(m.get("content"), list)]
        self.saw_parts.append(bool(parts))
        if parts:
            raise RuntimeError("HTTP 400: image parts unsupported")
        return {"tool": "none", "say": "text-only answer"}


class _FakeCatalog:
    def __init__(self, current):
        self.current = current


def _session(**kwargs):
    """Minimal session stand-in for the vision helpers."""
    from types import SimpleNamespace

    base: dict = {
        "llm": None,
        "vision_ok": None,
        "vision_disabled": False,
        "pending_images": [],
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _noise_png(width=200, height=100) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height))
    pix.samples_mv[:] = random.Random(42).randbytes(len(pix.samples_mv))
    return pix.tobytes("png")


def _scanned_pdf(path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PCIe topology: switch to CPU links")
    page.insert_image(pymupdf.Rect(72, 100, 372, 250), stream=_noise_png())
    doc.save(str(path))
    doc.close()


@pytest.fixture
def lib_dir(tmp_path):
    """A doc library with one captioned page, retrievable by caption words."""
    pdf = tmp_path / "arch.pdf"
    _scanned_pdf(pdf)
    lib = DocLibrary(tmp_path / "lib", ocr=lambda png: "PCIe switch topology diagram")
    status = lib.add([pdf])
    assert any("indexed arch.pdf" in s for s in status)
    return str(tmp_path / "lib")


# ---- build_messages / helpers ----

def test_build_messages_with_images_makes_content_parts():
    messages = build_messages(transcript=[], user_text="explain the topology",
                              images=[("arch.pdf p.1", "AAAA")])
    final = messages[-1]
    assert final["role"] == "user"
    content = final["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "arch.pdf p.1" in content[0]["text"]
    assert content[1] == {"type": "image_url",
                          "image_url": {"url": "data:image/png;base64,AAAA"}}
    assert messages_have_images(messages)
    # text-only strip: parts collapse to their text, images gone
    stripped = strip_images(messages)
    assert not messages_have_images(stripped)
    assert isinstance(stripped[-1]["content"], str)
    assert "explain the topology" in stripped[-1]["content"]


def test_build_messages_without_images_unchanged():
    messages = build_messages(transcript=[], user_text="hi")
    assert isinstance(messages[-1]["content"], str)
    assert not messages_have_images(messages)


# ---- vision gating ----

def test_gemini_assumed_vision_without_probe():
    from harness.diagnosis.llm import GeminiLLM

    session = _session(llm=GeminiLLM(api_key="k"))
    assert chat_agent  # import sanity
    from harness.operator.repl import _llm_supports_vision
    assert _llm_supports_vision(session) is True
    assert session.vision_ok is True


def test_stub_and_broken_endpoints_never_get_images():
    from harness.diagnosis.llm import StubLLM
    from harness.operator.repl import _llm_supports_vision

    session = _session(llm=StubLLM())
    assert _llm_supports_vision(session) is False

    broken = _session(llm=_FakeVisionLLM(fail=True))
    assert _llm_supports_vision(broken) is False
    assert broken.vision_ok is False

    working = _session(llm=_FakeVisionLLM())
    assert _llm_supports_vision(working) is True
    assert len(working.llm.calls) == 1  # 1x1 probe, exactly once (cached)


def test_vision_disabled_short_circuits_without_probe():
    from harness.operator.repl import _llm_supports_vision

    llm = _FakeVisionLLM()
    session = _session(llm=llm, vision_disabled=True)
    assert _llm_supports_vision(session) is False
    assert llm.calls == []


# ---- page-image queueing ----

def test_queue_page_images_renders_top_pages(lib_dir):
    from harness.operator.repl import _queue_page_images

    session = _session(llm=_FakeVisionLLM())
    note = _queue_page_images(session, "PCIe switch topology", lib_dir)
    assert note and "page image(s) attached" in note
    assert len(session.pending_images) == 1
    label, b64 = session.pending_images[0]
    assert label == "arch.pdf p.1"
    assert base64.b64decode(b64).startswith(b"\x89PNG")


def test_queue_page_images_skipped_for_non_vision(lib_dir):
    from harness.diagnosis.llm import StubLLM
    from harness.operator.repl import _queue_page_images

    session = _session(llm=StubLLM())
    assert _queue_page_images(session, "PCIe topology", lib_dir) is None
    assert session.pending_images == []


def test_queue_page_images_empty_library(tmp_path):
    from harness.operator.repl import _queue_page_images

    session = _session(llm=_FakeVisionLLM())
    assert _queue_page_images(session, "anything", str(tmp_path / "empty")) is None
    assert session.pending_images == []


def test_tool_docs_attaches_note(tmp_path, lib_dir, monkeypatch, capsys):
    from harness.operator.chat_agent import ChatTurn
    from harness.operator.repl import _tool_docs

    session = _session(llm=_FakeVisionLLM(), docs_lib=lib_dir, docs_dir=None,
                       pending_images=[])
    result = _tool_docs(session, ChatTurn(say="", tool="docs", query="PCIe topology"))
    assert "[arch.pdf p.1]" in result
    assert "page image(s) attached" in result
    assert session.pending_images


# ---- agent-loop text-only retry ----

def test_agent_retries_text_only_when_images_rejected():
    from harness.operator.repl import Session, _run_agent

    session = Session(
        mode="chat", inv_path="", inv=type("Inv", (), {"host_names": []})(),
        host=None, store=None, out_dir=None, session_dir=None,
        llm=None, router_llm=None, llm_mode="chat", docs_lib=None,
        docs_dir=None, parts_csv=None, secret_dir=None, console=False,
        max_tools=2)
    session.router_llm = _FakeRouter()
    session.pending_images = [("arch.pdf p.1", "AAAA")]

    events: list[str] = []
    result = _run_agent(session, "look at the diagram",
                        progress=events.append, cancel=threading.Event())
    assert session.vision_disabled is True
    assert session.vision_ok is False
    assert session.pending_images == []
    # first decide saw image parts and failed; retry was text-only
    assert session.router_llm.saw_parts == [True, False]
    assert result is None  # the say was streamed instead


# ---- ingest model resolution chain (cli._ingest_captioner) ----

def test_ingest_captioner_env_wins(monkeypatch):
    import harness.operator.cli as cli_mod

    monkeypatch.setenv("HARNESS_LLM_PROVIDER", "stub")
    cap = cli_mod._ingest_captioner()
    assert cap is not None  # env-selected captioner (stub backend)


def test_ingest_captioner_env_none_disables(monkeypatch):
    import harness.operator.cli as cli_mod

    monkeypatch.setenv("HARNESS_LLM_PROVIDER", "none")
    assert cli_mod._ingest_captioner() is None


def test_ingest_captioner_resolves_remembered_profile(monkeypatch):
    import harness.config.model_catalog as catalog_mod
    import harness.operator.cli as cli_mod
    from harness.docs.ingest import vision as vision_mod

    monkeypatch.delenv("HARNESS_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(cli_mod, "_discover_inventory", list)
    monkeypatch.setattr(cli_mod, "_prepare_llm_endpoint",
                        lambda ns, inv, store: setattr(ns, "llm_url", "http://fwd/v1/"))

    class _FakeModelCatalog:
        @staticmethod
        def load(inv=None):
            return _FakeCatalog(type("P", (), {
                "provider": "local", "model": "m", "needs_setup": False,
                "ident": "local/m",
                "build": lambda self, store: _FakeVisionLLM(),
            })())

    monkeypatch.setattr(catalog_mod, "ModelCatalog", _FakeModelCatalog)
    cap = cli_mod._ingest_captioner()
    assert isinstance(cap, vision_mod.VisionCaptioner)
    assert cap.llm.url == "http://fwd/v1"  # forward URL bound, trailing slash gone


def test_ingest_captioner_skips_stub_and_setup_profiles(monkeypatch):
    import harness.config.model_catalog as catalog_mod
    import harness.operator.cli as cli_mod

    monkeypatch.delenv("HARNESS_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(cli_mod, "_discover_inventory", list)

    def _catalog_with(current):
        class _C:
            @staticmethod
            def load(inv=None):
                return _FakeCatalog(current)
        return _C

    stub_profile = type("P", (), {"provider": "stub", "model": "stub",
                                  "needs_setup": False})()
    monkeypatch.setattr(catalog_mod, "ModelCatalog", _catalog_with(stub_profile))
    assert cli_mod._ingest_captioner() is None

    setup_profile = type("P", (), {"provider": "openai", "model": "harness-diag",
                                   "needs_setup": True})()
    monkeypatch.setattr(catalog_mod, "ModelCatalog", _catalog_with(setup_profile))
    assert cli_mod._ingest_captioner() is None


# ---- caption stats + page_image on the library ----

def test_manifest_captions_count_and_ls_display(tmp_path):
    pdf = tmp_path / "arch.pdf"
    _scanned_pdf(pdf)
    lib = DocLibrary(tmp_path / "lib", ocr=lambda png: "PCIe switch topology diagram")
    lib.add([pdf])
    entry = lib.entries()[0]
    assert entry.captions == 1
    assert json.loads((tmp_path / "lib" / "index.json").read_text())[
        "arch.pdf"]["captions"] == 1


def test_image_only_pdf_status_hint(tmp_path):
    """A PDF with no text and a dead captioner: indexed empty, hint suggests vision."""
    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(72, 100, 372, 250), stream=_noise_png())
    doc.save(str(tmp_path / "tree.pdf"))
    doc.close()

    def _broken(png):
        raise RuntimeError("endpoint down")

    lib = DocLibrary(tmp_path / "lib", ocr=_broken)
    status = lib.add([tmp_path / "tree.pdf"])
    assert any("indexed tree.pdf: 0 chunk(s)" in s for s in status)
    assert any("image-only PDF" in s for s in status)


def _lib_with(tmp_path, name: str, build_pdf) -> DocLibrary:
    """A library whose pdfs/ contains one built PDF (no chunks: renders only)."""
    lib = DocLibrary(tmp_path / "lib")
    lib.pdfs_dir.mkdir(parents=True, exist_ok=True)
    build_pdf(lib.pdfs_dir / name)
    return lib


def test_page_image_renders_and_bounds(tmp_path):
    def _build(path):
        _scanned_pdf(path)

    lib = _lib_with(tmp_path, "arch.pdf", _build)
    png = lib.page_image("arch.pdf", 1)
    assert png and png.startswith(b"\x89PNG")
    # rendered within the long-edge cap
    doc = pymupdf.open(stream=png, filetype="png")
    rect = doc[0].rect
    doc.close()
    assert max(rect.width, rect.height) <= 1200
    assert lib.page_image("arch.pdf", 99) is None      # out of range
    assert lib.page_image("arch.pdf", 0) is None
    assert lib.page_image("notes.md", 1) is None       # not a PDF
    assert lib.page_image("missing.pdf", 1) is None    # not indexed
    (tmp_path / "lib" / "pdfs" / "notes.md").write_text("# x")
    assert lib.page_image("notes.md", 1) is None


def test_page_image_text_page_fits_payload(tmp_path):
    """A realistic text page compresses well under the payload cap."""
    def _build(path):
        doc = pymupdf.open()
        page = doc.new_page()
        for i in range(40):
            page.insert_text((72, 72 + i * 14),
                             f"Register 0x{0x90 + i:02x} controls lane {i} power state")
        doc.save(str(path))
        doc.close()

    lib = _lib_with(tmp_path, "regs.pdf", _build)
    png = lib.page_image("regs.pdf", 1, max_bytes=60_000)
    assert png and len(png) <= 60_000


def test_page_image_over_cap_returns_best_effort(tmp_path):
    """When even the smallest render exceeds the cap, return it anyway --
    an oversized page image beats no image (noise images never compress)."""
    def _build(path):
        doc = pymupdf.open()
        doc.new_page().insert_image(pymupdf.Rect(72, 100, 372, 250),
                                    stream=_noise_png())
        doc.save(str(path))
        doc.close()

    lib = _lib_with(tmp_path, "noise.pdf", _build)
    png = lib.page_image("noise.pdf", 1, max_bytes=10_000)
    assert png is not None and len(png) > 10_000


def test_parse_pages_still_callable_with_injected_parser(tmp_path):
    """Regression guard: injected Parser callables (path -> pages) keep working."""
    def _fake(path):
        return [PageText(page=1, text="ECC rules")]

    lib = DocLibrary(tmp_path / "lib", parser=_fake)
    (tmp_path / "lib" / "pdfs").mkdir(parents=True)
    (tmp_path / "lib" / "pdfs" / "a.pdf").write_bytes(b"pdf")
    (tmp_path / "lib" / "index.json").write_text(json.dumps({"a.pdf": {"chunks": 1}}))
    chunks, error, captions = lib._parse_doc("a.pdf")
    assert error is None
    assert chunks and chunks[0].text == "ECC rules"
    assert captions == 0
