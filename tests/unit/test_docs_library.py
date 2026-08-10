"""Persisted RAG doc library: upload (add), rm, ls, reindex, retriever over cache."""

import pytest

from harness.docs.ingest.library import DocLibrary
from harness.docs.ingest.pdf_parser import PageText


def _fake_pdf(path) -> list[PageText]:
    return [PageText(page=1, text="memory ECC uncorrectable DIMM error IA32_MC0_STATUS")]


def _write_pdf(tmp_path, name: str, content: str):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_add_indexes_and_retriever_works(tmp_path):
    lib = DocLibrary(tmp_path / "lib", parser=_fake_pdf)
    src = _write_pdf(tmp_path, "arch.pdf", "pdf bytes v1")

    status = lib.add([src])
    assert any("indexed arch.pdf" in s for s in status)
    assert (tmp_path / "lib" / "pdfs" / "arch.pdf").exists()  # uploaded copy

    rag = lib.build_retriever()
    assert rag is not None
    lines = rag.lines("memory ECC error", top_k=1)
    assert lines and "[arch.pdf p.1]" in lines[0]

    entries = lib.entries()
    assert len(entries) == 1
    assert entries[0].name == "arch.pdf" and entries[0].chunks >= 1
    assert entries[0].error is None


def test_add_skips_unchanged_and_reindexes_changed(tmp_path):
    lib = DocLibrary(tmp_path / "lib", parser=_fake_pdf)
    src = _write_pdf(tmp_path, "arch.pdf", "v1")
    lib.add([src])

    status = lib.add([src])
    assert any("unchanged" in s for s in status)

    src.write_text("v2", encoding="utf-8")
    status = lib.add([src])
    assert any("indexed arch.pdf" in s for s in status)
    assert lib.build_retriever() is not None


def test_add_rejects_non_pdf_and_missing(tmp_path):
    lib = DocLibrary(tmp_path / "lib", parser=_fake_pdf)
    txt = _write_pdf(tmp_path, "notes.txt", "x")
    status = lib.add([txt, tmp_path / "missing.pdf"])
    assert any("not a .pdf" in s for s in status)
    assert any("not a file" in s for s in status)
    assert lib.entries() == []


def test_remove_and_ls(tmp_path):
    lib = DocLibrary(tmp_path / "lib", parser=_fake_pdf)
    lib.add([_write_pdf(tmp_path, "a.pdf", "1")])
    lib.add([_write_pdf(tmp_path, "b.pdf", "2")])

    lib.remove("a.pdf")
    assert [e.name for e in lib.entries()] == ["b.pdf"]
    assert not (tmp_path / "lib" / "pdfs" / "a.pdf").exists()
    assert lib.build_retriever() is not None  # cache rebuilt without the removed doc

    with pytest.raises(KeyError):
        lib.remove("nope.pdf")


def test_reindex_records_errors_and_retries(tmp_path):
    def flaky_parser(path):
        if path.read_text(encoding="utf-8").startswith("unreadable"):
            raise RuntimeError("unreadable")
        return [PageText(page=1, text="pcie link down")]

    lib = DocLibrary(tmp_path / "lib", parser=flaky_parser)
    good = _write_pdf(tmp_path, "good.pdf", "1")
    bad = _write_pdf(tmp_path, "bad.pdf", "unreadable content")

    status = lib.add([good, bad])
    assert any("FAILED bad.pdf" in s for s in status)
    by_name = {e.name: e for e in lib.entries()}
    assert by_name["bad.pdf"].error == "unreadable"
    assert by_name["good.pdf"].chunks >= 1

    (tmp_path / "lib" / "pdfs" / "bad.pdf").write_text("fixed now", encoding="utf-8")
    status = lib.reindex()
    assert any("indexed bad.pdf" in s for s in status)
    by_name = {e.name: e for e in lib.entries()}
    assert by_name["bad.pdf"].error is None


def test_reindex_picks_up_dropped_files(tmp_path):
    lib = DocLibrary(tmp_path / "lib", parser=_fake_pdf)
    dropped = tmp_path / "lib" / "pdfs" / "dropped.pdf"
    dropped.parent.mkdir(parents=True)
    dropped.write_text("manual upload", encoding="utf-8")

    status = lib.reindex()
    assert any("indexed dropped.pdf" in s for s in status)
    assert [e.name for e in lib.entries()] == ["dropped.pdf"]


def test_retriever_serves_from_cache_without_reparsing(tmp_path):
    calls = []

    def counting_parser(path):
        calls.append(path.name)
        return [PageText(page=1, text="kernel panic oops")]

    lib = DocLibrary(tmp_path / "lib", parser=counting_parser)
    lib.add([_write_pdf(tmp_path, "k.pdf", "1")])
    calls.clear()

    reopened = DocLibrary(tmp_path / "lib", parser=counting_parser)
    rag = reopened.build_retriever()
    assert rag is not None
    assert calls == []  # served from chunks.jsonl, PDFs untouched
    assert "[k.pdf p.1]" in rag.lines("kernel oops", top_k=1)[0]
