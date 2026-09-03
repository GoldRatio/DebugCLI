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


def test_add_rejects_unsupported_and_missing(tmp_path):
    lib = DocLibrary(tmp_path / "lib", parser=_fake_pdf)
    exe = _write_pdf(tmp_path, "notes.exe", "x")
    status = lib.add([exe, tmp_path / "missing.pdf"])
    assert any("unsupported type" in s for s in status)
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


def test_retag_sets_clears_and_multi_tags(tmp_path):
    lib = DocLibrary(tmp_path / "lib", parser=_fake_pdf)
    lib.add([_write_pdf(tmp_path, "node.pdf", "1")])

    status = lib.retag(["node.pdf"], platform="samoa")
    assert any("tagged node.pdf" in s for s in status)
    chunks = lib.build_retriever().retriever.chunks
    assert all(c.platforms == ["samoa"] for c in chunks)

    # multi-platform tag (e.g. a tray guide for node + rack)
    lib.retag(["node.pdf"], platform="nvl72,samoa")
    chunks = lib.build_retriever().retriever.chunks
    assert all(set(c.platforms) == {"nvl72", "samoa"} for c in chunks)

    # clearing returns the chunk to platform-neutral
    lib.retag(["node.pdf"])
    assert lib.entries()[0].platform is None
    chunks = lib.build_retriever().retriever.chunks
    assert all(c.platforms == [] for c in chunks)


def test_retag_reports_missing_and_survives_reload(tmp_path):
    lib = DocLibrary(tmp_path / "lib", parser=_fake_pdf)
    lib.add([_write_pdf(tmp_path, "node.pdf", "1")])
    lib.retag(["node.pdf"], platform="samoa")
    missing_status = lib.retag(["missing.pdf"], platform="samoa")

    reloaded = DocLibrary(tmp_path / "lib", parser=_fake_pdf)
    assert reloaded.entries()[0].platform == "samoa"
    assert any("missing.pdf" in s for s in missing_status)


# ---- multi-format ingestion (markdown / text / csv) ----

def test_add_markdown_splits_on_headings(tmp_path):
    lib = DocLibrary(tmp_path / "lib")
    md = tmp_path / "runbook.md"
    md.write_text(
        "# First Section\n\nmemory ECC uncorrectable DIMM error\n\n"
        "## Second\n\npcie link down after reboot\n",
        encoding="utf-8",
    )

    status = lib.add([md])
    assert any("indexed runbook.md" in s for s in status)
    assert (tmp_path / "lib" / "pdfs" / "runbook.md").exists()

    rag = lib.build_retriever()
    lines = rag.lines("DIMM ECC error", top_k=2)
    assert lines
    assert any("[runbook.md p.1]" in l for l in lines)
    entries = lib.entries()
    assert entries[0].name == "runbook.md" and entries[0].chunks >= 1


def test_add_markdown_strips_front_matter(tmp_path):
    lib = DocLibrary(tmp_path / "lib")
    md = tmp_path / "runbook.md"
    md.write_text("---\ntitle: Runbook\n---\n# Memory\n\ndimm error\n", encoding="utf-8")

    lib.add([md])
    assert lib.build_retriever() is not None


def test_add_csv_searchable_by_cell_values(tmp_path):
    lib = DocLibrary(tmp_path / "lib")
    csv = tmp_path / "parts.csv"
    csv.write_text("slot,part,status\n3,PSU-01,FAULT\n7,HDD-02,OK\n", encoding="utf-8")

    status = lib.add([csv])
    assert any("indexed parts.csv" in s for s in status)
    rag = lib.build_retriever()
    lines = rag.lines("PSU-01 FAULT", top_k=2)
    assert lines and "[parts.csv p.1]" in lines[0]


def test_add_plain_text(tmp_path):
    lib = DocLibrary(tmp_path / "lib")
    txt = tmp_path / "notes.txt"
    txt.write_text("CPLD records power faults in I2C registers 0x90-0x96\n", encoding="utf-8")

    lib.add([txt])
    assert lib.entries()[0].chunks >= 1
    lines = lib.build_retriever().lines("CPLD power fault I2C register", top_k=1)
    assert lines and "[notes.txt p.1]" in lines[0]


def test_add_rejects_unsupported_and_reindex_picks_up_text(tmp_path):
    lib = DocLibrary(tmp_path / "lib")
    exe = _write_pdf(tmp_path, "tool.exe", "nope")
    status = lib.add([exe])
    assert any("unsupported type" in s for s in status)
    assert lib.entries() == []

    dropped = tmp_path / "lib" / "pdfs" / "dropped.md"
    dropped.parent.mkdir(parents=True, exist_ok=True)
    dropped.write_text("# Notes\n\nmemory configuration guide\n", encoding="utf-8")
    status = lib.reindex()
    assert any("indexed dropped.md" in s for s in status)
    assert lib.build_retriever() is not None


def test_remove_text_file(tmp_path):
    lib = DocLibrary(tmp_path / "lib")
    lib.add([_write_pdf(tmp_path, "a.md", "# A\n\ncontent here")])
    lib.remove("a.md")
    assert lib.entries() == []
    assert not (tmp_path / "lib" / "pdfs" / "a.md").exists()
    assert lib.build_retriever() is None


def test_add_json_flattened_for_search(tmp_path):
    lib = DocLibrary(tmp_path / "lib")
    js = tmp_path / "sensors.json"
    js.write_text(
        '{"sensor": {"name": "CPU0 Temp", "reading": "72C",'
        ' "status": "threshold_high"}}', encoding="utf-8")

    status = lib.add([js])
    assert any("indexed sensors.json" in s for s in status)
    text = "\n".join(c.text for c in lib._load_chunks())
    assert "sensor.name: CPU0 Temp" in text
    assert "sensor.status: threshold_high" in text
    lines = lib.build_retriever().lines("CPU0 Temp threshold_high", top_k=1)
    assert lines and "[sensors.json p.1]" in lines[0]


def test_add_log_file(tmp_path):
    lib = DocLibrary(tmp_path / "lib")
    log = tmp_path / "sel.log"
    log.write_text("SEL: corrected ECC error on DIMM A1\nSEL: PSU0 input lost\n",
                   encoding="utf-8")

    status = lib.add([log])
    assert any("indexed sel.log" in s for s in status)
    lines = lib.build_retriever().lines("corrected ECC DIMM A1", top_k=1)
    assert lines and "[sel.log p.1]" in lines[0]


def test_add_docx_without_dependency_records_error(tmp_path, monkeypatch):
    """A .docx without python-docx is a staged per-file error, never a batch
    failure (and this stays true even when the dependency IS installed)."""
    import sys

    monkeypatch.setitem(sys.modules, "docx", None)  # force the ImportError path
    lib = DocLibrary(tmp_path / "lib")
    docx = tmp_path / "guide.docx"
    docx.write_bytes(b"PK\x03\x04 fake docx")

    status = lib.add([docx])
    assert any("FAILED guide.docx" in s for s in status)
    entry = lib.entries()[0]
    assert entry.error and "python-docx" in entry.error
    assert lib.build_retriever() is None  # no chunks cached for the failed file
