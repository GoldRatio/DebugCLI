"""Hybrid retrieval, parts graph, chunking, and operator gate smoke tests."""

import csv

from harness.diagnosis.schema import Action, Diagnosis, Risk
from harness.docs.ingest.chunk import CharTokenizer, Chunk, Chunker
from harness.docs.ingest.pdf_parser import PageText
from harness.docs.parts.parts_graph import load_parts_csv
from harness.docs.retrieval.hybrid_search import HybridRetriever, bm25_score
from harness.docs.retrieval.rag import RagPipeline
from harness.operator.tickets import NoOpTicketing


def test_chunker_overlap_metadata():
    tokenizer = CharTokenizer()
    chunker = Chunker(tokenizer, chunk_tokens=100, overlap=20)
    text = "word " * 260
    pages = [PageText(page=3, text=text)]
    chunks = chunker.chunk_pages(pages, title="Server_Arch_v2.3.pdf")
    assert len(chunks) >= 3
    assert all(c.page == 3 and c.title == "Server_Arch_v2.3.pdf" for c in chunks)


def test_chunker_preserves_text_roundtrip():
    tokenizer = CharTokenizer()
    chunker = Chunker(tokenizer, chunk_tokens=500, overlap=100)
    text = ("IA32_MC0_STATUS = 0x8000000000000001\n"
            "valid bit set: uncorrectable machine check on DIMM_A2\n")
    chunks = chunker.chunk_pages([PageText(page=7, text=text)], title="Server_Arch_v2.3.pdf")
    assert len(chunks) == 1
    assert "IA32_MC0_STATUS" in chunks[0].text
    assert "uncorrectable machine check" in chunks[0].text


def test_chunker_overlap_keeps_readable_text():
    tokenizer = CharTokenizer()
    chunker = Chunker(tokenizer, chunk_tokens=50, overlap=10)
    text = " ".join(f"register_{i}" for i in range(200))
    chunks = chunker.chunk_pages([PageText(page=1, text=text)], title="Arch.pdf")
    assert len(chunks) > 3
    for c in chunks:
        assert "register_" in c.text


def test_bm25_ranks_matching_higher():
    chunks = [Chunk("processor core cache issue", "A", 1, 0),
              Chunk("storage disk raid array", "B", 1, 1)]
    stats = (3.0, {})
    hi = bm25_score(["cache"], chunks[0], stats)
    lo = bm25_score(["cache"], chunks[1], stats)
    assert hi > lo


def test_hybrid_retrieval_returns_top_k_with_pages():
    chunks = [Chunk("memory ECC uncorrectable DIMM error", "Arch", 7, 0),
              Chunk("PCIe link down recovery", "Arch", 12, 1)]
    r = HybridRetriever(chunks)
    out = r.query("memory ECC error", top_k=2)
    assert out[0].page == 7
    assert all(c.page in (7, 12) for c in out)


def test_rag_lines_carry_citations():
    chunks = [Chunk("memory ECC uncorrectable DIMM error", "Arch", 7, 0)]
    rag = RagPipeline(HybridRetriever(chunks))
    lines = rag.lines("memory error", top_k=1)
    assert "[Arch p.7]" in lines[0]


def test_parts_csv_load(tmp_path):
    p = tmp_path / "parts.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "fru", "pn", "sn"])
        w.writerow(["DIMM_SLOT_A1", "12345", "PN-9", "SN-X"])
    graph = load_parts_csv(p)
    by_slot = graph.by_slot()
    assert by_slot["DIMM_SLOT_A1"]["fru"] == "12345"
    assert graph.entries[0].line == 2


def test_ticketing_submits_actions():
    diag = Diagnosis(
        diagnosis="d",
        confidence=0.5,
        actions=[Action(step=1, action="Reseat DIMM", rationale="doc p78",
                        risk=Risk.LOW, required_tool="Physical", impact="reboot")],
        references=[],
    )
    ticket = NoOpTicketing().submit(diag)
    assert ticket.startswith("DIAG-")