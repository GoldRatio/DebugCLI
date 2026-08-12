"""Platform-tagged RAG: chunk metadata, library persistence, filtered retrieval."""

import json
from pathlib import Path

from harness.diagnosis.prompt import build_prompt, build_turn_evidence
from harness.diagnosis.summarize import EvidenceSummary
from harness.docs.ingest.chunk import Chunk
from harness.docs.ingest.library import DocLibrary
from harness.docs.retrieval.hybrid_search import HybridRetriever
from harness.docs.retrieval.rag import RagPipeline
from harness.inspect.model import from_alias


def _chunks() -> list[Chunk]:
    return [
        Chunk(text="register map for the R650 memory controller",
              title="R650_regmap.pdf", page=3, index=0,
              platforms=["poweredge_r650"]),
        Chunk(text="generic SEL event interpretation guidance",
              title="SEL_guide.pdf", page=1, index=1, platforms=[]),
        Chunk(text="register map for the R750 memory controller",
              title="R750_regmap.pdf", page=4, index=2,
              platforms=["poweredge_r750"]),
    ]


def test_platform_filter_returns_tagged_and_untagged_only():
    retriever = HybridRetriever(_chunks())
    got = retriever.query("R650 memory register", top_k=10, platform="poweredge_r650")
    titles = {c.title for c in got}
    assert "R650_regmap.pdf" in titles
    assert "SEL_guide.pdf" in titles
    assert "R750_regmap.pdf" not in titles


def test_platform_filter_empty_corpus_returns_empty():
    retriever = HybridRetriever([c for c in _chunks()
                                 if c.title != "SEL_guide.pdf"])
    assert retriever.query("memory", platform="proliant_dl380g10") == []


def test_no_platform_filter_returns_everything():
    retriever = HybridRetriever(_chunks())
    got = retriever.query("memory", top_k=10)
    assert len(got) == 3


def test_rag_pipeline_passes_platform():
    rag = RagPipeline(HybridRetriever(_chunks()))
    lines = rag.lines("R650 memory", top_k=10, platform="poweredge_r650")
    assert all("R650_regmap" in line or "SEL_guide" in line for line in lines)
    assert not any("R750_regmap" in line for line in lines)


def _mini_doc(tmp_path: Path, text: str) -> Path:
    p = tmp_path / (text.split()[1] + ".txt") if len(text.split()) > 1 \
        else tmp_path / "doc.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_library_persists_platform_tags(tmp_path: Path):
    lib = DocLibrary(tmp_path / "lib")
    doc = _mini_doc(tmp_path, "r650 register map page one")
    lines = lib.add([doc], platform="poweredge_r650")
    assert any("indexed" in line for line in lines)
    entry = next(e for e in lib.entries() if e.name == doc.name)
    assert entry.platform == "poweredge_r650"

    reloaded = DocLibrary(tmp_path / "lib")
    chunks = reloaded.build_retriever().retriever.chunks
    assert chunks and all(c.platforms == ["poweredge_r650"] for c in chunks)


def test_library_legacy_chunk_line_loads_without_platforms(tmp_path: Path):
    lib_dir = tmp_path / "lib"
    (lib_dir / "pdfs").mkdir(parents=True)
    with open(lib_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "legacy chunk", "title": "old.pdf",
                             "page": 1, "index": 0}) + "\n")
    chunks = lib._load_chunks() if (lib := DocLibrary(lib_dir)) else []
    assert chunks and chunks[0].platforms == []


def test_library_reindex_preserves_platform(tmp_path: Path):
    lib = DocLibrary(tmp_path / "lib")
    doc = _mini_doc(tmp_path, "r750 register map page one")
    lib.add([doc], platform="poweredge_r750")
    lib.reindex()  # unchanged docs are skipped; platform must survive
    entry = next(e for e in lib.entries() if e.name == doc.name)
    assert entry.platform == "poweredge_r750"
    chunks = DocLibrary(tmp_path / "lib").build_retriever().retriever.chunks
    assert all(c.platforms == ["poweredge_r750"] for c in chunks)


def _prompt_lines(**kw):
    return build_turn_evidence(
        model=kw.get("model"),
        symptom="ECC",
        decoded=kw.get("decoded", []),
        summaries=EvidenceSummary(interesting=[], anomaly_count=0, total=0),
        doc_snippets=kw.get("doc_snippets", ["[doc p.1] a snippet"]),
        parts_refs=[],
        conversation=[],
    )


def test_prompt_renders_rag_platform_filter():
    p = build_prompt(model=from_alias("poweredge_r650"), decoded=[],
                     summaries=EvidenceSummary(interesting=[], anomaly_count=0,
                                               total=0),
                     doc_snippets=["[doc p.1] a snippet"], parts_refs=[],
                     symptom="ECC")
    assert "rag_platform_filter=poweredge_r650" in p
    assert "retrieved for platform poweredge_r650" in p

    p_none = build_prompt(model=None, decoded=[],
                          summaries=EvidenceSummary(interesting=[],
                                                    anomaly_count=0, total=0),
                          doc_snippets=["[doc p.1] a snippet"],
                          parts_refs=[], symptom="ECC")
    assert "rag_platform_filter=none" in p_none
    assert "retrieved with no platform filter" in p_none


def test_prompt_flags_platform_mismatched_decode():
    from harness.inspect.base import RegisterDecode
    decode = RegisterDecode(mnemonic="R750_MC", raw_hex="0x01",
                            platforms=["poweredge_r750"])
    p = build_prompt(model=from_alias("poweredge_r650"), decoded=[decode],
                     summaries=EvidenceSummary(interesting=[], anomaly_count=0,
                                               total=0),
                     doc_snippets=[], parts_refs=[], symptom="ECC")
    assert "[PLATFORM MISMATCH - verify]" in p

    ok = build_prompt(model=from_alias("poweredge_r750"), decoded=[decode],
                      summaries=EvidenceSummary(interesting=[], anomaly_count=0,
                                                total=0),
                      doc_snippets=[], parts_refs=[], symptom="ECC")
    assert "[PLATFORM MISMATCH - verify]" not in ok


def test_prompt_mismatch_marker_only_when_catalog_scoped():
    from harness.inspect.base import RegisterDecode
    generic = RegisterDecode(mnemonic="CPLD_A1", raw_hex="0x00", platforms=[])
    p = _prompt_lines(model=from_alias("poweredge_r650"), decoded=[generic])
    assert "[PLATFORM MISMATCH" not in p