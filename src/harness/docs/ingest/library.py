"""Persisted RAG document library ("upload" point for architecture docs).

Layout of a library directory::

    <lib>/
      pdfs/            uploaded source files, any supported format (copy-in;
                       this is the upload surface; keep the name for back-compat)
      index.json       manifest: name -> sha256/size/chunks/error/ingested_at
      chunks.jsonl     cached chunks (text/title/page/index) for fast retrieval

``add`` copies files in (PDF / Markdown / plain text / CSV / JSON / logs /
DOCX), re-parses only files whose sha256 changed (or failed before), and
records the manifest atomically. ``rm``/``ls``/``reindex`` manage the store,
and ``build_retriever`` returns a ``RagPipeline`` over the cached chunks
without touching the sources again.

Parsing is dispatched on file extension: PDFs use the layout-aware
``PdfParser`` -- injectable so the library is testable without pymupdf, with an
optional ``ocr`` hook (typically the vision captioner in ``vision.py``) that
captures scanned pages and embedded diagram images -- while markdown/text/CSV/
JSON/DOCX use the lightweight parsers in ``text_parser``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..retrieval.hybrid_search import HybridRetriever
from ..retrieval.rag import RagPipeline
from .chunk import CharTokenizer, Chunk, Chunker
from .pdf_parser import PageText, PdfParser
from .text_parser import _CsvParser, _DocxParser, _JsonParser, _MarkdownParser, _PlainTextParser

MANIFEST = "index.json"
CHUNKS = "chunks.jsonl"
PDFS = "pdfs"

Parser = Callable[[Path], list[PageText]]

_TEXT_PARSERS: dict[str, Callable[[], object]] = {
    ".md": _MarkdownParser,
    ".markdown": _MarkdownParser,
    ".txt": _PlainTextParser,
    ".text": _PlainTextParser,
    ".log": _PlainTextParser,
    ".csv": _CsvParser,
    ".json": _JsonParser,
    ".docx": _DocxParser,
}

SUPPORTED_SUFFIXES = frozenset({".pdf"} | set(_TEXT_PARSERS))


@dataclass(frozen=True)
class DocEntry:
    name: str
    sha256: str
    size: int
    chunks: int
    ingested_at: str
    error: str | None = None
    platform: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _ocr_failures(ocr) -> int:
    return int(getattr(ocr, "failures", 0) or 0)


def _append_ocr_warn(status: list[str], ocr, before: int) -> None:
    """One warn line when image captioning failed during this batch.

    The captioner latches off after its first failure (no per-page endpoint
    hammering), so the count is surfaced as a batch-level condition, not a
    per-image tally; ``docs reindex`` retries once the endpoint is back.
    """
    if _ocr_failures(ocr) > before:
        status.append("warn: image captioning failed (LLM unavailable) -- scanned "
                      "pages/diagram images were skipped; 'docs reindex' retries")


def parse_pages(path: Path, pdf_parser: Parser | None = None) -> list[PageText]:
    """Parse a source file (PDF / markdown / text / CSV) into ``PageText`` list.

    Dispatch on file extension; PDFs use the layout-aware ``PdfParser`` (or an
    injected parser), text formats use the ``text_parser`` implementations.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if pdf_parser is not None:
            return pdf_parser(path)
        return PdfParser().parse(path)
    parser_cls = _TEXT_PARSERS[suffix]
    return parser_cls().parse(path)


class DocLibrary:
    def __init__(self, root: str | Path, parser: Parser | None = None,
                 ocr: Callable[[bytes], str] | None = None) -> None:
        """``parser`` overrides PDF parsing entirely (tests); ``ocr`` is the
        image-captioning hook handed to the default ``PdfParser`` (vision
        ingest; see ``vision.py``). Late-bound so monkeypatched ``PdfParser``
        test fakes keep working."""
        self.root = Path(root)
        self.pdfs_dir = self.root / PDFS
        self.manifest_path = self.root / MANIFEST
        self.chunks_path = self.root / CHUNKS
        self.parser = parser
        self.ocr = ocr

    # ---- manifest ----

    def _manifest(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _save_manifest(self, manifest: dict[str, dict]) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def _load_chunks(self) -> list[Chunk]:
        if not self.chunks_path.exists():
            return []
        return [Chunk(**json.loads(line)) for line in
                self.chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _save_chunks(self, chunks: list[Chunk]) -> None:
        tmp = self.chunks_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps({"text": c.text, "title": c.title,
                                     "page": c.page, "index": c.index,
                                     "platforms": c.platforms}) + "\n")
        tmp.replace(self.chunks_path)

    # ---- operations ----

    def add(self, files: list[str | Path], platform: str | None = None) -> list[str]:
        """Upload documents into the library and index them. Returns status lines.

        ``platform`` tags every chunk of those documents with a canonical model
        key so retrieval can be filtered to the detected server model. None
        leaves the documents untagged (platform-neutral knowledge).
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest()
        status: list[str] = []
        changed = False
        ocr_before = _ocr_failures(self.ocr)
        for raw in files:
            src = Path(raw)
            if not src.is_file():
                status.append(f"skip {src}: not a file")
                continue
            suffix = src.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                status.append(
                    f"skip {src.name}: unsupported type (supported: "
                    f"{', '.join(sorted(SUPPORTED_SUFFIXES))})")
                continue
            digest = _sha256(src)
            entry = manifest.get(src.name)
            if entry and entry.get("sha256") == digest and not entry.get("error"):
                status.append(f"unchanged {src.name} (already indexed)")
                continue
            shutil.copy2(src, self.pdfs_dir / src.name)
            chunks, error = self._parse_doc(src.name, platform)
            if error:
                manifest[src.name] = {
                    "sha256": digest, "size": src.stat().st_size,
                    "chunks": 0, "ingested_at": _now(), "error": error,
                    "platform": platform,
                }
                status.append(f"FAILED {src.name}: {error}")
            else:
                manifest[src.name] = {
                    "sha256": digest, "size": src.stat().st_size,
                    "chunks": len(chunks), "ingested_at": _now(),
                    "platform": platform,
                }
                self._save_chunks(self._all_chunks(manifest))
                status.append(f"indexed {src.name}: {len(chunks)} chunk(s)")
            changed = True
        if changed:
            self._save_manifest(manifest)
        _append_ocr_warn(status, self.ocr, ocr_before)
        return status

    def reindex(self) -> list[str]:
        """(Re-)index every supported file in ``pdfs/``: retries failures,
        picks up dropped files (manual copy into the library counts as upload)."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest()
        status: list[str] = []
        changed = False
        ocr_before = _ocr_failures(self.ocr)
        files = sorted(
            f for f in self.pdfs_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES)
        for src in files:
            digest = _sha256(src)
            entry = manifest.get(src.name)
            if entry and entry.get("sha256") == digest and not entry.get("error"):
                continue
            chunks, error = self._parse_doc(src.name)
            if error:
                manifest[src.name] = {
                    "sha256": digest, "size": src.stat().st_size,
                    "chunks": 0, "ingested_at": _now(), "error": error,
                }
                status.append(f"FAILED {src.name}: {error}")
            else:
                manifest[src.name] = {
                    "sha256": digest, "size": src.stat().st_size,
                    "chunks": len(chunks), "ingested_at": _now(),
                }
                status.append(f"indexed {src.name}: {len(chunks)} chunk(s)")
            changed = True
        if changed:
            self._save_chunks(self._all_chunks(manifest))
            self._save_manifest(manifest)
        _append_ocr_warn(status, self.ocr, ocr_before)
        return status or ["nothing to do"]

    def remove(self, name: str) -> str:
        manifest = self._manifest()
        if name not in manifest and not (self.pdfs_dir / name).exists():
            raise KeyError(f"not in library: {name!r}")
        manifest.pop(name, None)
        pdf = self.pdfs_dir / name
        if pdf.exists():
            pdf.unlink()
        self._save_chunks(self._all_chunks(manifest))
        self._save_manifest(manifest)
        return f"removed {name}"

    def retag(self, names: list[str], platform: str | None = None) -> list[str]:
        """Set (or clear) the platform tag on already-indexed documents.

        ``platform`` may be a single canonical key or a comma-separated list of
        keys; None clears the tag (back to platform-neutral). Chunks are rebuilt
        from the manifest, so previously-tagged chunks are retagged too. Unknown
        names are reported, never fatal.
        """
        manifest = self._manifest()
        status: list[str] = []
        for name in names:
            if name not in manifest:
                status.append(f"missing {name}: not in library")
                continue
            if platform is None:
                manifest[name].pop("platform", None)
            else:
                manifest[name]["platform"] = ",".join(
                    p.strip() for p in platform.split(",") if p.strip())
            status.append(f"tagged {name}: platform={platform or '(none)'}")
        if status:
            self._save_chunks(self._all_chunks(manifest))
            self._save_manifest(manifest)
        return status

    def entries(self) -> list[DocEntry]:
        manifest = self._manifest()
        return [DocEntry(name=n, **fields) for n, fields in sorted(manifest.items())]

    def build_retriever(self) -> RagPipeline | None:
        """Retriever over the cached index. Returns None when the library is empty."""
        chunks = self._load_chunks()
        if not chunks:
            return None
        return RagPipeline(HybridRetriever(chunks))

    # ---- internals ----

    def _parse_doc(self, name: str, platform: str | None = None) -> tuple[list[Chunk], str | None]:
        """Parse any supported source file (PDF / markdown / text / CSV / JSON /
        log / DOCX) into chunks."""
        path = self.pdfs_dir / name
        try:
            pages = self._pages(path)
        except Exception as exc:  # noqa: BLE001 - record per-file, never fatal
            return [], str(exc)
        chunks = Chunker(CharTokenizer()).chunk_pages(pages, title=name)
        if platform:
            # A comma-separated tag tags one chunk with multiple platforms (e.g.
            # a compute-tray guide that applies to both the node and the rack).
            for chunk in chunks:
                chunk.platforms = [p.strip() for p in platform.split(",") if p.strip()]
        return chunks, None

    def _pages(self, path: Path) -> list[PageText]:
        if self.parser is None and self.ocr is not None and path.suffix.lower() == ".pdf":
            # Late-bound so monkeypatched PdfParser test fakes keep working.
            return PdfParser(ocr=self.ocr).parse(path)
        return parse_pages(path, pdf_parser=self.parser)

    def _all_chunks(self, manifest: dict[str, dict]) -> list[Chunk]:
        """Rebuild the chunk cache from indexed, error-free entries."""
        chunks: list[Chunk] = []
        for name, entry in manifest.items():
            if entry.get("error"):
                continue
            parsed, error = self._parse_doc(name, entry.get("platform"))
            if error or not parsed:
                continue
            chunks.extend(parsed)
        return chunks


def _now() -> str:
    return datetime.now(UTC).isoformat()
