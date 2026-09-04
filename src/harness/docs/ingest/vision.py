"""Vision captioning for RAG ingest: page/diagram images -> searchable text.

``vision_captioner`` returns an ``ocr``-style callable (PNG bytes -> text)
backed by the configured LLM adapter (OpenAI-compatible endpoint, Gemini, local
vLLM, or stub -- same env config as the diagnosis adapters). Returns ``None``
when no LLM is configured so ingest degrades to text-only without any
infrastructure. Caption failures latch the captioner off for the rest of the
run (one unreachable endpoint never becomes per-page timeouts) and are counted
on ``.failures`` so the library can surface a warn line; the parser degrades
each failed image to text-only.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from harness.diagnosis.llm import (
    CAPTION_PROMPT,
    GeminiLLM,
    LLMError,
    LocalLLM,
    OpenAICompatLLM,
    StubLLM,
)


def ingest_llm():
    """LLM adapter for document ingest, built from env config.

    ``HARNESS_LLM_PROVIDER`` (openai | gemini | local | stub, default openai)
    selects the adapter; endpoint/model/key come from the same env vars the
    diagnosis adapters read (``HARNESS_LLM_URL``, ``HARNESS_LLM_MODEL``,
    ``HARNESS_LLM_API_KEY``). ``none``/``off`` disables ingest captioning
    entirely (text-only, no endpoint attempts).
    """
    provider = os.environ.get("HARNESS_LLM_PROVIDER", "openai").strip().lower()
    if provider in ("", "none", "off"):
        return None
    model = os.environ.get("HARNESS_LLM_MODEL")
    if provider == "stub":
        return StubLLM()
    if provider == "gemini":
        return GeminiLLM(model=model or "gemini-2.5-flash")
    if provider == "local":
        return LocalLLM(model=model or "harness-diag")
    return OpenAICompatLLM(model=model or "harness-diag")


class VisionCaptioner:
    """PNG-bytes -> text over an LLM adapter, with failure latching.

    Transient per-image failures (an occasional empty reply from a
    thinking-then-answering model) are retried inside ``caption_image`` and
    tolerated here; the captioner only latches off after ``max_failures``
    consecutive failures, so a flaky page never kills a 200-page batch while
    a dead endpoint still costs bounded attempts. ``failures`` counts failed
    caption attempts and ``error`` holds the first failure, for the warn
    surface; the parser degrades each failed image to text-only.
    """

    max_consecutive_failures = 5

    def __init__(self, llm) -> None:
        self.llm = llm
        self.failures = 0
        self._consecutive = 0
        self.error: str | None = None  # first failure, for the warn surface
        self._dead = False

    def __call__(self, png: bytes) -> str:
        if self._dead:
            raise LLMError("vision captioner disabled after repeated failures")
        # The hook receives PNG bytes from page renders and native JPEG bytes
        # for small embedded JPEGs; sniff the mime from the magic bytes.
        mime = "image/jpeg" if png[:3] == b"\xff\xd8\xff" else "image/png"
        try:
            text = self.llm.caption_image(png, prompt=CAPTION_PROMPT, mime=mime)
        except Exception as exc:
            self.failures += 1
            self._consecutive += 1
            if self.error is None:
                self.error = f"{type(exc).__name__}: {exc}"
            if self._consecutive >= self.max_consecutive_failures:
                self._dead = True
            raise
        self._consecutive = 0
        return text


def vision_captioner(llm=None) -> Callable[[bytes], str] | None:
    """PNG-bytes -> text callable over ``llm`` (default: env-configured adapter).

    None when no LLM is available -- callers treat that as text-only ingest.
    """
    llm = llm or ingest_llm()
    if llm is None:
        return None
    return VisionCaptioner(llm)
