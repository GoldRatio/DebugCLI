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

    The first failed caption (unreachable endpoint, HTTP error, empty reply)
    raises and every later call short-circuits, so a 200-page scanned PDF with
    a dead endpoint costs one connection attempt instead of 200 timeouts.
    ``failures`` counts failed caption attempts for status/warn surfaces.
    """

    def __init__(self, llm) -> None:
        self.llm = llm
        self.failures = 0
        self._dead = False

    def __call__(self, png: bytes) -> str:
        if self._dead:
            raise LLMError("vision captioner disabled after a failed call")
        try:
            return self.llm.caption_image(png, prompt=CAPTION_PROMPT)
        except Exception:
            self.failures += 1
            self._dead = True
            raise


def vision_captioner(llm=None) -> Callable[[bytes], str] | None:
    """PNG-bytes -> text callable over ``llm`` (default: env-configured adapter).

    None when no LLM is available -- callers treat that as text-only ingest.
    """
    llm = llm or ingest_llm()
    if llm is None:
        return None
    return VisionCaptioner(llm)
