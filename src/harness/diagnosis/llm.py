"""LLM adapters: on-prem OpenAI-compatible endpoint, plus a stub for tests.

The harness never ships model weights or a cloud dependency. The default adapter
talks to any OpenAI-compatible chat-completions server (vLLM / llama.cpp / local
gateway) over stdlib ``urllib`` so it can run fully on-prem with proprietary data.

Env config: ``HARNESS_LLM_URL`` (default http://127.0.0.1:8000/v1),
``HARNESS_LLM_API_KEY``, ``HARNESS_LLM_MODEL``, ``HARNESS_LLM_TIMEOUT``.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from urllib import error as urllib_error

from .schema import Diagnosis


class LLMError(RuntimeError):
    pass


CAPTION_PROMPT = (
    "Transcribe every piece of text visible in this image verbatim (labels, "
    "headings, table cells, callouts, legend entries, axis names), preserving "
    "reading order. Then add one short paragraph describing what the image "
    "shows (diagram type, components, connections, states). Output plain text "
    "only, no markdown fences."
)


class OpenAICompatLLM:
    """Chat-completions client; ``__call__(prompt) -> Diagnosis`` (schema-validated)."""

    def __init__(self, url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 120.0) -> None:
        self.url = (url or os.environ.get("HARNESS_LLM_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("HARNESS_LLM_API_KEY")
        self.model = model or os.environ.get("HARNESS_LLM_MODEL", "harness-diag")
        self.timeout = float(os.environ.get("HARNESS_LLM_TIMEOUT", timeout))
        self.json_mode = True  # request response_format json_object; Gemini can't

    def __call__(self, prompt: str) -> Diagnosis:
        """Single-shot diagnosis: system message = the built diagnosis prompt.
        Retries once with the exact schema when the model skips required fields."""
        messages = [{"role": "system", "content": prompt}]
        raw = self.chat_json(messages)
        try:
            return self._parse(json.dumps(raw))
        except LLMError as exc:
            messages.append({"role": "assistant", "content": json.dumps(raw)})
            messages.append({"role": "user", "content": _schema_fix_prompt(str(exc))})
            return self._parse(json.dumps(self.chat_json(messages)))

    def chat_json(self, messages: list[dict], temperature: float = 0.0) -> dict:
        """Raw chat-completions call returning the parsed JSON object.

        Used by the multi-turn session engine where the agent answers with
        ``{kind: question|probe|diagnosis, ...}``. ``messages`` are
        ``[{"role": ..., "content": str | parts}]``; ``content`` may be a
        plain string or an OpenAI-style parts list (text + base64 image parts)
        for vision-capable endpoints.
        """
        if not any(m.get("role") == "user" for m in messages):
            # Several OpenAI-compatible layers (Gemini's, strict local
            # gateways) refuse system-only payloads with HTTP 400
            # ``No user query found in messages`` -- the single-shot
            # diagnosis path builds exactly that shape, so every adapter
            # guarantees a user turn on the wire.
            messages = [*messages, {"role": "user",
                                    "content": "Produce the requested JSON output now."}]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        data = self._post_chat(body)
        content = data["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            cleaned = _coerce_json_text(content)
            if cleaned is not None:
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
            raise LLMError(f"LLM returned non-JSON: {content[:200]!r}") from None

    def caption_image(self, image_bytes: bytes, prompt: str | None = None,
                      mime: str = "image/png") -> str:
        """Vision captioning: describe/transcribe an image via a multimodal turn.

        Sends the image as a base64 data URL part (OpenAI wire format, honored
        by Gemini's OpenAI-compat layer and vLLM vision models) and returns the
        model's plain-text reply. One retry on an empty/malformed reply (models
        that think-before-answering occasionally emit an empty content with an
        otherwise healthy envelope); transport failures raise immediately.
        Raises :class:`LLMError` on final failure so callers can degrade to
        text-only ingest.
        """
        b64 = base64.b64encode(image_bytes).decode("ascii")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt or CAPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}]
        body = {"model": self.model, "messages": messages, "temperature": 0.0}
        content = self._caption_content(self._post_chat(body))
        if content is not None:
            return content
        # One retry: models that think before answering occasionally emit an
        # empty content with an otherwise healthy envelope (observed on a
        # rack-served vLLM vision model mid-batch).
        content = self._caption_content(self._post_chat(body))
        if content is None:
            raise LLMError("LLM caption reply was empty")
        return content

    def _caption_content(self, data: dict) -> str | None:
        """Caption text from a reply envelope; None when the content is empty."""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM caption reply malformed: {exc}") from exc
        if isinstance(content, str) and content.strip():
            return content.strip()
        return None

    def _post_chat(self, body: dict) -> dict:
        """POST a chat-completions body, returning the parsed reply envelope."""
        request = urllib.request.Request(
            f"{self.url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise LLMError(f"LLM HTTP {exc.code}: {detail!r}") from exc
        except urllib_error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                # connect-phase timeout, wrapped by urllib
                raise LLMError(_timeout_message(self.timeout)) from exc
            raise LLMError(f"LLM unreachable: {exc.reason}") from exc
        except TimeoutError as exc:
            # read timeout mid-generation: the socket sat silent past the
            # budget while the model was still producing (or loading)
            raise LLMError(_timeout_message(self.timeout)) from exc
        except OSError as exc:
            raise LLMError(f"LLM connection failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError("LLM returned non-JSON reply envelope") from exc

    @staticmethod
    def _parse(content: str) -> Diagnosis:
        try:
            return Diagnosis.model_validate_json(content)
        except Exception as exc:  # pydantic validation error surface as-is
            raise LLMError(f"LLM output failed Diagnosis schema validation: {exc}") from exc


def _schema_fix_prompt(errors: str) -> str:
    """Correction turn after a schema-validation failure: hand the model the
    exact expected shape so a weaker model can self-correct."""
    schema = json.dumps(Diagnosis.model_json_schema(), indent=2)
    return (
        "Your previous reply failed the required JSON schema validation:\n"
        f"{errors[:500]}\n\n"
        "Reply with ONLY strict JSON matching this exact schema (no markdown "
        f"fences, no extra fields, no prose):\n{schema}")


def _timeout_message(timeout: float) -> str:
    """Staged explanation for a chat-call socket timeout: state the budget and
    the two knobs (env override, profile timeout) instead of a bare
    ``timed out Timeout error``."""
    return (f"LLM response timed out after {timeout:.0f}s -- local models "
            "over a tunnel (long prompt, cold start) may need more; set "
            "HARNESS_LLM_TIMEOUT or the model profile timeout, or retry "
            "(a warm server generates faster)")


def _coerce_json_text(text: str) -> str | None:
    """Extract the JSON payload from model output (markdown fences, prose)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return None


class LocalLLM(OpenAICompatLLM):
    """Locally hosted OpenAI-compatible endpoint (vLLM, llama.cpp, Ollama ``/v1``).

    Same wire protocol as ``OpenAICompatLLM`` -- vLLM serves chat-completions
    natively and honors ``response_format: json_object``. Typically no API key;
    set ``HARNESS_LLM_API_KEY`` when a gateway fronts the server. When the
    server sits behind the rack-manager hop, pair with
    ``engine.tunnel.LLMForward`` (``--llm-tunnel HOST:PORT``) and pass the
    forward's local URL here. Env config: ``HARNESS_LLM_URL`` (default
    http://127.0.0.1:8000/v1), ``HARNESS_LLM_MODEL``.
    """

    def __init__(self, url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 120.0) -> None:
        super().__init__(url=url, api_key=api_key, model=model, timeout=timeout)
        self.json_mode = True  # vLLM supports structured JSON responses


def list_models(url: str | None = None, api_key: str | None = None,
                timeout: float = 10.0) -> list[str]:
    """Preflight: GET ``{url}/models`` and return the served model ids.

    Raises :class:`LLMError` when unreachable or the reply is malformed --
    used by ``harness llm check`` to stage transport vs HTTP failures.
    """
    base = (url or os.environ.get("HARNESS_LLM_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
    request = urllib.request.Request(f"{base}/models", method="GET")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise LLMError(f"LLM HTTP {exc.code}: {detail!r}") from exc
    except urllib_error.URLError as exc:
        raise LLMError(f"LLM unreachable: {exc.reason}") from exc
    except OSError as exc:
        # Socket-level resets/mid-read drops (e.g. a refused tunnel forward)
        # escape urllib as raw ConnectionResetError/TimeoutError -- a staged
        # LLMError keeps every caller on the warning path, never a traceback.
        raise LLMError(f"LLM connection failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LLMError("LLM /models returned non-JSON") from exc
    try:
        return [str(m["id"]) for m in data["data"]]
    except (KeyError, TypeError) as exc:
        raise LLMError(f"LLM /models reply missing model ids: {exc}") from exc


def probe_chat(url: str | None = None, api_key: str | None = None,
               model: str | None = None, timeout: float = 10.0) -> str:
    """Preflight: minimal ``chat/completions`` request in the exact wire
    shape the single-shot diagnosis call sends (system + guaranteed user
    turn, ``response_format: json_object``) with a 1-token cap.

    ``/models`` reachability says nothing about request validation: a server
    that refuses the run's payload (``No user query found in messages``, an
    unsupported ``response_format``) only says so at the ``reason`` step --
    after a full run's collection phase. This probe surfaces such 400s in
    seconds; ``harness llm check`` runs it after the transport stage.
    Returns the model id the server reports serving. Raises
    :class:`LLMError` on any failure stage.
    """
    base = (url or os.environ.get("HARNESS_LLM_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
    body = {
        "model": model or os.environ.get("HARNESS_LLM_MODEL", "harness-diag"),
        "messages": [
            {"role": "system", "content": 'Reply with the JSON object {"ok": true}.'},
            {"role": "user", "content": "Produce the requested JSON output now."},
        ],
        "temperature": 0.0,
        "max_tokens": 1,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise LLMError(f"LLM HTTP {exc.code}: {detail!r}") from exc
    except urllib_error.URLError as exc:
        raise LLMError(f"LLM unreachable: {exc.reason}") from exc
    except OSError as exc:
        # Socket-level resets/mid-read drops surface here too -- same staged
        # error contract as ``list_models`` (warning path, never a traceback).
        raise LLMError(f"LLM connection failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LLMError("LLM chat probe returned non-JSON") from exc
    if not isinstance(data, dict):
        raise LLMError("LLM chat probe reply malformed")
    return str(data.get("model") or body["model"])


class GeminiLLM(OpenAICompatLLM):
    """Gemini via its OpenAI-compatible endpoint.

    Same chat-completions wire protocol as ``OpenAICompatLLM``; only the default
    endpoint, key env var, and default model differ. Env config: ``GEMINI_API_KEY``
    (or ``HARNESS_LLM_API_KEY``), ``HARNESS_LLM_URL`` (default
    https://generativelanguage.googleapis.com/v1beta/openai), ``HARNESS_LLM_MODEL``
    (default gemini-2.5-flash).
    """

    def __init__(self, url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 120.0) -> None:
        super().__init__(
            url=url or os.environ.get("HARNESS_LLM_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai",
            api_key=api_key if api_key is not None
            else os.environ.get("GEMINI_API_KEY") or os.environ.get("HARNESS_LLM_API_KEY"),
            model=model or os.environ.get("HARNESS_LLM_MODEL", "gemini-2.5-flash"),
            timeout=timeout,
        )
        self.json_mode = False  # Gemini's OpenAI-compat layer rejects response_format


class StubLLM:
    """Deterministic stand-in: valid Diagnosis, no actions.

    Lets the whole pipeline (collect -> decode -> RAG -> score -> audit) run and
    be tested without any LLM infrastructure. The diagnosis text makes it obvious
    that no reasoning happened. As the REPL conversation agent it never calls a
    tool (tool "none"), so structured intents fall through to the keyword
    router.
    """

    def __init__(self) -> None:
        self._asked = False

    def __call__(self, prompt: str) -> Diagnosis:
        return Diagnosis(
            diagnosis=(
                "No LLM configured (set HARNESS_LLM_URL or use --llm openai); "
                "pipeline ran without reasoning."
            ),
            confidence=0.0,
            actions=[],
        )

    def chat_json(self, messages: list[dict]) -> dict:
        return {
            "say": "(stub agent) no reasoning LLM is configured; I cannot "
                   "decide on tools.",
            "tool": "none",
        }

    def caption_image(self, png_bytes: bytes, prompt: str | None = None,
                      mime: str = "image/png") -> str:
        """Deterministic offline caption so vision paths run (and test) without infra."""
        return "(stub caption) no vision LLM configured; image content unavailable."
