"""LLM adapters: on-prem OpenAI-compatible endpoint, plus a stub for tests.

The harness never ships model weights or a cloud dependency. The default adapter
talks to any OpenAI-compatible chat-completions server (vLLM / llama.cpp / local
gateway) over stdlib ``urllib`` so it can run fully on-prem with proprietary data.

Env config: ``HARNESS_LLM_URL`` (default http://127.0.0.1:8000/v1),
``HARNESS_LLM_API_KEY``, ``HARNESS_LLM_MODEL``, ``HARNESS_LLM_TIMEOUT``.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from urllib import error as urllib_error

from .schema import Diagnosis


class LLMError(RuntimeError):
    pass


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
        ``[{"role": "system"|"user"|"assistant", "content": str}]``.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
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
                data = json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise LLMError(f"LLM HTTP {exc.code}: {detail!r}") from exc
        except urllib_error.URLError as exc:
            raise LLMError(f"LLM unreachable: {exc.reason}") from exc
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
    except json.JSONDecodeError as exc:
        raise LLMError("LLM /models returned non-JSON") from exc
    try:
        return [str(m["id"]) for m in data["data"]]
    except (KeyError, TypeError) as exc:
        raise LLMError(f"LLM /models reply missing model ids: {exc}") from exc


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

    def chat_json(self, messages: list[dict], temperature: float = 0.0) -> dict:
        """Gemini's OpenAI-compat layer refuses system-only requests
        (systemInstruction with empty ``contents``). Guarantee a user turn."""
        if not any(m.get("role") == "user" for m in messages):
            messages = [*messages, {"role": "user",
                                    "content": "Produce the requested JSON output now."}]
        return super().chat_json(messages, temperature=temperature)


class StubLLM:
    """Deterministic stand-in: valid Diagnosis, no actions.

    Lets the whole pipeline (collect -> decode -> RAG -> score -> audit) run and
    be tested without any LLM infrastructure. The diagnosis text makes it obvious
    that no reasoning happened. In session mode it asks one question, then
    diagnoses, so multi-turn flows are exercised deterministically.
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
        if not self._asked:
            self._asked = True
            return {
                "kind": "question",
                "question": "What previous repair actions were already attempted?",
            }
        diag = self.__call__(messages[-1]["content"])
        return {"kind": "diagnosis", "diagnosis": diag.model_dump()}
