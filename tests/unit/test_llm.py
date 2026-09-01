"""LLM adapters: schema validation, stub behavior, and error surfacing."""

import pytest

from harness.diagnosis.llm import (
    GeminiLLM,
    LLMError,
    LocalLLM,
    OpenAICompatLLM,
    StubLLM,
    list_models,
    probe_chat,
)
from harness.diagnosis.schema import Diagnosis, Risk


def test_stub_llm_returns_valid_empty_diagnosis():
    diag = StubLLM()("any prompt")
    assert isinstance(diag, Diagnosis)
    assert diag.actions == []
    assert "No LLM configured" in diag.diagnosis


def test_parse_valid_json():
    content = ('{"diagnosis": "memory error", "confidence": 0.5,'
               '"actions": [{"step": 1, "action": "Reseat DIMM", "rationale": "doc p78",'
               '"risk": "low", "required_tool": "Physical", "impact": "reboot"}],'
               '"references": []}')
    diag = OpenAICompatLLM._parse(content)
    assert diag.actions[0].risk == Risk.LOW
    assert diag.actions[0].step == 1


def test_parse_rejects_invalid_json():
    with pytest.raises(LLMError):
        OpenAICompatLLM._parse('{"diagnosis": 42}')  # wrong type -> schema error


def test_llm_unreachable_server_raises_clear_error():
    llm = OpenAICompatLLM(url="http://127.0.0.1:1/v1", timeout=1.0)  # nothing listens
    with pytest.raises(LLMError):
        llm("prompt")


def test_gemini_llm_defaults(monkeypatch):
    monkeypatch.delenv("HARNESS_LLM_URL", raising=False)
    monkeypatch.delenv("HARNESS_LLM_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HARNESS_LLM_API_KEY", raising=False)
    llm = GeminiLLM()
    assert "generativelanguage.googleapis.com" in llm.url
    assert llm.model == "gemini-2.5-flash"
    assert llm.api_key is None


def test_gemini_llm_env_and_explicit_overrides(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k-env")
    monkeypatch.setenv("HARNESS_LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("HARNESS_LLM_URL", "http://gateway:8000/v1")
    assert GeminiLLM().api_key == "k-env"
    assert GeminiLLM().model == "gemini-2.0-flash"
    assert GeminiLLM().url == "http://gateway:8000/v1"
    assert GeminiLLM(api_key="k-explicit").api_key == "k-explicit"


def test_gemini_llm_unreachable_raises_clear_error():
    llm = GeminiLLM(url="http://127.0.0.1:1/v1", api_key="k", timeout=1.0)
    with pytest.raises(LLMError):
        llm("prompt")


def test_gemini_disables_response_format_json_mode():
    # Gemini's OpenAI-compat layer rejects response_format; json_mode must be off.
    assert GeminiLLM().json_mode is False
    assert OpenAICompatLLM().json_mode is True


def test_json_fenced_output_is_coerced():
    from harness.diagnosis.llm import _coerce_json_text

    assert _coerce_json_text('```json\n{"ok": true}\n```') == '{"ok": true}'
    assert _coerce_json_text('```\n{"ok": true}\n```') == '{"ok": true}'
    assert _coerce_json_text('sure, here: {"ok": true} done.') == '{"ok": true}'
    assert _coerce_json_text("no json here") is None


def test_fenced_reply_parses_via_chat_json(monkeypatch):
    import json

    import harness.diagnosis.llm as llm_mod

    class _FakeResp:
        def __init__(self, data: bytes):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        assert b"response_format" not in request.data  # never sent for Gemini
        return _FakeResp(json.dumps({
            "choices": [{"message": {"content": '```json\n{"ok": true}\n```'}}],
        }).encode())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    llm = GeminiLLM(url="http://127.0.0.1:9/v1", api_key="k", timeout=1.0)
    assert llm.chat_json([{"role": "user", "content": "hi"}]) == {"ok": True}


@pytest.mark.parametrize("cls", [OpenAICompatLLM, LocalLLM, GeminiLLM])
def test_system_only_request_gets_user_turn_all_adapters(monkeypatch, cls):
    """The single-shot diagnosis path builds a system-only payload; some
    OpenAI-compatible layers (Gemini's, strict local gateways -- observed as
    HTTP 400 ``No user query found in messages`` from a rack-served vLLM)
    refuse it. Every adapter must guarantee a user turn on the wire, and
    never double it when one is already present."""
    import json

    import harness.diagnosis.llm as llm_mod

    class _FakeResp:
        def __init__(self, data: bytes):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen = []

    def fake_urlopen(request, timeout):
        seen.append(json.loads(request.data)["messages"])
        return _FakeResp(json.dumps({
            "choices": [{"message": {"content": '{"ok": true}'}}],
        }).encode())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    llm = cls(url="http://127.0.0.1:9/v1", api_key="k", timeout=1.0)
    assert llm.chat_json([{"role": "system", "content": "diagnose"}]) == {"ok": True}
    assert [m["role"] for m in seen[0]] == ["system", "user"]
    assert llm.chat_json([{"role": "user", "content": "hi"}]) == {"ok": True}
    assert [m["role"] for m in seen[1]] == ["user"]  # passthrough, no duplicate


def _diagnosis_json(confidence=0.5):
    return {
        "diagnosis": "memory error", "confidence": confidence,
        "actions": [{
            "step": 1, "action": "Reseat DIMM", "rationale": "doc p78",
            "risk": "low", "required_tool": "Physical", "impact": "reboot",
        }],
        "references": [],
    }


def test_single_shot_retries_with_schema_on_validation_failure(monkeypatch):
    import json

    import harness.diagnosis.llm as llm_mod

    class _FakeResp:
        def __init__(self, data: bytes):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    calls = []

    def fake_urlopen(request, timeout):
        messages = json.loads(request.data)["messages"]
        calls.append(messages)
        if len(calls) == 1:
            # model returns a structurally wrong Diagnosis (no confidence/step/...)
            content = json.dumps({"diagnosis": "the system is degraded",
                                  "actions": [{"action": "review logs"}]})
        else:
            content = json.dumps(_diagnosis_json(confidence=0.9))
        return _FakeResp(json.dumps(
            {"choices": [{"message": {"content": content}}]}).encode())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    llm = OpenAICompatLLM(url="http://127.0.0.1:9/v1", api_key="k", timeout=1.0)
    diag = llm("Produce a Diagnosis.")
    assert diag.confidence == 0.9
    assert len(calls) == 2  # one corrective retry happened
    fix = calls[1][-1]["content"]
    assert "exact schema" in fix
    assert "confidence" in fix  # schema embedded in the correction turn


# ---- local provider (vLLM / llama.cpp / Ollama) ----

def test_local_llm_defaults(monkeypatch):
    monkeypatch.delenv("HARNESS_LLM_URL", raising=False)
    monkeypatch.delenv("HARNESS_LLM_MODEL", raising=False)
    llm = LocalLLM()
    assert llm.url == "http://127.0.0.1:8000/v1"
    assert llm.model == "harness-diag"
    assert llm.api_key is None
    assert llm.json_mode is True  # vLLM honors response_format json_object


def test_local_llm_env_and_explicit_overrides(monkeypatch):
    monkeypatch.setenv("HARNESS_LLM_URL", "http://10.0.0.42:8000/v1")
    monkeypatch.setenv("HARNESS_LLM_MODEL", "Qwen2.5-7B-Instruct")
    monkeypatch.setenv("HARNESS_LLM_API_KEY", "tok")
    llm = LocalLLM()
    assert llm.url == "http://10.0.0.42:8000/v1"
    assert llm.model == "Qwen2.5-7B-Instruct"
    assert llm.api_key == "tok"
    explicit = LocalLLM(url="http://127.0.0.1:9/v1", model="m")
    assert explicit.url == "http://127.0.0.1:9/v1"
    assert explicit.api_key == "tok"   # None falls back to env (shared adapter contract)
    assert LocalLLM(api_key=None).api_key == "tok"
    assert LocalLLM(api_key="direct").api_key == "direct"


def test_list_models_parses_served_ids(monkeypatch):
    import json

    import harness.diagnosis.llm as llm_mod

    class _FakeResp:
        def __init__(self, data: bytes):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        return _FakeResp(json.dumps(
            {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}, {"id": "m2"}]}).encode())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    ids = list_models("http://127.0.0.1:9/v1/", timeout=5.0)
    assert ids == ["Qwen/Qwen2.5-7B-Instruct", "m2"]
    assert seen["url"].endswith("/models")  # trailing slash stripped


def test_list_models_unreachable_raises_llm_error():
    with pytest.raises(LLMError):
        list_models("http://127.0.0.1:1/v1", timeout=1.0)


def test_list_models_connection_reset_is_staged(monkeypatch):
    """A refused tunnel forward resets the socket mid-read -- the raw
    ConnectionResetError must surface as a staged LLMError, never a
    traceback out of the wizard."""
    import harness.diagnosis.llm as llm_mod

    def reset_during_read(request, timeout):
        raise ConnectionResetError(10054, "forcibly closed by the remote host")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", reset_during_read)
    with pytest.raises(LLMError, match="LLM connection failed"):
        list_models("http://127.0.0.1:9/v1")


def test_list_models_malformed_reply_raises(monkeypatch):
    import harness.diagnosis.llm as llm_mod

    class _FakeResp:
        def read(self):
            return b'{"unexpected": true}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen",
                        lambda request, timeout: _FakeResp())
    with pytest.raises(LLMError):
        list_models("http://127.0.0.1:9/v1")


# ---- probe_chat: minimal chat/completions preflight (llm check stage) ----

def test_probe_chat_mirrors_single_shot_wire_shape(monkeypatch):
    """The probe must send the same request shape the run's ``reason`` step
    sends -- system + guaranteed user turn, json response format -- so a
    server-side validation 400 surfaces here, before a run's collection
    phase."""
    import json

    import harness.diagnosis.llm as llm_mod

    class _FakeResp:
        def __init__(self, data: bytes):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return _FakeResp(json.dumps({
            "model": "Qwen/Qwen3.8-27B",
            "choices": [{"message": {"content": '{"ok"'}}],
        }).encode())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    served = probe_chat("http://127.0.0.1:9/v1/", model="Qwen/Qwen3.8-27B",
                        timeout=5.0)
    assert served == "Qwen/Qwen3.8-27B"      # server-echoed serving id
    assert seen["url"].endswith("/chat/completions")  # trailing slash stripped
    body = seen["body"]
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["max_tokens"] == 1
    assert body["response_format"] == {"type": "json_object"}


def test_probe_chat_stages_http_400_detail(monkeypatch):
    import io
    import urllib.error as urllib_error

    import harness.diagnosis.llm as llm_mod

    def refuse(request, timeout):
        raise urllib_error.HTTPError(
            "http://127.0.0.1:9/v1/chat/completions", 400, "Bad Request",
            {}, io.BytesIO(
                b'{"error":{"message":"No user query found in messages."}}'))

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", refuse)
    with pytest.raises(LLMError, match="No user query found in messages"):
        probe_chat("http://127.0.0.1:9/v1", model="m")


def test_probe_chat_connection_reset_is_staged(monkeypatch):
    import harness.diagnosis.llm as llm_mod

    def reset_during_read(request, timeout):
        raise ConnectionResetError(10054, "forcibly closed by the remote host")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", reset_during_read)
    with pytest.raises(LLMError, match="LLM connection failed"):
        probe_chat("http://127.0.0.1:9/v1", model="m")


def test_probe_chat_unreachable_raises_llm_error():
    with pytest.raises(LLMError):
        probe_chat("http://127.0.0.1:1/v1", model="m", timeout=1.0)


# ---- generation timeouts: staged, actionable, never "connection failed" ----

def test_chat_read_timeout_is_staged_with_actionable_hint(monkeypatch):
    """A mid-generation read timeout (local model over a tunnel, socket idle
    while the model produces) must state the budget and the fix -- not look
    like a dead connection (``LLM connection failed: timed out Timeout
    error``)."""
    import urllib.error as urllib_error

    import harness.diagnosis.llm as llm_mod

    monkeypatch.delenv("HARNESS_LLM_TIMEOUT", raising=False)
    llm = LocalLLM(url="http://127.0.0.1:9/v1", timeout=600.0)

    def stall_mid_generation(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", stall_mid_generation)
    with pytest.raises(LLMError, match="timed out after 600s"):
        llm.chat_json([{"role": "user", "content": "hi"}])
    with pytest.raises(LLMError, match="HARNESS_LLM_TIMEOUT"):
        llm.chat_json([{"role": "user", "content": "hi"}])

    def stall_during_connect(request, timeout):
        # connect-phase timeouts arrive wrapped by urllib
        raise urllib_error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", stall_during_connect)
    with pytest.raises(LLMError, match="timed out after 600s"):
        llm.chat_json([{"role": "user", "content": "hi"}])
