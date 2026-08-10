"""LLM adapters: schema validation, stub behavior, and error surfacing."""

import pytest

from harness.diagnosis.llm import GeminiLLM, LLMError, OpenAICompatLLM, StubLLM
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


def test_gemini_system_only_request_gets_user_turn(monkeypatch):
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
        seen["messages"] = json.loads(request.data)["messages"]
        return _FakeResp(json.dumps({
            "choices": [{"message": {"content": '{"ok": true}'}}],
        }).encode())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    llm = GeminiLLM(url="http://127.0.0.1:9/v1", api_key="k", timeout=1.0)
    assert llm.chat_json([{"role": "system", "content": "diagnose"}]) == {"ok": True}
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]


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
