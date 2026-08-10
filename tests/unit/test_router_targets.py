"""Keyword and LLM routing with dynamic targets (rack/cable/IP/alias)."""


from harness.diagnosis.llm import StubLLM
from harness.operator.router import (
    _keyword_route,
    extract_target,
    route_message,
)

_HOSTS = ("h1", "h2")


class _FakeRouterLLM:
    """Router-LLM returning a canned chat_json with rack/cable."""

    def __init__(self, payload: dict):
        self.payload = payload

    def chat_json(self, *, system, messages, schema, temperature=None, max_tokens=None):
        return self.payload


# ---- extract_target ----

def test_extract_rack_cable_from_prose():
    cleaned, target = extract_target("Diagnose the server in Q61 Cable 8")
    assert cleaned == "Diagnose the server in"
    assert target["rack"] == "Q61"
    assert target["cable"] == "8"


def test_extract_rack_cable_variants():
    for text, rack, cable in [
        ("Check q61 cable8", "Q61", "8"),
        ("Check cable 08", None, "8"),
        ("Check cable 12 rack Q 3", "Q3", "12"),
    ]:
        _cleaned, target = extract_target(text)
        assert target.get("rack") == rack and target.get("cable") == cable, text


def test_extract_ip():
    cleaned, target = extract_target("Diagnose 10.0.0.50")
    assert cleaned == "Diagnose"
    assert target["ip"] == "10.0.0.50"


def test_extract_keeps_unrelated_numbers():
    _cleaned, target = extract_target("Memory errors in rack, 3 events, cable problem")
    assert target == {}


# ---- keyword routing ----

def test_keyword_route_captures_target_and_cleans_symptom():
    cmd = _keyword_route("Diagnose the server in Q61 Cable 8")
    assert cmd is not None
    assert cmd.intent == "diagnose"
    assert cmd.symptom == "Diagnose the server in"
    assert cmd.rack == "Q61"
    assert cmd.cable == "8"


def test_keyword_route_plain_diagnose():
    cmd = _keyword_route("Diagnose h1 please")
    assert cmd is not None
    assert cmd.host is None  # keyword router never names hosts
    assert cmd.rack is None


# ---- route_message ----

def test_route_message_keyword_target_is_propagated():
    router_llm = _FakeRouterLLM({
        "intent": "question",
        "question": "Which host do you mean?",
        "symptom": "",
    })
    cmd = route_message(
        text="Diagnose the server in Q61 Cable 8",
        llm=router_llm,
        host_names=_HOSTS,
    )
    assert cmd is not None
    assert cmd.intent == "diagnose"
    assert cmd.symptom == "Diagnose the server in"
    assert cmd.rack == "Q61"
    assert cmd.cable == "8"
    assert cmd.host is None  # dynamic target, not an inventory host


def test_route_message_llm_target_fallback_to_text_extraction():
    # LLM fails to name host; target phrase still extracted from the text.
    router_llm = _FakeRouterLLM({"intent": "docs", "symptom": "Diagnose 10.0.0.50"})
    cmd = route_message(
        text="Diagnose 10.0.0.50",
        llm=router_llm,
        host_names=_HOSTS,
    )
    assert cmd is not None
    assert cmd.ip == "10.0.0.50"


def test_route_message_garbage_returns_reply():
    router_llm = _FakeRouterLLM({"intent": "chat", "symptom": ""})
    cmd = route_message(text="asdf qwerty", llm=router_llm, host_names=_HOSTS)
    assert cmd is not None
    assert cmd.intent == "reply"


def test_route_message_plain_diagnose_uses_active_target():
    router_llm = _FakeRouterLLM({"intent": "question", "question": "?", "symptom": ""})
    cmd = route_message(text="Diagnose h1", llm=router_llm, host_names=_HOSTS)
    assert cmd is not None
    assert cmd.intent == "diagnose"
    assert cmd.host is None
    assert cmd.rack is None


def test_route_message_llm_exception_falls_back_to_keyword():
    class Boom:
        def chat_json(self, **kwargs):
            raise RuntimeError("llm down")

    cmd = route_message(text="Diagnose h2 memory", llm=Boom(), host_names=_HOSTS)
    assert cmd is not None
    assert cmd.intent == "diagnose"
    assert cmd.symptom == "Diagnose h2 memory"


def test_router_llm_backed_by_stub_falls_back_cleanly():
    # StubLLM answers "question" -> model_validate fails -> keyword fallback.
    cmd = route_message(text="Diagnose h1 slow", llm=StubLLM(), host_names=_HOSTS)
    assert cmd is not None
    assert cmd.intent == "diagnose"
    assert cmd.symptom == "Diagnose h1 slow"


def test_extract_target_empty_text():
    cleaned, target = extract_target("")
    assert cleaned == ""
    assert target == {}
