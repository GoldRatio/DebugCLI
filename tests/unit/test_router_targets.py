"""Keyword fallback routing with dynamic targets (rack/cable/IP/alias)."""


from harness.operator.router import (
    _keyword_route,
    extract_target,
)

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


def test_keyword_route_previous_run_followup_diagnoses():
    # "It didn't give explicit repair actions" must NOT become a chat reply:
    # the operator wants a fresh read-only diagnosis with concrete actions.
    cmd = _keyword_route(
        "There was a previous run on this server that was .74 percent "
        "confidence. It didn't give explicit repair actions of what should "
        "be replaced though")
    assert cmd is not None
    assert cmd.intent == "diagnose"
    assert "previous run" in cmd.symptom


def test_keyword_route_low_confidence_diagnoses():
    cmd = _keyword_route("The last diagnosis was only 0.74 confidence")
    assert cmd.intent == "diagnose"


def test_extract_target_empty_text():
    cleaned, target = extract_target("")
    assert cleaned == ""
    assert target == {}
