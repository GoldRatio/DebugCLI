"""Conversation agent: ChatTurn decisions, evidence digest, message building."""

from harness.diagnosis.schema import Action, Diagnosis, FailurePoint, Risk, ServerState
from harness.operator.chat_agent import (
    CHAT_CONTRACT,
    CHAT_SYSTEM,
    ChatTurn,
    build_evidence,
    build_messages,
    decide,
    fallback_turn,
)
from harness.operator.router import _keyword_route


class _ChatLLM:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    def chat_json(self, messages):
        self.messages = messages
        return self.payload


def _diag():
    return Diagnosis(
        state=ServerState.FAULT,
        diagnosis="Memory controller logged a corrected MCA on DIMM_A2.",
        confidence=0.74,
        subsystems_considered=["memory"],
        actions=[Action(step=1, action="Reseat DIMM_A2",
                        rationale="Server_Arch_v2.3.pdf p. 78", risk=Risk.LOW,
                        required_tool="ipmitool", impact="requires reboot")],
        references=[],
        evidence=[{
            "mnemonic": "IA32_MC0_STATUS", "raw_hex": "0x8000000000000001",
            "decoded_fields": [{"name": "over", "raw_value": "1",
                                "meaning": "overflow"}], "unknown": False,
        }],
        unknown_registers=["CPLD_1D_RUN_FAULTS"],
    )


# ---- decide ----

def test_decide_accepts_chat_turn():
    llm = _ChatLLM({"say": "Re-checking ECC counters.",
                    "tool": "diagnose", "symptom": "ECC errors", "host": "h1"})
    turn = decide(llm, [{"role": "user", "content": "ECC errors"}],
                  host_names=("h1", "h2"))
    assert turn is not None
    assert turn.tool == "diagnose"
    assert turn.symptom == "ECC errors"
    assert turn.host == "h1"
    assert llm.messages == [{"role": "user", "content": "ECC errors"}]  # passthrough


def test_decide_rejects_old_router_format():
    # The pre-agent router format {"intent": ...} has no "tool": it must NOT
    # silently become a no-op turn; the caller falls back to keywords.
    llm = _ChatLLM({"intent": "reply", "text": "hi"})
    assert decide(llm, []) is None


def test_decide_rejects_garbage():
    for payload in (["not", "a", "dict"], {"tool": 42}, None, "nope"):
        assert decide(_ChatLLM(payload), []) is None


def test_decide_rejects_semantic_violations():
    assert decide(_ChatLLM({"say": "x", "tool": "diagnose"}), []) is None
    assert decide(_ChatLLM({"say": "x", "tool": "docs"}), []) is None
    assert decide(_ChatLLM({"say": "x", "tool": "probe"}), []) is None
    assert decide(_ChatLLM({"say": "x", "tool": "diagnose", "symptom": "  "}),
                  []) is None


def test_decide_llm_exception_falls_back_to_none():
    class Boom:
        def chat_json(self, messages):
            raise RuntimeError("llm down")

    assert decide(Boom(), []) is None


def test_decide_unknown_host_falls_back_to_active():
    llm = _ChatLLM({"say": "x", "tool": "probe", "subsystems": ["memory"],
                    "host": "nope"})
    turn = decide(llm, [], host_names=("h1",))
    assert turn is not None
    assert turn.host is None


def test_decide_extracts_target_from_symptom():
    llm = _ChatLLM({"say": "x", "tool": "diagnose",
                    "symptom": "ECC errors on Q61 cable 8"})
    turn = decide(llm, [], host_names=())
    assert turn.rack == "Q61"
    assert turn.cable == "8"
    assert "Q61" not in turn.symptom


def test_decide_accepts_run_turn():
    llm = _ChatLLM({"say": "Loading that run.",
                    "tool": "run", "run": "harness_runs/abc"})
    turn = decide(llm, [])
    assert turn is not None
    assert turn.tool == "run"
    assert turn.run == "harness_runs/abc"


def test_decide_rejects_run_without_reference():
    assert decide(_ChatLLM({"say": "x", "tool": "run"}), []) is None
    assert decide(_ChatLLM({"say": "x", "tool": "run", "run": "  "}), []) is None


def test_chat_contract_documents_run_and_crisp_symptoms():
    assert '"run"' in CHAT_CONTRACT
    assert "run (required" in CHAT_CONTRACT
    assert "crisp symptom" in CHAT_CONTRACT
    assert "run id" in CHAT_CONTRACT


# ---- fallback_turn ----

def test_fallback_turn_diagnose_carries_target():
    cmd = _keyword_route("Diagnose the server in Q61 Cable 8")
    turn = fallback_turn(cmd, cmd.symptom or "")
    assert turn.tool == "diagnose"
    assert turn.rack == "Q61" and turn.cable == "8"


def test_fallback_turn_probe_docs_verify():
    probe = fallback_turn(_keyword_route("probe the memory controller"), "")
    assert probe.tool == "probe" and probe.subsystems == ["memory"]

    docs = fallback_turn(_keyword_route("look up DIMM in the manual"), "")
    assert docs.tool == "docs" and "DIMM" in docs.query

    verify = fallback_turn(_keyword_route("verify the counter"), "")
    assert verify.tool == "verify" and verify.metric == "ecc"


def test_fallback_turn_reply_is_conversational():
    turn = fallback_turn(_keyword_route("hello there"), "hello there")
    assert turn.tool == "none"
    assert "describe a symptom" in turn.say


# ---- build_messages ----

def test_build_messages_structure():
    messages = build_messages(
        transcript=[
            {"role": "user", "kind": "message", "content": "diagnose it"},
            {"role": "agent", "kind": "say", "content": "checking"},
            {"role": "tool", "kind": "result", "content": "decoded registers"},
        ],
        user_text="diagnose it",
        evidence_digest="## Latest Run Evidence\nstate: fault",
        host_names=("h1",),
        target_label="h1",
        pending=["Replaced PSU last week"],
    )
    assert messages[0]["role"] == "system"
    assert CHAT_SYSTEM in messages[0]["content"]
    assert messages[1]["content"] == "diagnose it"           # user message verbatim
    assert messages[2]["role"] == "assistant"                # agent say
    assert messages[2]["content"] == "checking"
    assert messages[3]["content"] == "[tool result] decoded registers"
    final = messages[-1]["content"]
    assert "## Latest Run Evidence" in final
    assert "Replaced PSU last week" in final
    assert "## Active Target\nh1" in final
    assert "## Available Hosts\nh1" in final
    assert final.endswith(CHAT_CONTRACT)


def test_build_messages_empty_evidence_placeholder():
    messages = build_messages(transcript=[], user_text="hi", evidence_digest="")
    assert "no diagnosis run yet" in messages[-1]["content"]


def test_build_messages_run_reference_hint():
    messages = build_messages(
        transcript=[], user_text="is the run on harness_runs\\bcf28c2bc94448919445af3b2e66fdcc "
                                 "related to a bianca error?",
        evidence_digest="")
    final = messages[-1]["content"]
    assert "## Referenced Runs" in final
    assert "bcf28c2bc94448919445af3b2e66fdcc" in final
    assert "use the 'run' tool" in final.lower()


def test_build_messages_bare_run_id_hint_and_no_false_positive():
    messages = build_messages(
        transcript=[], user_text="what did 0deadbeefdeadbeefdeadbeefdeadbeef find?",
        evidence_digest="")
    assert "## Referenced Runs" in messages[-1]["content"]

    messages = build_messages(transcript=[], user_text="probe the memory controller",
                              evidence_digest="")
    assert "## Referenced Runs" not in messages[-1]["content"]


# ---- build_evidence ----

def test_build_evidence_covers_diagnosis():
    digest = build_evidence(_diag(), run_dir="/runs/abc")
    assert "## Latest Run Evidence" in digest
    assert "/runs/abc" in digest
    assert "state: fault" in digest
    assert "confidence: 0.74" in digest
    assert "Memory controller" in digest
    assert "Reseat DIMM_A2" in digest
    assert "risk: low" in digest
    assert "IA32_MC0_STATUS" in digest
    assert "over=1(overflow)" in digest
    assert "CPLD_1D_RUN_FAULTS" in digest


def test_build_evidence_failure_point():
    diag = _diag()
    diag.failure_point = FailurePoint(rail_tokens="RUN", reasons=["run_pwrup_flt"],
                                      suspects=["DIMM_A2", "SWB CPLD"])
    digest = build_evidence(diag)
    assert "failure point" in digest
    assert "suspects=DIMM_A2, SWB CPLD" in digest


def test_chat_turn_round_trips_model_validate():
    raw = {"say": "probing", "tool": "probe", "subsystems": ["memory"],
           "doc_topics": ["boot state"]}
    turn = ChatTurn.model_validate(raw)
    assert turn.tool == "probe"
    assert turn.subsystems == ["memory"]
    assert turn.say == "probing"