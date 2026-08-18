"""Model provenance: canonicalization, tri-state sources, engine fallback chain."""



from harness.diagnosis.engine import DiagnosticEngine, EngineContext
from harness.diagnosis.llm import StubLLM
from harness.diagnosis.prompt import SYSTEM_PREAMBLE, build_prompt
from harness.diagnosis.summarize import EvidenceSummary
from harness.engine.allowlist import AllowPolicy, AllowRule
from harness.engine.runner import Runner
from harness.inspect.decoder import Decoder
from harness.inspect.model import (
    MODEL_ALIASES,
    DetectedModel,
    from_alias,
    from_operator,
    normalize_product,
)

_DMIDECODE_OK = (
    "Product Name: PowerEdge R750\n"
    "BIOS Vendor: Dell\nBIOS Version: 2.3\n"
)


def _dmi_policy() -> AllowPolicy:
    return AllowPolicy([AllowRule("/bin/dmidecode", ())])


def test_normalize_product_resolves_alias_variants():
    assert normalize_product("PowerEdge R650") == "poweredge_r650"
    assert normalize_product("System R650 Server") == "poweredge_r650"
    assert normalize_product("poweredge r650") == "poweredge_r650"
    assert normalize_product("PROLIANT DL380 Gen10 (SFF)") == "proliant_dl380g10"


def test_normalize_product_resolves_fleet_platforms():
    # Grace-Blackwell fleet: node board / spec names -> canonical platform key,
    # so RAG/topology filters and case-library keying are consistent.
    assert normalize_product("C4A15") == "samoa"
    assert normalize_product("C4A14 Server") == "samoa"
    assert normalize_product("GB200 NVL72") == "nvl72"
    assert normalize_product("NVL72") == "nvl72"
    model = DetectedModel(product_name="C4A15", bios_vendor="Microsoft",
                          bios_version=None, raw="", source="fru")
    assert model.model_key == "samoa"


def test_normalize_product_unknown_passes_clean_slug():
    assert normalize_product("SuperMicro X11DPi-N") == "supermicro_x11dpi-n"


def test_normalize_product_empty_is_none():
    assert normalize_product("") is None
    assert normalize_product("   ") is None
    assert normalize_product("(special)") is None


def test_detected_model_key_uses_canonical_alias():
    model = DetectedModel(product_name="PowerEdge R650", bios_vendor="Dell",
                          bios_version="2.3", raw="", source="dmidecode")
    assert model.model_key == "poweredge_r650"


def test_factories_record_source():
    assert from_operator("SuperMicro").source == "operator"
    assert from_operator("SuperMicro").bios_vendor == "operator"
    assert from_alias("poweredge_r650").source == "alias"
    assert from_alias("poweredge_r650").model_key == "poweredge_r650"


def test_detected_model_source_default_is_unknown():
    model = DetectedModel(product_name="X", bios_vendor="Y",
                          bios_version=None, raw="")
    assert model.source == "unknown"
    assert MODEL_ALIASES  # the curated alias map is non-empty by contract


class _BrokenRunner(Runner):
    """dmidecode runs but fails: no model reachable."""

    def __init__(self) -> None:
        super().__init__(_dmi_policy())

    def _exec(self, argv, timeout=30.0):
        from harness.engine.runner import CommandResult
        return CommandResult(argv=argv, stdout="", stderr="denied",
                             exit_code=1, elapsed_ms=1)


def _ctx(runner=None, **kw) -> EngineContext:
    return EngineContext(
        runner=runner if runner is not None else _BrokenRunner(),
        decoder=Decoder(),
        collector_factory=lambda name, _r: None,
        llm=StubLLM(),
        **kw,
    )


def test_fallback_chain_alias_then_operator_then_none():
    engine = DiagnosticEngine(_ctx(model_hint="poweredge_r650"))
    model, drifted = engine._detect_model()
    assert model is not None and model.source == "alias"
    assert model.model_key == "poweredge_r650"
    assert drifted is False

    asked = []

    def ask() -> str | None:
        asked.append(True)
        return "SuperMicro X11"

    engine = DiagnosticEngine(_ctx(model_ask=ask))
    model, drifted = engine._detect_model()
    assert model is not None and model.source == "operator"
    assert asked

    engine = DiagnosticEngine(_ctx(model_ask=lambda: None))
    model, drifted = engine._detect_model()
    assert model is None and drifted is False


def test_fallback_chain_operator_failure_is_a_skip():
    def ask() -> str:
        raise RuntimeError("no terminal")

    engine = DiagnosticEngine(_ctx(model_ask=ask))
    model, _ = engine._detect_model()
    assert model is None  # a broken question never crashes the run


def test_live_detection_wins_over_alias_and_flags_drift():
    class _HostRunner(Runner):
        def __init__(self) -> None:
            super().__init__(_dmi_policy())

        def _exec(self, argv, timeout=30.0):
            from harness.engine.runner import CommandResult
            return CommandResult(argv=argv, stdout=_DMIDECODE_OK, stderr="",
                                 exit_code=0, elapsed_ms=1)

    engine = DiagnosticEngine(_ctx(runner=_HostRunner(),
                                    model_hint="poweredge_r650"))
    model, drifted = engine._detect_model()
    assert model is not None and model.source == "dmidecode"
    assert model.model_key == "poweredge_r750"
    assert drifted is True


def test_run_passess_detected_model_key_to_retriever_and_hook():
    seen = {}
    hooks = []

    def retriever(query, model_key):
        seen["model_key"] = model_key
        return []

    class _HostRunner(Runner):
        def __init__(self) -> None:
            super().__init__(_dmi_policy())

        def _exec(self, argv, timeout=30.0):
            from harness.engine.runner import CommandResult
            out = ("Product Name: PowerEdge R650\n"
                   "BIOS Vendor: Dell\nBIOS Version: 2.3\n")
            return CommandResult(argv=argv, stdout=out, stderr="",
                                 exit_code=0, elapsed_ms=1)

    engine = DiagnosticEngine(_ctx(
        runner=_HostRunner(),
        docs_retriever=retriever,
        model_hook=lambda m, drifted: hooks.append((m.model_key, drifted)),
    ))
    d = engine.run("MCE ECC")
    assert seen == {"model_key": "poweredge_r650"}
    assert hooks and hooks[0] == ("poweredge_r650", False)
    assert d.diagnosis  # pipeline completed


def test_run_retrieves_with_none_when_no_model():
    seen = {}

    def retriever(query, model_key):
        seen["model_key"] = model_key
        return []

    engine = DiagnosticEngine(_ctx(docs_retriever=retriever,
                                    model_ask=lambda: None))
    engine.run("MCE ECC")
    assert seen == {"model_key": None}


def test_run_passes_alias_model_key_when_detection_fails():
    seen = {}

    def retriever(query, model_key):
        seen["model_key"] = model_key
        return []

    engine = DiagnosticEngine(_ctx(docs_retriever=retriever,
                                    model_hint="poweredge_r650"))
    built = engine.run("MCE ECC")
    assert seen == {"model_key": "poweredge_r650"}
    assert built.diagnosis


def test_prompt_renders_provenance_and_unknown_warning():
    from harness.diagnosis.prompt import build_turn_evidence

    no_model = build_prompt(
        model=None,
        decoded=[],
        summaries=EvidenceSummary(interesting=[], anomaly_count=0, total=0),
        doc_snippets=["(none)"],
        parts_refs=[],
        symptom="ECC",
    )
    assert "model=unknown" in no_model
    assert "model_source=unknown" in no_model
    assert "detection failed" in no_model

    m = from_operator("SuperMicro X11")
    with_model = build_turn_evidence(
        model=m,
        symptom="ECC",
        decoded=[],
        summaries=EvidenceSummary(interesting=[], anomaly_count=0, total=0),
        doc_snippets=[], parts_refs=[], conversation=[],
    )
    assert "model=SuperMicro X11" in with_model
    assert "model_source=operator" in with_model


def test_preamble_warns_against_silent_model_assumptions():
    assert "hardware-detected FACT" in SYSTEM_PREAMBLE