"""Rack-level Redfish debug-step collection: client surface, transport
fallback, collector staging, and pipeline wiring.

The Redfish path is the jumpin-free evidence source: GETs straight to the
rack manager (event logs, service conditions) with no serial session. These
tests pin the safety envelope (GET-only, vaulted credential, trust gate) and
the honest-failure behavior of the collector.
"""

import base64
import http.client
import json

import pytest

from harness.config.models import ConsoleDomain
from harness.config.vault import MemorySecretStore
from harness.diagnosis.engine import DiagnosticEngine, EngineContext
from harness.diagnosis.schema import Action, Diagnosis, Reference, Risk
from harness.engine.allowlist import AllowPolicy, AllowRule
from harness.engine.redfish import RedfishClient, RedfishError, RedfishGet
from harness.engine.runner import Runner
from harness.inspect.collectors.redfish import RedfishCollector, _sort_event_log
from harness.inspect.decoder import Decoder
from harness.inspect.registry import make_collector

REDFISH_SECRET = "secret/harness/rackmgr/redfish"

# Minimal runner policy: model detection runs `/bin/dmidecode` first and must
# not raise; everything else the engine plans is skipped via the factory.
FAKE_POLICY = AllowPolicy([AllowRule("/bin/dmidecode", ())])


# ---- fixtures ----

def _domain(tmp_path, trust: str = "lab", rack: str = "61", cable: str = "8",
            **kw) -> ConsoleDomain:
    return ConsoleDomain(
        address="10.0.128.74", user="log",
        identity_vault_path="secret/harness/rackmgr/id_ed25519",
        known_hosts_path=str(tmp_path / "kh"),
        rack=rack, cable=cable, trust_level=trust,
        redfish_user="root",
        redfish_password_vault_path=REDFISH_SECRET,
        **kw)


class _FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def read(self, n=-1):
        if n is None or n < 0:
            return self._body
        return self._body[:n]


@pytest.fixture
def fake_http(monkeypatch):
    """Swap HTTPSConnection inside http.client; returns (conns, spec).

    Connections are created lazily inside ``get``, so tests script the canned
    behavior through ``spec`` BEFORE issuing the request, then assert on the
    captured ``conns`` positionally. ``spec["fail"]`` is either True (every
    conn fails) or an int (that many conns fail before one succeeds).
    """
    conns: list = []
    spec = {"fail": False, "status": 200, "body": b"{}"}

    def _next_fail():
        remaining = spec["fail"]
        if isinstance(remaining, bool):
            return remaining
        spec["fail"] = max(0, remaining - 1)
        return remaining > 0

    class Conn:
        def __init__(self, host, port, timeout=None, context=None):
            self.host, self.port = host, port
            self.fail = _next_fail()
            self.status = spec["status"]
            self.body = spec["body"]
            self.method = self.uri = self.headers = None
            conns.append(self)

        def request(self, method, uri, headers=None):
            self.method, self.uri, self.headers = method, uri, headers or {}
            if self.fail:
                raise ConnectionRefusedError("refused")

        def getresponse(self):
            return _FakeResponse(self.status, self.body)

        def close(self):
            pass

    monkeypatch.setattr(http.client, "HTTPSConnection", Conn)
    return conns, spec


def _client(tmp_path, store=None, **kw) -> RedfishClient:
    store = store or MemorySecretStore({REDFISH_SECRET: b"$pl3nd1D"})
    return RedfishClient(_domain(tmp_path), "8", store, **kw)


# ---- URL surface (derived, never configured) ----

def test_base_url_derived_from_rack_and_cable(tmp_path):
    assert _client(tmp_path).base_url == \
        "https://10.0.128.74/8/amc1/redfish/v1"


def test_base_url_honors_per_rack_address_map(tmp_path):
    domain = _domain(tmp_path, rack="61",
                     rack_addresses={"Q61": "10.0.128.99"})
    client = RedfishClient(domain, "8", MemorySecretStore({REDFISH_SECRET: b"x"}))
    assert client.base_url == "https://10.0.128.99/8/amc1/redfish/v1"


# ---- safety envelope ----

def test_client_surface_is_get_only(tmp_path):
    client = _client(tmp_path)
    for verb in ("post", "put", "patch", "delete", "head"):
        assert not hasattr(client, verb), verb


@pytest.mark.parametrize("path", [
    "ServiceConditions",        # no leading slash
    "../etc/passwd",            # traversal
    "/a/../b",
    "/ServiceConditions?expand",  # query strings not expressible
    "/Service Conditions",
    "/x%2e%2e",
])
def test_unsafe_paths_rejected(tmp_path, path):
    with pytest.raises(RedfishError) as exc:
        _client(tmp_path).get(path)
    assert exc.value.stage == "path"


def test_trust_gate_blocks_prod(tmp_path):
    prod = _domain(tmp_path, trust="prod")
    with pytest.raises(RedfishError) as exc:
        RedfishClient(prod, "8", MemorySecretStore({REDFISH_SECRET: b"x"}))
    assert exc.value.stage == "trust"


def test_password_resolved_from_vault(tmp_path, fake_http):
    conns, _spec = fake_http
    _client(tmp_path).get("/ServiceConditions")
    expected = "Basic " + base64.b64encode(b"root:$pl3nd1D").decode()
    assert conns[0].headers["Authorization"] == expected


def test_missing_vault_secret_stages_auth_error(tmp_path):
    with pytest.raises(RedfishError) as exc:
        _client(tmp_path, store=MemorySecretStore()).get("/ServiceConditions")
    assert exc.value.stage == "auth"


def test_invalid_cable_rejected(tmp_path):
    with pytest.raises(RedfishError):
        RedfishClient(_domain(tmp_path), "8; rm -rf /",
                      MemorySecretStore({REDFISH_SECRET: b"x"}))


# ---- transport: direct first, tunnel on connect failure ----

def test_direct_transport_served_first(tmp_path, fake_http):
    conns, spec = fake_http
    spec["body"] = json.dumps({"ServiceConditions": "ok"}).encode()
    result = _client(tmp_path).get("/ServiceConditions")
    assert result.transport == "direct"
    assert result.status == 200
    assert conns[0].uri == "/8/amc1/redfish/v1/ServiceConditions"
    assert conns[0].host == "10.0.128.74" and conns[0].port == 443


def test_preflight_gets_service_root(tmp_path, fake_http):
    conns, _spec = fake_http
    _client(tmp_path).preflight()
    assert conns[0].method == "GET"
    assert conns[0].uri == "/8/amc1/redfish/v1"


def test_tunnel_fallback_when_direct_unreachable(tmp_path, fake_http):
    conns, spec = fake_http
    spec["fail"] = 1                                # direct fails once, tunnel ok
    closed = []

    class FakeForward:
        def start(self):
            return "http://127.0.0.1:45678/v1"

        def close(self):
            closed.append(True)

    client = _client(tmp_path, forward_factory=lambda h, p: FakeForward())
    result = client.get("/ServiceConditions")
    assert result.transport == "tunnel"
    assert result.status == 200
    assert conns[1].host == "127.0.0.1" and conns[1].port == 45678
    assert conns[1].headers["Host"] == "10.0.128.74"  # service sees the rack ip
    client.close()
    assert closed == [True]


def test_tunnel_refusal_stages_error(tmp_path, fake_http):
    _conns, spec = fake_http
    spec["fail"] = True

    class RefusingForward:
        def start(self):
            raise RuntimeError("ssh channel refused")

    client = _client(tmp_path, forward_factory=lambda h, p: RefusingForward())
    with pytest.raises(RedfishError) as exc:
        client.get("/ServiceConditions")
    assert exc.value.stage == "forward"


def test_body_truncation_recorded(tmp_path, fake_http):
    _conns, spec = fake_http
    spec["body"] = b"x" * 4096
    client = _client(tmp_path, max_body=1024)
    result = client.get("/ServiceConditions")
    assert result.truncated is True
    assert len(result.body) == 1024


def test_http_error_status_is_data_not_exception(tmp_path, fake_http):
    _conns, spec = fake_http
    spec["status"] = 404
    result = _client(tmp_path).get("/ServiceConditions")
    assert result.status == 404
    assert result.transport == "direct"


# ---- collector ----

class _FakeClient:
    """Scripted RedfishClient: path -> RedfishGet or exception; unscripted
    paths return a healthy empty service response."""

    def __init__(self, preflight, gets):
        self._preflight = preflight
        self._gets = gets

    def preflight(self):
        if isinstance(self._preflight, Exception):
            raise self._preflight
        return self._preflight

    def get(self, path):
        item = self._gets.get(path, _get())
        if isinstance(item, Exception):
            raise item
        return item


def _get(status=200, body="{}", transport="direct") -> RedfishGet:
    return RedfishGet(status=status, body=body, elapsed_ms=12,
                      transport=transport)


def _event_log(*created: str) -> str:
    members = [{"Id": str(i), "Created": c, "Severity": "Warning",
                "Message": f"event {i}"}
               for i, c in enumerate(created)]
    return json.dumps({"Members": members})


def test_collector_collects_preflight_and_endpoints_sorted():
    client = _FakeClient(
        preflight=_get(),
        gets={"/Systems/HGX_Baseboard_0/LogServices/EventLog/Entries":
              _get(body=_event_log("2026-08-02T10:00:00Z",
                                   "2026-08-01T09:00:00Z")),
              "/ServiceConditions": _get(body='{"Health": "Critical"}')})
    dumps = RedfishCollector(client).collect()
    assert [d.source for d in dumps] == [
        "GET <preflight>",
        "GET /Systems/HGX_Baseboard_0/LogServices/EventLog/Entries",
        "GET /ServiceConditions",
    ]
    assert all(d.ok for d in dumps)
    assert all(d.subsystem == "redfish" for d in dumps)
    event = json.loads(dumps[1].raw)
    created = [m["Created"] for m in event["Members"]]
    assert created == sorted(created)          # get_event_logs parity
    assert dumps[1].meta["kind"] == "event_log"
    assert dumps[2].meta == {"kind": "service_conditions", "status": 200,
                             "elapsed_ms": 12, "transport": "direct",
                             "truncated": False}


def test_collector_stages_preflight_failure():
    client = _FakeClient(preflight=RedfishError("connect", "unreachable"),
                         gets={})
    dumps = RedfishCollector(client).collect()
    assert len(dumps) == 1
    assert dumps[0].ok is False
    assert "connect" in dumps[0].raw


def test_collector_stages_endpoint_failure_but_continues():
    path = "/Systems/HGX_Baseboard_0/LogServices/EventLog/Entries"
    client = _FakeClient(
        preflight=_get(),
        gets={path: RedfishError("http", "boom"),
              "/ServiceConditions": _get(body='{"Health": "Ok"}')})
    dumps = RedfishCollector(client).collect()
    assert len(dumps) == 3
    assert dumps[1].ok is False and dumps[1].meta["status"] is None
    assert dumps[2].ok is True


def test_collector_records_http_error_status():
    client = _FakeClient(preflight=_get(),
                         gets={"/Systems/HGX_Baseboard_0/LogServices/EventLog/Entries":
                               _get(body=_event_log("2026-08-01T09:00:00Z")),
                               "/ServiceConditions": _get(status=503,
                                                          body="busy")})
    dumps = RedfishCollector(client).collect()
    assert dumps[-1].ok is False
    assert dumps[-1].meta["status"] == 503
    assert dumps[-1].raw == "busy"


def test_sort_event_log_passthrough_on_unexpected_shapes():
    for body in ("not json", "[]", json.dumps({"Members": "nope"})):
        assert _sort_event_log(body) == body

# ---- pipeline wiring ----

class _FakeRunner(Runner):
    """Canned dmidecode so model detection resolves cleanly; nothing else runs."""

    def __init__(self):
        super().__init__(FAKE_POLICY)

    def _exec(self, argv, timeout=30.0):
        from harness.engine.runner import CommandResult
        out = ("Product Name: model_x\nBIOS Vendor: Intel\nBIOS Version: 2.3\n"
               if argv == ["/bin/dmidecode"] else "")
        return CommandResult(argv=argv, stdout=out, stderr="",
                             exit_code=0, elapsed_ms=1)


def test_engine_appends_extra_redfish_collector_dumps():
    """extra_collectors rides beside the classified plan; redfish evidence
    lands in dump_sets even though no subsystem classification names it."""
    seen: dict = {}

    def factory(name, _runner):
        if name == "redfish":
            return RedfishCollector(_FakeClient(
                preflight=_get(),
                gets={"/ServiceConditions": _get(
                    body='{"Health": "Critical"}')}))
        if name in ("cpu_msr", "kernel", "pcie", "storage", "ipmi"):
            return None
        return make_collector(name, _runner)

    def scorer(d, dump_sets):
        seen.update(dump_sets)
        return d

    engine = DiagnosticEngine(EngineContext(
        runner=_FakeRunner(),
        decoder=Decoder(),
        collector_factory=factory,
        llm=_fake_llm,
        scorer=scorer,
        extra_collectors=("redfish",),
    ))
    engine.run("MCE uncorrectable ECC error")
    redfish = seen.get("redfish")
    assert redfish is not None
    assert any("ServiceConditions" in d.source for d in redfish)


def test_engine_without_extra_collectors_is_unchanged():
    """No redfish configured -> factory never asked for it (byte-identical
    behavior for inventories without the redfish block)."""
    asked: list[str] = []

    def factory(name, _runner):
        asked.append(name)

    engine = DiagnosticEngine(EngineContext(
        runner=_FakeRunner(),
        decoder=Decoder(),
        collector_factory=factory,
        llm=_fake_llm,
        scorer=lambda d, ds: d,
    ))
    engine.run("MCE uncorrectable ECC error")
    assert "redfish" not in asked


# ---- cli helper ----

def test_build_redfish_collector_requires_vault_path(tmp_path):
    from harness.operator.cli import _build_redfish_collector
    domain = ConsoleDomain(
        address="10.0.128.74", user="log",
        identity_vault_path="secret/harness/rackmgr/id_ed25519",
        known_hosts_path=str(tmp_path / "kh"),
        rack="61", cable="8", trust_level="lab",
    )
    notes: list[str] = []
    assert _build_redfish_collector(domain, MemorySecretStore(), [],
                                    notes.append) is None
    assert notes == []


def test_build_redfish_collector_missing_secret_stages_note(tmp_path):
    from harness.operator.cli import _build_redfish_collector
    notes: list[str] = []
    secrets: list[str] = []
    got = _build_redfish_collector(_domain(tmp_path), MemorySecretStore(),
                                   secrets, notes.append)
    assert got is None
    assert notes and "missing from vault" in notes[0]
    assert secrets == []


def test_build_redfish_collector_wired_with_secret(tmp_path):
    from harness.operator.cli import _build_redfish_collector
    secrets: list[str] = []
    collector = _build_redfish_collector(
        _domain(tmp_path), MemorySecretStore({REDFISH_SECRET: b"$pl3nd1D"}),
        secrets, lambda _msg: None)
    assert isinstance(collector, RedfishCollector)
    assert secrets == ["$pl3nd1D"]          # password joins the redaction set
    assert collector.client.base_url.endswith("/8/amc1/redfish/v1")


def _fake_llm(_prompt: str) -> Diagnosis:
    return Diagnosis(
        diagnosis="Memory ECC errors on DIMM_A2",
        confidence=0.0,
        actions=[Action(
            step=1,
            action="Reseat DIMM in slot A2",
            rationale="Memory ECC uncorrectable errors; architecture doc page 78",
            risk=Risk.LOW,
            required_tool="Physical access",
            impact="requires reboot",
            references=[],
        )],
        references=[Reference(source="Server_Arch_v2.3.pdf", page="78")],
    )
