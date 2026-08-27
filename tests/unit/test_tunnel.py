"""Rack-manager LLM forward: staged failures, key cleanup, pump plumbing."""

import queue
import socket
from pathlib import Path

import pytest

import harness.engine.tunnel as tunnel_mod
from harness.config.models import ConsoleDomain
from harness.engine.tunnel import LLMForward, TunnelError, parse_tunnel_spec

# ---- spec parsing ----

def test_parse_tunnel_spec_valid_and_invalid():
    assert parse_tunnel_spec("10.0.0.42:8000") == ("10.0.0.42", 8000)
    assert parse_tunnel_spec(" host-7.lab:9001 ") == ("host-7.lab", 9001)
    for bad in ("10.0.0.42", "10.0.0.42:http", ":8000", "10.0.0.42:99999",
                "10.0.0.42:0x1f90", "host -7.lab:9001"):
        with pytest.raises(ValueError):
            parse_tunnel_spec(bad)


# ---- fixtures ----

def _domain(tmp_path: Path, trust: str = "lab") -> ConsoleDomain:
    return ConsoleDomain(
        address="192.168.202.51", user="log",
        identity_vault_path="secret/harness/rackmgr/id_ed25519",
        known_hosts_path=str(tmp_path / "rackmgr_known_hosts"),  # absent: fail closed
        rack="03", cable="12", trust_level=trust,
    )


class _FakeTransport:
    def __init__(self, client, channel_error=None, channel=None):
        self._client = client
        self._channel_error = channel_error
        self._channel = channel

    def open_channel(self, kind, dest, src, timeout=None):
        assert kind == "direct-tcpip"
        if self._channel_error is not None:
            raise self._channel_error
        return self._channel


class _FakeChannel:
    """Loopback channel: sendall feeds recv (echoes through the tunnel)."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self.closed = False

    def recv(self, n):
        try:
            return self._q.get(timeout=2.0)
        except queue.Empty:
            return b""  # EOF-ish: ends the pipe loop deterministically

    def sendall(self, data):
        self._q.put(data)

    def shutdown_write(self):
        pass

    def close(self):
        self.closed = True


def _install_fakes(monkeypatch, *, auth_error=None, channel_error=None,
                   channel=None) -> dict:
    """Swap paramiko.SSHClient + load_key_material inside the tunnel module."""
    seen: dict = {}

    class _FakeClient:
        def __init__(self):
            self._transport = _FakeTransport(self, channel_error, channel)
            self.closed = False
            seen["client"] = self

        def set_missing_host_key_policy(self, policy):
            seen["policy"] = policy

        def load_host_keys(self, path):
            seen["host_keys"] = path

        def connect(self, hostname, username, key_filename,
                    look_for_keys, allow_agent, timeout):
            if auth_error is not None:
                raise auth_error
            seen["connect"] = (hostname, username, key_filename)

        def get_transport(self):
            return self._transport

        def close(self):
            self.closed = True

    monkeypatch.setattr(tunnel_mod.paramiko, "SSHClient", _FakeClient)

    def _fake_key(store, vault_path, tmp_dir):
        path = Path(tmp_dir) / "fake_key.pem"
        path.write_bytes(b"key-material")
        seen["key_path"] = path
        return path

    monkeypatch.setattr(tunnel_mod, "load_key_material", _fake_key)
    return seen


# ---- lifecycle ----

def test_forward_rejected_at_prod_trust_level(tmp_path):
    with pytest.raises(TunnelError) as exc:
        LLMForward("10.0.0.42", 8000, _domain(tmp_path, trust="prod"), None)
    assert exc.value.stage == "trust"


def test_auth_failure_is_staged_and_key_material_cleaned(tmp_path, monkeypatch):
    err = tunnel_mod.paramiko.SSHException("auth boom")
    seen = _install_fakes(monkeypatch, auth_error=err)
    fwd = LLMForward("10.0.0.42", 8000, _domain(tmp_path), None,
                     tmp_dir=tmp_path)
    with pytest.raises(TunnelError) as exc:
        fwd.start()
    assert exc.value.stage == "auth"
    assert "auth boom" in str(exc.value)
    assert not seen["key_path"].exists()      # materialized key always unlinked
    assert fwd.url is None


def test_start_binds_local_url_and_context_manager_closes(tmp_path, monkeypatch):
    _install_fakes(monkeypatch)
    fwd = LLMForward("10.0.0.42", 8000, _domain(tmp_path), None,
                     tmp_dir=tmp_path)
    with fwd as live:
        assert live.url.startswith("http://127.0.0.1:")
        assert live.url.endswith("/v1")
        host, port = live._server.getsockname()
        assert (host, port) == ("127.0.0.1", int(live.url.split(":")[2].split("/")[0]))
    assert fwd._server is None                # listener torn down
    fwd.close()                               # idempotent


# ---- data path ----

def _deadline_poll(predicate, timeout=5.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_pump_carries_bytes_both_ways(tmp_path, monkeypatch):
    channel = _FakeChannel()
    _install_fakes(monkeypatch, channel=channel)
    with LLMForward("10.0.0.42", 8000, _domain(tmp_path), None,
                    tmp_dir=tmp_path) as fwd:
        port = int(fwd.url.split(":")[2].split("/")[0])
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as sock:
            sock.settimeout(2.0)
            sock.sendall(b"PING")
            assert sock.recv(65536) == b"PING"          # echoed by the fake channel
        assert _deadline_poll(lambda: channel.closed)   # pair closed after EOF


def test_forward_refusal_recorded_and_connection_dropped(tmp_path, monkeypatch):
    refusal = tunnel_mod.paramiko.SSHException("forwarding disabled")
    _install_fakes(monkeypatch, channel_error=refusal)
    fwd = LLMForward("10.0.0.42", 8000, _domain(tmp_path), None,
                     tmp_dir=tmp_path)
    url = fwd.start()                          # listener still comes up
    try:
        port = int(url.split(":")[2].split("/")[0])
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as sock:
            sock.settimeout(2.0)
            assert sock.recv(65536) == b""     # refused -> immediate EOF
        assert _deadline_poll(lambda: fwd.forward_error is not None)
        assert "forwarding disabled" in fwd.forward_error
    finally:
        fwd.close()
