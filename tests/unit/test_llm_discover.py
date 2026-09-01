"""Node-side vLLM endpoint discovery: output parsers, probe batch over the
console, inventory fast path, and honest failure (notes, never raises)."""

import pytest

from harness.config.inventory_lint import load_inventory
from harness.config.vault import MemorySecretStore
from harness.engine.runner import CommandResult
from harness.operator.llm_discover import (
    discover,
    parse_docker_ports,
    parse_listening_ports,
    parse_node_addresses,
)
from harness.targets.resolver import TargetError

_SS = (
    "State   Recv-Q  Send-Q   Local Address:Port   Peer Address:Port\n"
    "LISTEN  0       4096       0.0.0.0:8000        0.0.0.0:*\n"
    "LISTEN  0       128        0.0.0.0:22          0.0.0.0:*\n"
    "LISTEN  0       4096            [::]:8000           [::]:*\n"
)

_DOCKER = (
    "CONTAINER ID   IMAGE        COMMAND                 CREATED    STATUS    "
    "PORTS                                      NAMES\n"
    "abc123         qwen/vllm    \"python -m vllm.ser\"   3 days ago Up 3 days "
    "0.0.0.0:8000->8000/tcp, :::8000->8000/tcp  vllm-qwen\n"
    "def456         redis        \"redis-server\"         3 days ago Up 3 days "
    "                                           redis\n"
)

_HOSTNAME = "10.0.0.42 192.168.202.7 fe80::1%eth0\n"

_CONSOLE_DEFAULTS_INVENTORY = (
    "trust_level: lab\n"
    "console_defaults:\n"
    "  address: 192.168.202.51\n"
    "  user: log\n"
    "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
    "  known_hosts_path: config/rackmgr_known_hosts\n"
    "  tool: jumpin\n"
    "  trust_level: lab\n"
    "  port: 2200\n"
    "  sudo_vault_path: secret/harness/bmc/sudo\n"
    "hosts: []\n"
)

_INVENTORY_WITH_CONSOLE_HOST = (
    "trust_level: lab\n"
    "hosts:\n"
    "  - name: h1\n"
    "    address: 10.0.0.10\n"
    "    model: model_x\n"
    "    ssh:\n"
    "      user: diagbot\n"
    "      identity_vault_path: secret/harness/diagbot/id_ed25519\n"
    "      known_hosts_path: config/known_hosts\n"
    "    bmc:\n"
    "      address: 10.0.0.11\n"
    "      username: bmc-ro\n"
    "      password_vault_path: secret/harness/bmc/bmc-ro\n"
    "    console:\n"
    "      address: 192.168.202.51\n"
    "      user: log\n"
    "      identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
    "      known_hosts_path: config/rackmgr_known_hosts\n"
    "      rack: 03\n"
    "      cable: 12\n"
    "      trust_level: lab\n"
    "      port: 2200\n"
    "      sudo_vault_path: secret/harness/bmc/sudo\n"
)


# ---- parsers ----

def test_parse_listening_ports_dedupes_and_sorts():
    assert parse_listening_ports(_SS) == [22, 8000]


def test_parse_docker_ports_reads_published_mapping():
    assert parse_docker_ports(_DOCKER) == {"vllm-qwen": 8000}


def test_parse_node_addresses_keeps_ipv4_only():
    assert parse_node_addresses(_HOSTNAME) == ["10.0.0.42", "192.168.202.7"]
    assert parse_node_addresses("") == []


# ---- discover() ----

class _FakeRunner:
    def __init__(self, results):
        self.results = results
        self.probes = None

    def batch_execute(self, probes):
        self.probes = list(probes)
        return self.results


def _ok(cmd, stdout):
    return CommandResult(argv=cmd.split(), stdout=stdout, stderr="",
                         exit_code=0, elapsed_ms=1)


def _failed(cmd, stderr):
    return CommandResult(argv=cmd.split(), stdout="", stderr=stderr,
                         exit_code=1, elapsed_ms=1)


def _console_inv(tmp_path, body=_CONSOLE_DEFAULTS_INVENTORY):
    path = tmp_path / "inventory.yaml"
    path.write_text(body, encoding="utf-8")
    return load_inventory(str(path))


def test_discover_parses_probe_batch(tmp_path):
    seen = {}

    class _Recording(_FakeRunner):
        def __init__(self):
            super().__init__([
                _ok("hostname -I", _HOSTNAME),
                _ok("ss -l -t -n", _SS),
                _ok("sudo -S docker ps", _DOCKER),
            ])

        def batch_execute(self, probes):
            seen["probes"] = list(probes)
            return super().batch_execute(probes)

    result = discover("Q61", "8", _console_inv(tmp_path),
                      MemorySecretStore(), console_factory=_Recording)
    assert seen["probes"] == ["hostname -I", "ss -l -t -n", "sudo -S docker ps"]
    assert result.addresses == ["10.0.0.42", "192.168.202.7"]
    assert result.containers == {"vllm-qwen": 8000}
    assert result.suggested_ports() == [8000, 22]
    assert result.notes == []


def test_discover_derives_node_sudo_path_when_unpinned(tmp_path):
    """No sudo_vault_path in the config: discovery layers the per-rack node
    sudo path (the wizard captures that password fresh at every setup), so
    the docker probe still runs and its output parses."""
    body = _CONSOLE_DEFAULTS_INVENTORY.replace(
        "  sudo_vault_path: secret/harness/bmc/sudo\n", "")
    inv = _console_inv(tmp_path, body=body)
    runner = _FakeRunner([
        _ok("hostname -I", _HOSTNAME),
        _ok("ss -l -t -n", _SS),
        _ok("sudo -S docker ps", _DOCKER),
    ])
    result = discover("Q61", "8", inv, MemorySecretStore(),
                      console_factory=lambda: runner)
    assert runner.probes == ["hostname -I", "ss -l -t -n", "sudo -S docker ps"]
    assert result.containers == {"vllm-qwen": 8000}   # docker probe ran + parsed


def test_discover_domain_derivation(tmp_path):
    """The discovery domain derives the per-rack node sudo path when the
    config does not pin one, and honors a pinned path."""
    from harness.operator.llm_discover import _node_sudo_path, discover_domain

    plain = _console_inv(tmp_path, body=_CONSOLE_DEFAULTS_INVENTORY.replace(
        "  sudo_vault_path: secret/harness/bmc/sudo\n", ""))
    domain = discover_domain(plain, "Q71", "8", MemorySecretStore())
    assert domain.sudo_vault_path == _node_sudo_path("Q71")
    assert domain.sudo_vault_path == "secret/harness/llm/node-sudo-71"

    pinned = _console_inv(tmp_path)   # console_defaults carry the BMC sudo
    pinned_domain = discover_domain(pinned, "Q61", "8", MemorySecretStore())
    assert pinned_domain.sudo_vault_path == "secret/harness/bmc/sudo"


def test_discover_node_user_layers_login_credentials(tmp_path):
    """The wizard's captured node user rides the domain (with the derived
    node-sudo vault path) so the login handshake can run at the getty."""
    from harness.operator.llm_discover import (
        _node_sudo_path,
        discover_domain,
    )

    plain = _console_inv(tmp_path, body=_CONSOLE_DEFAULTS_INVENTORY.replace(
        "  sudo_vault_path: secret/harness/bmc/sudo\n", ""))
    domain = discover_domain(plain, "Q71", "8", MemorySecretStore(),
                             node_user="yemankyaw")
    assert domain.node_user == "yemankyaw"
    assert domain.node_password_vault_path == _node_sudo_path("Q71")
    # config-pinned node_user wins over the captured one
    pinned = _console_inv(tmp_path, body=_CONSOLE_DEFAULTS_INVENTORY.replace(
        "  sudo_vault_path: secret/harness/bmc/sudo\n",
        "  sudo_vault_path: secret/harness/bmc/sudo\n"
        "  node_user: otheruser\n"))
    assert discover_domain(pinned, "Q71", "8", MemorySecretStore(),
                           node_user="yemankyaw").node_user == "otheruser"


def test_discover_probe_failures_become_notes(tmp_path):
    runner = _FakeRunner([
        _failed("hostname -I", "not found"),
        _failed("ss -l -t -n", "not found"),
    ])
    result = discover("Q61", "8", _console_inv(tmp_path),
                      MemorySecretStore(), console_factory=lambda: runner)
    assert result.addresses == [] and result.ports == []
    assert "hostname -I: not found" in result.notes
    assert "ss -l -t -n: not found" in result.notes
    assert any("no node addresses" in n for n in result.notes)


def test_discover_console_hop_failure_becomes_note(tmp_path):
    """An unreachable rack manager (e.g. no route from the workstation) is a
    note naming the tool + address -- discovery never crashes the wizard."""

    class _UnreachableRunner:
        def batch_execute(self, probes):
            raise RuntimeError("ssh to rack manager 10.0.128.98 failed: "
                               "[WinError 10060] timed out")

    result = discover("Q71", "8", _console_inv(tmp_path),
                      MemorySecretStore(),
                      console_factory=lambda: _UnreachableRunner())
    assert result.addresses == [] and result.ports == []
    assert len(result.notes) == 1
    assert "10.0.128.98" in result.notes[0]
    assert "tried via jumpin" in result.notes[0]


def test_discover_inventory_fast_path_skips_console(tmp_path):
    inv = _console_inv(tmp_path, body=_INVENTORY_WITH_CONSOLE_HOST)

    def _no_console():
        raise AssertionError("console must not open on the fast path")

    result = discover("Q3", "12", inv, MemorySecretStore(),
                      console_factory=_no_console)  # "3"/"03" spellings vary
    assert result.addresses == ["10.0.0.10"]
    assert any("inventory" in n for n in result.notes)


def test_discover_targeting_errors_raise(tmp_path):
    with pytest.raises(TargetError, match="console_defaults"):
        discover("Q61", "8", load_inventory(str(_write_plain(tmp_path))),
                 MemorySecretStore())


def _write_plain(tmp_path):
    path = tmp_path / "plain.yaml"
    path.write_text("trust_level: lab\nhosts: []\n", encoding="utf-8")
    return path


def test_discover_uses_llm_console_over_console_defaults(tmp_path):
    """The LLM-only block wins for discovery (direct tool, host-SOL port,
    per-rack manager) while the debug console_defaults stay jumpin."""
    inv = _console_inv(tmp_path, body=(
        "trust_level: lab\n"
        "console_defaults:\n"
        "  address: 192.168.202.51\n"
        "  user: log\n"
        "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        "  tool: jumpin\n"
        "  trust_level: lab\n"
        "  port: 2200\n"
        "  sudo_vault_path: secret/harness/bmc/sudo\n"
        "llm_console:\n"
        "  address: 192.168.202.51\n"
        "  user: root\n"
        "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        "  tool: direct\n"
        "  trust_level: lab\n"
        "  port: 22\n"
        "  sudo_vault_path: secret/harness/bmc/sudo\n"
        "  rack_addresses: {Q71: 10.0.128.98}\n"
        "hosts: []\n"))
    seen = {}

    class _Recording(_FakeRunner):
        def __init__(self):
            super().__init__([
                _ok("hostname -I", _HOSTNAME),
                _ok("ss -l -t -n", _SS),
                _ok("sudo -S docker ps", _DOCKER),
            ])

        def batch_execute(self, probes):
            seen["probes"] = list(probes)
            return super().batch_execute(probes)

    result = discover("Q71", "8", inv, MemorySecretStore(),
                      console_factory=_Recording)
    assert result.addresses == ["10.0.0.42", "192.168.202.7"]
    assert result.suggested_ports() == [8000, 22]


def test_llm_console_falls_back_to_console_defaults(tmp_path):
    from harness.operator.llm_discover import (
        llm_bastion_domain,
        llm_console,
        llm_console_domain,
    )

    inv = _console_inv(tmp_path)
    assert llm_console(inv) is inv.console_defaults
    assert llm_bastion_domain(inv) is None     # no bastion configured
    assert llm_console_domain(inv, "Q71", "8") is not None
    plain = load_inventory(str(_write_plain(tmp_path)))
    assert llm_console(plain) is None
    assert llm_bastion_domain(plain) is None


def test_resolver_console_domain_carries_llm_console_fields(tmp_path):
    """Regression: the resolver-built console domain (the one discovery
    actually uses) must carry the llm_console bastion + password fields --
    dropping them silently degraded the nested hop to key-only auth."""
    from harness.targets.resolver import TargetSpec, resolve_target

    inv = _console_inv(tmp_path, body=(
        "trust_level: lab\n"
        "llm_console:\n"
        "  address: 192.168.202.51\n"
        "  user: root\n"
        "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        "  tool: direct\n"
        "  trust_level: lab\n"
        "  port: 22\n"
        "  bastion: 192.168.202.51\n"
        "  password_vault_path: secret/harness/llm/rackmgr-password\n"
        "  rack_addresses: {Q71: 10.0.128.98}\n"
        "hosts: []\n"))
    target = resolve_target(TargetSpec(rack="Q71", cable="8"), inv,
                            MemorySecretStore(),
                            console_defaults=inv.llm_console)
    console = target.console
    assert console.bastion == "192.168.202.51"
    assert console.password_vault_path == "secret/harness/llm/rackmgr-password"
    assert console.address_for_rack() == "10.0.128.98"
    assert console.tool == "direct" and console.port == 22


def test_llm_bastion_uses_debug_console_credentials(tmp_path):
    """``llm_console.bastion`` is reached with the DEBUG console's proven
    credentials (log@debug-host + key) -- the only workstation-routable hop."""
    from harness.operator.llm_discover import (
        llm_bastion_domain,
        llm_console_domain,
    )

    inv = _console_inv(tmp_path, body=(
        "trust_level: lab\n"
        "console_defaults:\n"
        "  address: 192.168.202.51\n"
        "  user: log\n"
        "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        "  tool: jumpin\n"
        "  trust_level: lab\n"
        "llm_console:\n"
        "  address: 192.168.202.51\n"
        "  user: root\n"
        "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        "  tool: direct\n"
        "  trust_level: lab\n"
        "  port: 22\n"
        "  bastion: 192.168.202.51\n"
        "  password_vault_path: secret/harness/llm/rackmgr-password\n"
        "  rack_addresses: {Q71: 10.0.128.98}\n"
        "hosts: []\n"))
    bastion = llm_bastion_domain(inv, "Q71", "8")
    assert bastion is not None
    assert bastion.user == "log"               # debug credentials, not root
    assert bastion.address_for_rack() == "192.168.202.51"
    domain = llm_console_domain(inv, "Q71", "8")
    assert domain.address_for_rack() == "10.0.128.98"   # inner hop, per-rack
    assert domain.password_vault_path == "secret/harness/llm/rackmgr-password"


def _pin_inv(tmp_path) -> object:
    return _console_inv(tmp_path, body=(
        "trust_level: lab\n"
        "console_defaults:\n"
        "  address: 192.168.202.51\n"
        "  user: log\n"
        "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        "  tool: jumpin\n"
        "  trust_level: lab\n"
        "llm_console:\n"
        "  address: 192.168.202.51\n"
        "  user: root\n"
        "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        "  tool: direct\n"
        "  trust_level: lab\n"
        "  port: 22\n"
        "  bastion: 192.168.202.51\n"
        "  rack_addresses: {Q71: 10.0.128.98}\n"
        "hosts: []\n"))


def test_pin_llm_host_key_fetches_through_bastion(tmp_path, monkeypatch):
    """The pin connects to the bastion, opens a channel to rackmgr:22, reads
    the offered host key (no auth), and writes it to the known_hosts file."""
    import harness.operator.llm_discover as ld_mod
    from harness.config.vault import MemorySecretStore
    from harness.operator.llm_discover import pin_llm_host_key

    connects = []

    class _Key:
        def get_name(self):
            return "ssh-ed25519"

        def get_fingerprint(self):
            return bytes.fromhex("ab" * 32)

    class _Transport:
        def __init__(self, sock=None):
            connects.append({"transport_sock": sock})

        def open_channel(self, kind, target, src, timeout=None):
            connects.append({"channel": (kind, target)})
            return object()

        def start_client(self, timeout=None):
            pass

        def get_remote_server_key(self):
            return _Key()

        def close(self):
            pass

    class _HostKeys:
        def add(self, host, keytype, key):
            self.entry = (host, keytype)

        def save(self, path):
            from pathlib import Path
            Path(path).write_text(f"{self.entry[0]} {self.entry[1]} pinned\n",
                                  encoding="utf-8")

    class _Client:
        def __init__(self):
            self._transport = _Transport()
            self._host_keys = _HostKeys()

        def get_transport(self):
            return self._transport

        def get_host_keys(self):
            return self._host_keys

        def set_missing_host_key_policy(self, policy):
            pass

        def load_host_keys(self, path):
            pass

        def connect(self, **kw):
            connects.append({"hostname": kw["hostname"],
                             "user": kw["username"]})

        def close(self):
            pass

    monkeypatch.setattr(ld_mod.paramiko, "SSHClient", _Client)
    monkeypatch.setattr(ld_mod.paramiko, "Transport", _Transport)
    monkeypatch.setattr(ld_mod, "load_key_material",
                        lambda store, vault, tmp: tmp / "id.pem")

    monkeypatch.chdir(tmp_path)
    summary = pin_llm_host_key("Q71", "8", _pin_inv(tmp_path),
                               MemorySecretStore())
    assert "10.0.128.98" in summary and "ssh-ed25519" in summary
    assert {"hostname": "192.168.202.51", "user": "log"} in connects
    assert {"channel": ("direct-tcpip", ("10.0.128.98", 22))} in connects
    known_hosts = tmp_path / "config" / "rackmgr_known_hosts"
    assert "10.0.128.98" in known_hosts.read_text(encoding="utf-8")


def test_pin_llm_host_key_gates(tmp_path, monkeypatch):
    """Pinning needs a bastion and lab/qa trust -- both failures are staged."""
    from harness.config.vault import MemorySecretStore
    from harness.operator.llm_discover import pin_llm_host_key

    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception, match="needs an llm_console block"):
        pin_llm_host_key("Q71", "8", load_inventory(str(_write_plain(tmp_path))),
                         MemorySecretStore())
    prod = _console_inv(tmp_path, body=(
        "trust_level: lab\n"
        "llm_console:\n"
        "  address: 192.168.202.51\n"
        "  user: root\n"
        "  identity_vault_path: secret/k\n"
        "  known_hosts_path: config/kh\n"
        "  tool: direct\n"
        "  trust_level: prod\n"
        "  bastion: 192.168.202.51\n"
        "hosts: []\n"))
    with pytest.raises(Exception, match="only at lab/qa"):
        pin_llm_host_key("Q71", "8", prod, MemorySecretStore())
