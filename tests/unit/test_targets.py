"""Dynamic target resolution: named host, console (rack/cable), SSH-by-IP,
and aliases -- all without per-server YAML entries."""

import pytest

from harness.config.inventory_lint import load_inventory
from harness.config.vault import MemorySecretStore
from harness.engine.sol import SerialProbeDenied
from harness.targets import Target, TargetError, TargetSpec, resolve_target

_KEY = b"-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n-----END OPENSSH PRIVATE KEY-----\n"

_INVENTORY = """\
trust_level: lab
llm:
  provider: stub
console_defaults:
  address: 192.168.202.51
  user: log
  identity_vault_path: secret/harness/rackmgr/id_ed25519
  known_hosts_path: config/rackmgr_known_hosts
  tool: jumpin
  trust_level: lab
  prompts: ["RScmCli#", "~#"]
  port: 2200
  sudo_vault_path: secret/harness/bmc/sudo
hosts:
  - name: h1
    address: 10.0.0.10
    model: model_x
    ssh:
      user: diagbot
      identity_vault_path: secret/harness/diagbot/id_ed25519
      known_hosts_path: config/known_hosts
    bmc:
      address: 10.0.0.11
      username: bmc-ro
      password_vault_path: secret/harness/bmc/bmc-ro
"""


def _inv(tmp_path, yaml_text: str = _INVENTORY):
    path = tmp_path / "inventory.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return load_inventory(path)


def _store(seed: dict[str, bytes] | None = None) -> MemorySecretStore:
    return MemorySecretStore(seed or {})


# ---- named hosts (backward compat) ----

def test_named_host_resolution(tmp_path):
    target = resolve_target(TargetSpec(name="h1"), _inv(tmp_path), _store())
    assert target.kind == "named"
    assert target.label == "h1"
    assert target.host.name == "h1"
    assert target.host.model == "model_x"


def test_unknown_named_host_raises(tmp_path):
    with pytest.raises(TargetError, match="unknown host"):
        resolve_target(TargetSpec(name="nope"), _inv(tmp_path), _store())


# ---- console targeting via fleet console_defaults ----

def test_console_target_from_defaults(tmp_path):
    target = resolve_target(TargetSpec(rack="Q61", cable="8"), _inv(tmp_path), _store())
    assert target.kind == "console"
    assert target.label == "Q61-cable8"
    assert target.host.console is not None
    assert target.host.console.rack == "Q61"
    assert target.host.console.cable == "8"
    assert target.host.console.address == "192.168.202.51"
    assert target.host.console.port == 2200
    assert target.host.console.sudo_vault_path == "secret/harness/bmc/sudo"
    assert target.trust_level == "lab"


def test_console_target_requires_console_defaults(tmp_path):
    inv = _inv(tmp_path, "trust_level: lab\nhosts: []\n")
    with pytest.raises(TargetError, match="console_defaults"):
        resolve_target(TargetSpec(rack="Q61", cable="8"), inv, _store())


def test_console_target_normalizes_rack_id(tmp_path):
    for raw in ("61", "q61", "Q61"):
        target = resolve_target(TargetSpec(rack=raw, cable="8"), _inv(tmp_path), _store())
        assert target.label == "Q61-cable8"
        assert target.host.console.rack == "Q61"


def test_console_target_gated_to_lab_qa(tmp_path):
    inv = _inv(tmp_path, _INVENTORY.replace("trust_level: lab", "trust_level: prod"))
    with pytest.raises(TargetError, match="only lab/qa"):
        resolve_target(TargetSpec(rack="Q61", cable="8"), inv, _store())


def test_console_target_rejects_injection(tmp_path):
    with pytest.raises(SerialProbeDenied):
        resolve_target(TargetSpec(rack="Q61; reboot", cable="8"), _inv(tmp_path), _store())


def test_rack_without_cable_is_no_target(tmp_path):
    with pytest.raises(TargetError, match="no target given"):
        resolve_target(TargetSpec(rack="Q61"), _inv(tmp_path), _store())


# ---- SSH-by-IP targeting ----

def test_ssh_target_uses_per_ip_identity(tmp_path):
    store = _store({"secret/harness/ssh/10.0.0.50": _KEY})
    target = resolve_target(TargetSpec(ip="10.0.0.50"), _inv(tmp_path), store)
    assert target.kind == "ssh"
    assert target.label == "10.0.0.50"
    assert target.host.address == "10.0.0.50"
    assert target.host.ssh.identity_vault_path == "secret/harness/ssh/10.0.0.50"
    assert target.host.ssh.user == "diagbot"
    assert target.host.ssh.known_hosts_path == "config/known_hosts"
    assert target.host.bmc.password_vault_path == ""  # no BMC channel


def test_ssh_target_falls_back_to_legacy_diagbot_key(tmp_path):
    store = _store({"secret/harness/diagbot/id_ed25519": _KEY})
    target = resolve_target(TargetSpec(ip="10.0.0.50"), _inv(tmp_path), store)
    assert target.host.ssh.identity_vault_path == "secret/harness/diagbot/id_ed25519"


def test_ssh_target_explicit_identity_and_user(tmp_path):
    store = _store({"secret/custom/key": _KEY})
    target = resolve_target(TargetSpec(ip="10.0.0.50"), _inv(tmp_path), store,
                            ssh_user="svc", identity_vault_path="secret/custom/key",
                            known_hosts_path="config/custom_known_hosts")
    assert target.host.ssh.user == "svc"
    assert target.host.ssh.identity_vault_path == "secret/custom/key"
    assert target.host.ssh.known_hosts_path == "config/custom_known_hosts"


def test_ssh_target_missing_identity_raises_with_hint(tmp_path):
    with pytest.raises(TargetError, match="no SSH identity.*add-ssh"):
        resolve_target(TargetSpec(ip="10.0.0.50"), _inv(tmp_path), _store())


def test_ssh_target_explicit_identity_missing_raises(tmp_path):
    with pytest.raises(TargetError, match="secret/custom/key"):
        resolve_target(TargetSpec(ip="10.0.0.50"), _inv(tmp_path), _store(),
                       identity_vault_path="secret/custom/key")


# ---- aliases ----

def _aliases_file(tmp_path, body: str):
    path = tmp_path / "targets.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_alias_to_console_target(tmp_path):
    tf = _aliases_file(tmp_path, "targets:\n- {alias: t1, rack: Q61, cable: 8}\n")
    target = resolve_target(TargetSpec(alias="t1"), _inv(tmp_path), _store(),
                            targets_path=tf)
    assert target.kind == "console"
    assert target.host.console.rack == "Q61"


def test_alias_to_ssh_target(tmp_path):
    store = _store({"secret/harness/ssh/10.0.0.9": _KEY})
    tf = _aliases_file(tmp_path, "targets:\n- {alias: t2, address: 10.0.0.9}\n")
    target = resolve_target(TargetSpec(alias="t2"), _inv(tmp_path), store,
                            targets_path=tf)
    assert target.kind == "ssh"
    assert target.host.address == "10.0.0.9"


def test_unknown_alias_raises(tmp_path):
    tf = _aliases_file(tmp_path, "targets: []\n")
    with pytest.raises(TargetError, match="alias"):
        resolve_target(TargetSpec(alias="nope"), _inv(tmp_path), _store(),
                       targets_path=tf)


# ---- no target ----

def test_no_target_raises(tmp_path):
    with pytest.raises(TargetError, match="no target given"):
        resolve_target(TargetSpec(), _inv(tmp_path), _store())


def test_target_is_frozen_dataclass(tmp_path):
    import dataclasses

    target = resolve_target(TargetSpec(name="h1"), _inv(tmp_path), _store())
    assert isinstance(target, Target)
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.label = "mutated"
