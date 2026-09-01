"""Inventory lint: no inline secrets allowed, vault-paths only."""

import pytest

from harness.config.inventory_lint import InventoryError, lint_inventory, load_inventory
from harness.config.models import (
    BMCDomain,
    ConsoleDomain,
    Host,
    Inventory,
    LLMConfig,
    SSHDomain,
)


def _host(**over) -> Host:
    ssh = SSHDomain(user="diagbot", identity_vault_path="secret/harness/ssh",
                    known_hosts_path="config/known_hosts")
    bmc = BMCDomain(address="10.0.0.11", username="bmc-ro",
                    password_vault_path="secret/harness/bmc")
    base = {"name": "h1", "address": "10.0.0.10", "model": "model_x",
            "ssh": ssh, "bmc": bmc, "collector_profile": "model_x_diag"}
    base.update(over)
    return Host(**base)


def test_vault_paths_clean():
    inv = Inventory(trust_level="prod", hosts=[_host()])
    assert lint_inventory(inv) == []


def test_console_defaults_parse_rack_addresses(tmp_path):
    """Per-rack addresses, the bastion chain, and the node login fields all
    round-trip into the ConsoleDefaults/ConsoleDomain model; the LLM-only
    ``llm_console`` block parses alongside the debug console_defaults."""
    inv_path = tmp_path / "inv.yaml"
    inv_path.write_text(
        "trust_level: lab\n"
        "console_defaults:\n"
        "  address: 192.168.202.51\n"
        "  user: log\n"
        "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        "  tool: jumpin\n"
        "  port: 2200\n"
        "  sudo_vault_path: secret/harness/bmc/sudo\n"
        "llm_console:\n"
        "  address: 192.168.202.51\n"
        "  user: root\n"
        "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        "  tool: direct\n"
        "  port: 22\n"
        "  bastion: 192.168.202.51\n"
        "  password_vault_path: secret/harness/llm/rackmgr-password\n"
        "  node_user: yemankyaw\n"
        "  node_password_vault_path: secret/harness/llm/node-sudo-71\n"
        "  rack_addresses:\n"
        "    Q61: 10.0.128.74\n"
        "    Q71: 10.0.128.98\n"
        "hosts: []\n", encoding="utf-8")
    inv = load_inventory(str(inv_path))
    d = inv.console_defaults
    assert d.tool == "jumpin" and d.port == 2200          # debug path unchanged
    assert d.rack_addresses is None
    lc = inv.llm_console
    assert lc.tool == "direct" and lc.port == 22 and lc.user == "root"
    assert lc.bastion == "192.168.202.51"
    assert lc.password_vault_path == "secret/harness/llm/rackmgr-password"
    assert lc.node_user == "yemankyaw"
    assert lc.node_password_vault_path == "secret/harness/llm/node-sudo-71"
    assert lc.rack_addresses == {"Q61": "10.0.128.74", "Q71": "10.0.128.98"}
    assert lc.address_for_rack("Q71") == "10.0.128.98"
    assert lc.address_for_rack("Q63") == "192.168.202.51"
    assert lint_inventory(inv) == []


def test_inline_secret_flagged():
    bmc = BMCDomain(address="10.0.0.11", username="bmc-ro",
                    password_vault_path="AKIA1234567890123456")
    inv = Inventory(trust_level="prod", hosts=[_host(bmc=bmc)])
    issues = lint_inventory(inv)
    assert issues and any("inline secret" in i.message for i in issues)


def test_load_rejects_inline(tmp_path):
    bad = tmp_path / "inv.yaml"
    bad.write_text(
        "trust_level: prod\nhosts:\n"
        "- name: h1\n  address: 10.0.0.10\n  model: model_x\n"
        "  ssh: {user: diagbot, identity_vault_path: secret/a}\n"
        "  bmc: {address: 1.1.1.1, username: u, password_vault_path: 'AKIA1234567890123456'\n}\n",
        encoding="utf-8")
    with pytest.raises(InventoryError):
        load_inventory(bad)


def _console(port: int = 2200, sudo: str | None = "secret/bmc/sudo") -> ConsoleDomain:
    return ConsoleDomain(
        address="192.168.202.51", user="log",
        identity_vault_path="secret/rackmgr/id", known_hosts_path="config/kh",
        rack="03", cable="12", trust_level="lab", port=port, sudo_vault_path=sudo,
    )


def test_load_console_port_and_sudo_vault(tmp_path):
    inv = tmp_path / "inv.yaml"
    inv.write_text(
        "trust_level: lab\nhosts:\n"
        "- name: h1\n  address: 10.0.0.10\n  model: model_x\n"
        "  ssh: {user: diagbot, identity_vault_path: secret/a, known_hosts_path: config/kh}\n"
        "  bmc: {address: 1.1.1.1, username: u, password_vault_path: secret/b}\n"
        "  console:\n"
        "    address: 192.168.202.51\n    user: log\n"
        "    identity_vault_path: secret/rackmgr/id\n    known_hosts_path: config/kh\n"
        "    rack: 03\n    cable: 12\n    trust_level: lab\n"
        "    port: 2200\n    sudo_vault_path: secret/harness/bmc/sudo\n",
        encoding="utf-8")
    host = load_inventory(inv).get("h1")
    assert host.console is not None
    assert host.console.port == 2200
    assert host.console.sudo_vault_path == "secret/harness/bmc/sudo"
    assert host.console.tool == "jumpin"


def test_lint_console_inline_secret_flagged():
    host = _host(console=_console(sudo="AKIA1234567890123456"))
    issues = lint_inventory(Inventory(trust_level="lab", hosts=[host]))
    assert any("inline secret" in i.message and "sudo_vault_path" in i.field
               for i in issues)
    # a non-path non-secret value is still flagged as a path violation
    host = _host(console=_console(sudo="0penBmc"))
    issues = lint_inventory(Inventory(trust_level="lab", hosts=[host]))
    assert any("vault/config path" in i.message and "sudo_vault_path" in i.field
               for i in issues)


def test_lint_console_port_range_flagged():
    host = _host(console=_console(port=70000))
    issues = lint_inventory(Inventory(trust_level="lab", hosts=[host]))
    assert any("port 70000 out of range" in i.message for i in issues)


def test_load_llm_block(tmp_path):
    inv = tmp_path / "inv.yaml"
    inv.write_text(
        "trust_level: lab\n"
        "llm:\n"
        "  provider: gemini\n"
        "  model: gemini-2.5-flash\n"
        "  url: https://generativelanguage.googleapis.com/v1beta/openai\n"
        "  api_key_vault_path: secret/harness/llm/gemini-key\n"
        "  timeout: 60\n"
        "hosts: []\n",
        encoding="utf-8")
    loaded = load_inventory(inv)
    assert loaded.llm is not None
    assert loaded.llm.provider == "gemini"
    assert loaded.llm.model == "gemini-2.5-flash"
    assert loaded.llm.api_key_vault_path == "secret/harness/llm/gemini-key"
    assert loaded.llm.timeout == 60.0


def test_load_rejects_unknown_llm_provider(tmp_path):
    inv = tmp_path / "inv.yaml"
    inv.write_text(
        "trust_level: lab\nllm: {provider: claude}\nhosts: []\n", encoding="utf-8")
    with pytest.raises(InventoryError, match="unknown provider"):
        load_inventory(inv)


def test_lint_llm_key_requires_vault_path():
    inv = Inventory(trust_level="lab", hosts=[],
                    llm=LLMConfig(provider="gemini", api_key_vault_path="0penBmc"))
    issues = lint_inventory(inv)
    assert any("vault/config path" in i.message and i.field == "llm.api_key_vault_path"
               for i in issues)

    inv = Inventory(trust_level="lab", hosts=[],
                    llm=LLMConfig(provider="gemini", api_key_vault_path="A" * 70))
    issues = lint_inventory(inv)
    assert any("inline secret" in i.message for i in issues)
