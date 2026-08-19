"""``harness setup`` wizard: inventory creation/patcing, LLM key registration,
SSH key generation-or-file, BMC creds, and vault-path verification.

Drives the wizard through ``overrides`` (ask/confirm/secret fakes) so no tty or
network is needed; key generation is monkeypatched to written fixtures.
"""

import pytest

from harness.config.inventory_lint import load_inventory
from harness.config.vault import DirSecretStore
from harness.operator import setup_cli as mod
from harness.operator.setup_cli import (
    BMC_PASSWORD_VAULT,
    BMC_SUDO_VAULT,
    DIAGBOT_SSH_VAULT,
    LLM_VAULT,
    SetupError,
    _check_inventory_secrets,
    _patch_llm_block,
    run_setup,
)

FAKE_PRIVATE = b"-----BEGIN OPENSSH PRIVATE KEY-----\nfake\n-----END OPENSSH PRIVATE KEY-----\n"


def _args(tmp_path, name="inventory.yaml", inventory=None):
    class Args:
        inventory = str(tmp_path / name) if name else None
        secret_dir = str(tmp_path / "secrets")

    if inventory is not None:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(inventory, encoding="utf-8")
    return Args()


def _answers(secret="test-api-key-123"):
    def ask(prompt, default=""):
        if "provider" in prompt:
            return "gemini"
        if "Model for" in prompt:
            return "gemini-2.5-flash"
        if prompt.startswith("Generate a new"):
            return "generate"
        return default or ""

    return {
        "ask": ask,
        "confirm": lambda prompt, default=False: default,
        "secret": lambda prompt: secret,
    }


def _generate_fixture(tmp_path):
    def fake_generate(private_path):
        private_path.parent.mkdir(parents=True, exist_ok=True)
        private_path.write_bytes(FAKE_PRIVATE)
        private_path.with_suffix(".pub").write_text(
            "ssh-ed25519 AAAA fake@harness", encoding="utf-8")
    return fake_generate


def test_setup_creates_inventory_and_registers_everything(tmp_path, monkeypatch,
                                                          capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_generate_ed25519", _generate_fixture(tmp_path))
    args = _args(tmp_path, name=None)  # no inventory yet
    assert run_setup(args, overrides=_answers()) == 0

    inv_path = tmp_path / "inventory.yaml"
    assert inv_path.exists()
    inv = load_inventory(inv_path)
    assert inv.llm.provider == "gemini"
    assert inv.llm.model == "gemini-2.5-flash"
    assert inv.llm.api_key_vault_path == f"{LLM_VAULT}/gemini-key"

    store = DirSecretStore(tmp_path / "secrets")
    assert store.get(f"{LLM_VAULT}/gemini-key") == b"test-api-key-123"
    assert store.get(DIAGBOT_SSH_VAULT) == FAKE_PRIVATE
    out = capsys.readouterr().out
    assert "ssh-ed25519 AAAA fake@harness" in out  # authorized_keys one-liner
    assert "setup complete" in out


def test_setup_patches_existing_inventory_not_hosts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_generate_ed25519", _generate_fixture(tmp_path))
    body = (
        "trust_level: prod\n"
        "llm:\n"
        "  provider: gemini\n"
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
        "      password_vault_path: secret/harness/bmc/bmc-ro\n")
    args = _args(tmp_path, inventory=body)
    yes = dict(_answers(), confirm=lambda prompt, default=False: True)
    assert run_setup(args, overrides=yes) == 0

    text = (tmp_path / "inventory.yaml").read_text(encoding="utf-8")
    assert "provider: gemini" in text
    assert "model: gemini-2.5-flash" in text
    assert "api_key_vault_path: secret/harness/llm/gemini-key" in text
    assert "name: h1" in text  # untouched
    assert "password_vault_path: secret/harness/bmc/bmc-ro" in text


def test_setup_skips_when_already_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_generate_ed25519", _generate_fixture(tmp_path))
    store = DirSecretStore(tmp_path / "secrets")
    store.put(f"{LLM_VAULT}/gemini-key", b"old")
    store.put(DIAGBOT_SSH_VAULT, FAKE_PRIVATE)
    body = (
        "trust_level: lab\n"
        "llm:\n"
        "  provider: gemini\n"
        "  model: gemini-2.5-flash\n"
        "  api_key_vault_path: secret/harness/llm/gemini-key\n"
        "hosts: []\n")
    args = _args(tmp_path, inventory=body)

    def spy_secret(prompt):
        raise AssertionError("should not ask for a secret when configured")

    overrides = dict(_answers(secret="new"), secret=spy_secret)
    assert run_setup(args, overrides=overrides) == 0
    assert store.get(f"{LLM_VAULT}/gemini-key") == b"old"  # not overwritten


def test_setup_stub_provider_skips_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_generate_ed25519", _generate_fixture(tmp_path))
    body = "trust_level: lab\nllm:\n  provider: stub\nhosts: []\n"
    args = _args(tmp_path, inventory=body)
    assert run_setup(args, overrides=_answers()) == 0
    store = DirSecretStore(tmp_path / "secrets")
    assert store.get(DIAGBOT_SSH_VAULT) == FAKE_PRIVATE
    with pytest.raises(KeyError):
        store.get(f"{LLM_VAULT}/gemini-key")


def test_setup_uses_existing_key_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key_file = tmp_path / "id_ed25519"
    key_file.write_bytes(FAKE_PRIVATE)
    body = "trust_level: lab\nllm:\n  provider: gemini\nhosts: []\n"

    def ask(prompt, default=""):
        if prompt.startswith("Generate a new"):
            return "file"
        if "Provider" in prompt or "provider" in prompt:
            return "gemini"
        if prompt.startswith("Private key FILE"):
            return str(key_file)
        return default or ""

    answers = dict(_answers(), ask=ask)
    args = _args(tmp_path, inventory=body)
    assert run_setup(args, overrides=answers) == 0
    store = DirSecretStore(tmp_path / "secrets")
    assert store.get(DIAGBOT_SSH_VAULT) == FAKE_PRIVATE


def test_setup_reports_unregistered_inventory_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_generate_ed25519", _generate_fixture(tmp_path))
    body = (
        "trust_level: lab\n"
        "llm:\n"
        "  provider: gemini\n"
        "hosts:\n"
        "  - name: h1\n"
        "    address: 10.0.0.1\n"
        "    model: model_x\n"
        "    ssh:\n"
        "      user: diagbot\n"
        "      identity_vault_path: secret/harness/diagbot/id_ed25519\n"
        "      known_hosts_path: config/known_hosts\n"
        "    bmc:\n"
        "      address: 10.0.0.2\n"
        "      username: bmc-ro\n"
        "      password_vault_path: secret/harness/bmc/bmc-ro\n")
    answers = dict(_answers(),
                   confirm=lambda prompt, default=False: False)  # no BMC asks
    args = _args(tmp_path, inventory=body)
    assert run_setup(args, overrides=answers) == 1
    store = DirSecretStore(tmp_path / "secrets")
    with pytest.raises(KeyError):
        store.get(BMC_PASSWORD_VAULT)
    with pytest.raises(KeyError):
        store.get(BMC_SUDO_VAULT)


def test_setup_missing_ssh_keygen_is_graceful(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    body = "trust_level: lab\nllm:\n  provider: gemini\nhosts: []\n"
    args = _args(tmp_path, inventory=body)
    with pytest.raises(SetupError, match="ssh-keygen not found"):
        run_setup(args, overrides=_answers())


def test_patch_llm_block_inserts_block_when_absent():
    text = "trust_level: lab\nhosts: []\n"
    out = _patch_llm_block(text, provider="gemini", model="m1",
                           vault_path="secret/harness/llm/gemini-key")
    assert out.startswith("llm:\n  provider: gemini\n  model: m1\n"
                          "  api_key_vault_path: secret/harness/llm/gemini-key\n")
    assert "hosts: []" in out


def test_patch_llm_block_preserves_rest():
    text = ("trust_level: prod\n"
            "llm:\n"
            "  provider: stub\n"
            "  timeout: 30\n"
            "hosts: []\n")
    out = _patch_llm_block(text, provider="openai", model="gpt-4o",
                           vault_path="secret/harness/llm/openai-key")
    assert "provider: openai" in out
    assert "model: gpt-4o" in out
    assert "api_key_vault_path: secret/harness/llm/openai-key" in out
    assert "timeout: 30" in out
    assert "hosts: []" in out


def test_check_inventory_secrets_lists_missing(tmp_path):
    body = (
        "trust_level: lab\n"
        "llm:\n"
        "  provider: gemini\n"
        "  api_key_vault_path: secret/harness/llm/gemini-key\n"
        "hosts: []\n")
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text(body, encoding="utf-8")
    store = DirSecretStore(tmp_path / "secrets")
    missing = _check_inventory_secrets(inv_path, store)
    assert missing == ["llm api key: secret/harness/llm/gemini-key"]
    store.put("secret/harness/llm/gemini-key", b"k")
    assert _check_inventory_secrets(inv_path, store) == []


# ---- rack manager console (console_defaults) ----

def test_console_defaults_block_reuses_diagbot_identity():
    block = mod._console_defaults_block("192.168.202.51")
    assert block.startswith("console_defaults:\n")
    assert "address: 192.168.202.51" in block
    assert f"identity_vault_path: {DIAGBOT_SSH_VAULT}" in block
    assert f"sudo_vault_path: {BMC_SUDO_VAULT}" in block


def test_insert_console_defaults_before_hosts():
    text = ("trust_level: lab\n"
            "llm:\n  provider: gemini\n"
            "hosts: []\n")
    out = mod._insert_console_defaults(
        text, mod._console_defaults_block("10.0.0.5"))
    assert out.index("console_defaults:") < out.index("hosts: []")
    assert "trust_level: lab" in out.split("console_defaults:")[0]
    assert "hosts: []" in out


def test_insert_console_defaults_appends_when_no_hosts():
    text = "trust_level: lab\n"
    out = mod._insert_console_defaults(
        text, mod._console_defaults_block("10.0.0.5"))
    assert out.index("console_defaults:") > out.index("trust_level:")
    assert "  address: 10.0.0.5" in out
    assert out.endswith(f"  sudo_vault_path: {BMC_SUDO_VAULT}\n")
    assert "hosts:" not in out


def test_setup_console_defaults_writes_block(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text("trust_level: lab\nhosts: []\n", encoding="utf-8")
    store = DirSecretStore(tmp_path / "secrets")
    store.put(DIAGBOT_SSH_VAULT, FAKE_PRIVATE)

    answers = dict(_answers(), ask=lambda prompt, default="":
                   "192.168.202.51"
                   if "console IP" in prompt else (default or ""))
    mod._setup_console_defaults(_args(tmp_path), answers, inv_path, store)

    inv = load_inventory(inv_path)
    assert inv.console_defaults is not None
    assert inv.console_defaults.address == "192.168.202.51"
    assert inv.console_defaults.user == "log"
    assert store.get(mod.RACKMGR_SSH_VAULT) == FAKE_PRIVATE  # identity mirrored
    out = capsys.readouterr().out
    assert "console_defaults: rack manager 192.168.202.51" in out


def test_setup_console_defaults_skipped_on_blank(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text("trust_level: lab\nhosts: []\n", encoding="utf-8")
    store = DirSecretStore(tmp_path / "secrets")
    store.put(DIAGBOT_SSH_VAULT, FAKE_PRIVATE)

    mod._setup_console_defaults(_args(tmp_path), _answers(), inv_path, store)

    assert load_inventory(inv_path).console_defaults is None
    assert "console_defaults: skipped" in capsys.readouterr().out


def test_setup_console_defaults_keeps_existing_block(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.chdir(tmp_path)
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text(
        "trust_level: lab\n"
        "console_defaults:\n"
        "  address: 192.168.202.99\n"
        "  user: log\n"
        f"  identity_vault_path: {DIAGBOT_SSH_VAULT}\n"
        "hosts: []\n", encoding="utf-8")
    store = DirSecretStore(tmp_path / "secrets")
    store.put(DIAGBOT_SSH_VAULT, FAKE_PRIVATE)

    mod._setup_console_defaults(_args(tmp_path), _answers(), inv_path, store)

    inv = load_inventory(inv_path)
    assert inv.console_defaults.address == "192.168.202.99"
    assert "already present" in capsys.readouterr().out


def test_setup_full_flow_writes_console_defaults(tmp_path, monkeypatch, capsys):
    """End-to-end: a fresh `harness setup` that answers the console-IP prompt
    produces a discoverable inventory with a working console_defaults block."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "_generate_ed25519", _generate_fixture(tmp_path))
    args = _args(tmp_path, name=None)

    def ask(prompt, default=""):
        if "provider" in prompt:
            return "gemini"
        if "Model for" in prompt:
            return "gemini-2.5-flash"
        if prompt.startswith("Generate a new"):
            return "generate"
        if "console IP" in prompt:
            return "192.168.202.51"
        return default or ""

    overrides = {
        "ask": ask,
        "confirm": lambda prompt, default=False: True,
        "secret": lambda prompt: "test-api-key-123",
    }
    assert run_setup(args, overrides=overrides) == 0

    inv = load_inventory(tmp_path / "inventory.yaml")
    assert inv.console_defaults is not None
    assert inv.console_defaults.address == "192.168.202.51"
    assert "setup complete" in capsys.readouterr().out


def _real_key_bytes() -> bytes:
    """A real ed25519 private key (OpenSSH format) for install-path tests."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    return priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _install_overrides():
    """confirm accepts the install + host-key prompts; secret supplies a password."""
    def confirm(prompt, default=False):
        if "Host key fingerprint" in prompt:
            return True
        return True

    return dict(_answers(), confirm=confirm)


def test_install_rackmgr_key_pins_host_key(tmp_path, monkeypatch, capsys):
    from harness.operator import credential_gate as cg

    calls = {"install": [], "save": []}
    monkeypatch.setattr(
        cg, "append_pubkey_to_target",
        lambda *a, **k: calls["install"].append((a, k)) or (True, "installed", object()))
    monkeypatch.setattr(
        cg, "save_host_key",
        lambda path, host, key: calls["save"].append((path, host, key)))

    store = DirSecretStore(tmp_path / "secrets")
    store.put(mod.RACKMGR_SSH_VAULT, _real_key_bytes())
    mod._install_rackmgr_key(_args(tmp_path), _install_overrides(),
                             "192.168.202.51", store)

    assert calls["install"], "append_pubkey_to_target never called"
    args, kwargs = calls["install"][0]
    assert args[0] == "192.168.202.51"  # host
    assert args[1] == "log"             # user
    assert args[2] == 22                # port
    assert "ssh-ed25519 " in args[3]    # publine
    assert "confirm_host" in kwargs
    assert calls["save"][0][0] == "config/rackmgr_known_hosts"
    assert calls["save"][0][1] == "192.168.202.51"
    out = capsys.readouterr().out
    assert "host key recorded at config/rackmgr_known_hosts" in out


def test_install_rackmgr_key_declined_prints_manual(tmp_path, monkeypatch,
                                                    capsys):
    store = DirSecretStore(tmp_path / "secrets")
    store.put(mod.RACKMGR_SSH_VAULT, _real_key_bytes())
    overrides = dict(_answers(), confirm=lambda prompt, default=False: False)
    mod._install_rackmgr_key(_args(tmp_path), overrides, "192.168.202.51", store)
    out = capsys.readouterr().out
    assert "grant access from the rack manager" in out
    assert "ssh-ed25519 " in out


def test_install_rackmgr_key_no_password_prints_manual(tmp_path, monkeypatch,
                                                       capsys):
    store = DirSecretStore(tmp_path / "secrets")
    store.put(mod.RACKMGR_SSH_VAULT, _real_key_bytes())
    overrides = dict(_answers(), secret=lambda prompt: "")
    mod._install_rackmgr_key(_args(tmp_path), overrides, "192.168.202.51", store)
    assert "grant access from the rack manager" in capsys.readouterr().out


def test_install_rackmgr_key_failure_prints_manual(tmp_path, monkeypatch,
                                                   capsys):
    from harness.operator import credential_gate as cg

    monkeypatch.setattr(
        cg, "append_pubkey_to_target", lambda *a, **k: (False, "auth failed", None))
    store = DirSecretStore(tmp_path / "secrets")
    store.put(mod.RACKMGR_SSH_VAULT, _real_key_bytes())
    mod._install_rackmgr_key(_args(tmp_path), _install_overrides(),
                             "192.168.202.51", store)
    out = capsys.readouterr().out
    assert "auth failed" in out
    assert "grant access from the rack manager" in out


def test_setup_console_defaults_writes_and_installs_key(tmp_path, monkeypatch,
                                                        capsys):
    """Full console_defaults step: block written, identity mirrored, and the
    key installed on the rack manager with the host key pinned."""
    from harness.operator import credential_gate as cg

    calls = {"save": []}
    monkeypatch.setattr(
        cg, "append_pubkey_to_target",
        lambda *a, **k: (True, "installed", object()))
    monkeypatch.setattr(
        cg, "save_host_key",
        lambda path, host, key: calls["save"].append((path, host)))

    monkeypatch.chdir(tmp_path)
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text("trust_level: lab\nhosts: []\n", encoding="utf-8")
    store = DirSecretStore(tmp_path / "secrets")
    store.put(DIAGBOT_SSH_VAULT, _real_key_bytes())

    answers = dict(_install_overrides(), ask=lambda prompt, default="":
                   "192.168.202.51" if "console IP" in prompt else (default or ""))
    mod._setup_console_defaults(_args(tmp_path), answers, inv_path, store)

    inv = load_inventory(inv_path)
    assert inv.console_defaults.address == "192.168.202.51"
    assert calls["save"] == [("config/rackmgr_known_hosts", "192.168.202.51")]
    out = capsys.readouterr().out
    assert "console: installed" in out
    assert "host key recorded at config/rackmgr_known_hosts" in out


# ---- BMC credential prompts (sudo first, bmc-ro second, inventory-conditioned) ----


def _bmc_host_inventory() -> str:
    return (
        "trust_level: lab\n"
        "llm:\n"
        "  provider: gemini\n"
        "  model: gemini-2.5-flash\n"
        "  api_key_vault_path: secret/harness/llm/gemini-key\n"
        "hosts:\n"
        "  - name: h1\n"
        "    address: 10.0.0.1\n"
        "    model: model_x\n"
        "    ssh:\n"
        "      user: diagbot\n"
        "      identity_vault_path: secret/harness/diagbot/id_ed25519\n"
        "      known_hosts_path: config/known_hosts\n"
        "    bmc:\n"
        "      address: 10.0.0.2\n"
        "      username: bmc-ro\n"
        "      password_vault_path: secret/harness/bmc/bmc-ro\n")


def _console_sudo_inventory() -> str:
    return (
        "trust_level: lab\n"
        "llm:\n"
        "  provider: gemini\n"
        "  model: gemini-2.5-flash\n"
        "  api_key_vault_path: secret/harness/llm/gemini-key\n"
        "console_defaults:\n"
        "  address: 192.168.202.51\n"
        "  user: log\n"
        f"  identity_vault_path: {DIAGBOT_SSH_VAULT}\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        f"  sudo_vault_path: {BMC_SUDO_VAULT}\n"
        "hosts: []\n")


def _both_bmc_inventory() -> str:
    """console_defaults.sudo_vault_path AND a host bmc block: both referenced."""
    return (
        "trust_level: lab\n"
        "llm:\n"
        "  provider: gemini\n"
        "  model: gemini-2.5-flash\n"
        "  api_key_vault_path: secret/harness/llm/gemini-key\n"
        "console_defaults:\n"
        "  address: 192.168.202.51\n"
        "  user: log\n"
        f"  identity_vault_path: {DIAGBOT_SSH_VAULT}\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        f"  sudo_vault_path: {BMC_SUDO_VAULT}\n"
        "hosts:\n"
        "  - name: h1\n"
        "    address: 10.0.0.1\n"
        "    model: model_x\n"
        "    ssh:\n"
        "      user: diagbot\n"
        "      identity_vault_path: secret/harness/diagbot/id_ed25519\n"
        "      known_hosts_path: config/known_hosts\n"
        "    bmc:\n"
        "      address: 10.0.0.2\n"
        "      username: bmc-ro\n"
        "      password_vault_path: secret/harness/bmc/bmc-ro\n")


def test_setup_bmc_skips_when_nothing_referenced(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text("trust_level: lab\nhosts: []\n", encoding="utf-8")
    store = DirSecretStore(tmp_path / "secrets")
    monkeypatch.setattr(
        mod, "_confirm", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not prompt when nothing is referenced")))
    monkeypatch.setattr(
        mod, "_ask_secret", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not ask a secret when nothing is referenced")))
    mod._setup_bmc(_args(tmp_path), _answers(), inv_path, store)
    assert "no BMC vaults referenced" in capsys.readouterr().out


def test_setup_bmc_prompts_sudo_first_when_console_references(tmp_path,
                                                              monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text(_console_sudo_inventory(), encoding="utf-8")
    store = DirSecretStore(tmp_path / "secrets")

    prompts = []
    monkeypatch.setattr(
        mod, "_ask_secret",
        lambda *a, **k: prompts.append(a[2]) or "sudo-pw")
    overrides = dict(_answers(), confirm=lambda prompt, default=False: True)
    mod._setup_bmc(_args(tmp_path), overrides, inv_path, store)

    assert store.get(BMC_SUDO_VAULT) == b"sudo-pw"
    assert "BMC sudo password" in prompts[0]
    with pytest.raises(KeyError):
        store.get(BMC_PASSWORD_VAULT)  # bmc-ro not referenced -> never asked
    out = capsys.readouterr().out
    assert f"bmc: {BMC_SUDO_VAULT} stored" in out
    assert "bmc-ro" not in out


def test_setup_bmc_prompts_bmc_ro_when_host_references(tmp_path, monkeypatch,
                                                       capsys):
    monkeypatch.chdir(tmp_path)
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text(_bmc_host_inventory(), encoding="utf-8")
    store = DirSecretStore(tmp_path / "secrets")

    prompts = []
    monkeypatch.setattr(
        mod, "_ask_secret",
        lambda *a, **k: prompts.append(a[2]) or "bmc-ro-pw")
    overrides = dict(_answers(), confirm=lambda prompt, default=False: True)
    mod._setup_bmc(_args(tmp_path), overrides, inv_path, store)

    assert store.get(BMC_PASSWORD_VAULT) == b"bmc-ro-pw"
    assert "BMC read-only password" in prompts[0]
    with pytest.raises(KeyError):
        store.get(BMC_SUDO_VAULT)  # no console_defaults -> sudo not asked
    out = capsys.readouterr().out
    assert f"bmc: {BMC_PASSWORD_VAULT} stored" in out


def test_setup_bmc_shared_password_shortcut(tmp_path, monkeypatch, capsys):
    """Both vaults referenced: sudo prompted first, bmc-ro reuses its value."""
    monkeypatch.chdir(tmp_path)
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text(_both_bmc_inventory(), encoding="utf-8")
    store = DirSecretStore(tmp_path / "secrets")

    secret_calls = []
    confirm_prompts = []
    monkeypatch.setattr(
        mod, "_ask_secret",
        lambda *a, **k: secret_calls.append(a[2]) or "shared-pw")
    overrides = dict(
        _answers(),
        confirm=lambda prompt, default=False: confirm_prompts.append(prompt) or True)
    mod._setup_bmc(_args(tmp_path), overrides, inv_path, store)

    assert store.get(BMC_SUDO_VAULT) == b"shared-pw"
    assert store.get(BMC_PASSWORD_VAULT) == b"shared-pw"
    assert len(secret_calls) == 1  # second vault reused the first
    assert any("Use the same password" in p for p in confirm_prompts)
    assert "BMC sudo password" in secret_calls[0]


def test_setup_bmc_shared_password_declined_prompts_again(tmp_path,
                                                          monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text(_both_bmc_inventory(), encoding="utf-8")
    store = DirSecretStore(tmp_path / "secrets")

    def confirm(prompt, default=False):
        return "same password" not in prompt

    secret_calls = []
    monkeypatch.setattr(
        mod, "_ask_secret",
        lambda *a, **k: secret_calls.append(a[2]) or f"pw-{len(secret_calls)}")
    overrides = dict(_answers(), confirm=confirm)
    mod._setup_bmc(_args(tmp_path), overrides, inv_path, store)

    assert store.get(BMC_SUDO_VAULT) == b"pw-1"
    assert store.get(BMC_PASSWORD_VAULT) == b"pw-2"
    assert len(secret_calls) == 2  # shortcut declined -> separate prompt