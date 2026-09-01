"""On-demand credential prompting: the store wrapper prompts the OPERATOR (never
the agent) for a missing vault path, stores it, and returns it. Non-interactive
contexts and declined prompts re-raise KeyError unchanged."""

import sys

import pytest

from harness.config.inventory_lint import load_inventory
from harness.config.vault import MemorySecretStore
from harness.operator import credential_gate as mod
from harness.operator.cli import _make_store
from harness.operator.credential_gate import (
    DIAGBOT_SSH_VAULT,
    LLM_VAULT,
    CredentialPrompter,
    OnDemandSecretStore,
    append_pubkey_to_target,
    apply_ssh_context,
    classify,
    derive_public_key_line,
    label_for,
)
from harness.targets import TargetSpec, resolve_target


def _key_bytes() -> bytes:
    """A real ed25519 private key (OpenSSH format) for derive/install tests."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    return priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _answers(secret="s3cr3t-1"):
    def ask(prompt, default=""):
        if "Generate a new" in prompt:
            return "generate"
        return default or ""

    def confirm(prompt, default=False):
        if "Host key fingerprint" in prompt:
            return True  # trust the (fake) host key in tests
        return True

    return {
        "ask": ask,
        "confirm": confirm,
        "secret": lambda prompt: secret,
    }


def _prompter(store, **kw) -> CredentialPrompter:
    return CredentialPrompter(store, interactive=True, **kw)


# ---- classification ----


def test_classify_paths():
    assert classify(f"{LLM_VAULT}/gemini-key") == "llm"
    assert classify("secret/harness/ssh/10.0.0.50") == "ssh"
    assert classify(DIAGBOT_SSH_VAULT) == "ssh"
    assert classify("secret/harness/rackmgr/id_ed25519") == "ssh"
    assert classify("secret/harness/bmc/sudo") == "password"
    assert classify("secret/harness/passwords/bmc-ro") == "password"
    assert classify("secret/harness/llm") == "llm"


def test_classify_dash_password_paths_as_password():
    """An SSH/login password stored under the llm/ tree (e.g. the
    rack-manager password for the LLM hop) prompts as a PASSWORD, not an
    API key."""
    assert classify("secret/harness/llm/rackmgr-password") == "password"
    assert classify("secret/harness/llm/hop-password") == "password"
    assert classify(f"{LLM_VAULT}/gemini-key") == "llm"   # API keys unchanged


def test_classify_sudo_paths_as_password():
    """The per-rack node-sudo entry (captured fresh at every model setup)
    prompts as a PASSWORD with a filename-based label -- never the invisible
    LLM API-key prompt."""
    assert classify("secret/harness/llm/node-sudo-71") == "password"
    assert label_for("secret/harness/llm/node-sudo-71") == "node-sudo-71 password"
    assert "sudo" in label_for("secret/harness/bmc/sudo")   # BMC label preserved


def test_label_for_password_paths():
    assert "sudo" in label_for("secret/harness/bmc/sudo")
    assert "BMC" in label_for("secret/harness/bmc/bmc-ro")
    assert label_for("secret/harness/passwords/bmc-ro") == "bmc-ro password"


def test_derive_public_key_line():
    line = derive_public_key_line(_key_bytes())
    assert line.startswith("ssh-ed25519 ")
    assert " " in line


def test_derive_public_key_line_rejects_garbage():
    with pytest.raises(ValueError):
        derive_public_key_line(b"not a key at all")


# ---- OnDemandSecretStore ----


def test_on_demand_store_prompts_stores_and_returns(tmp_path):
    inner = MemorySecretStore()
    prompter = _prompter(inner, overrides=_answers())
    store = OnDemandSecretStore(inner, prompter)
    material = store.get(f"{LLM_VAULT}/gemini-key")
    assert material == b"s3cr3t-1"
    assert inner.get(f"{LLM_VAULT}/gemini-key") == b"s3cr3t-1"
    assert store.get(f"{LLM_VAULT}/gemini-key") == b"s3cr3t-1"  # no re-prompt


def test_on_demand_store_non_interactive_raises(tmp_path):
    inner = MemorySecretStore()
    prompter = CredentialPrompter(inner, interactive=False, overrides=_answers())
    store = OnDemandSecretStore(inner, prompter)
    with pytest.raises(KeyError):
        store.get(f"{LLM_VAULT}/gemini-key")
    assert inner.keys() == []


def test_on_demand_store_empty_path_never_prompts():
    inner = MemorySecretStore()

    def spy_secret(_prompt):
        raise AssertionError("must not prompt for an empty vault path")

    prompter = _prompter(inner, overrides={"secret": spy_secret})
    store = OnDemandSecretStore(inner, prompter)
    with pytest.raises(KeyError):
        store.get("")
    assert inner.keys() == []


def test_declined_prompt_raises_and_does_not_store():
    inner = MemorySecretStore()
    prompter = _prompter(inner, overrides=dict(_answers(), confirm=lambda p, default=False: False))
    store = OnDemandSecretStore(inner, prompter)
    with pytest.raises(KeyError):
        store.get(DIAGBOT_SSH_VAULT)
    assert inner.keys() == []


def test_bridge_returns_material():
    inner = MemorySecretStore()
    prompter = _prompter(inner)
    prompter.set_bridge(lambda vault: b"from-bridge")
    store = OnDemandSecretStore(inner, prompter)
    assert store.get("secret/harness/passwords/x") == b"from-bridge"
    assert inner.get("secret/harness/passwords/x") == b"from-bridge"


def test_bridge_decline_raises():
    inner = MemorySecretStore()
    prompter = _prompter(inner)
    prompter.set_bridge(lambda vault: None)
    store = OnDemandSecretStore(inner, prompter)
    with pytest.raises(KeyError):
        store.get("secret/harness/passwords/x")


# ---- prompt kinds ----


def test_prompt_llm_uses_secret_override():
    store = MemorySecretStore()
    prompter = _prompter(store, overrides=_answers(secret="sk-abc"))
    material = prompter.prompt_now(f"{LLM_VAULT}/gemini-key")
    assert material == b"sk-abc"


def test_prompt_password_mismatch_raises():
    calls = {"n": 0}
    store = MemorySecretStore()

    def secret(_prompt):
        calls["n"] += 1
        return "one" if calls["n"] == 1 else "two"

    prompter = _prompter(store, overrides=dict(_answers(), secret=secret))
    with pytest.raises(KeyError):
        prompter.prompt_now("secret/harness/bmc/sudo")
    assert store.keys() == []


def test_prompt_ssh_generates_and_installs(tmp_path, monkeypatch, capsys):
    store = MemorySecretStore()

    def fake_generate(private_path):
        private_path.parent.mkdir(parents=True, exist_ok=True)
        private_path.write_bytes(_key_bytes())
        private_path.with_suffix(".pub").write_text("ssh-ed25519 AAAA fake", encoding="utf-8")

    calls = {"save": 0}
    monkeypatch.setattr(mod, "_generate_ed25519", fake_generate)
    monkeypatch.setattr(
        mod,
        "append_pubkey_to_target",
        lambda host, user, port, pub, pw, confirm_host=None: (
            confirm_host("f" * 43),
            "installed",
            object(),
        ),
    )
    monkeypatch.setattr(
        mod, "save_host_key", lambda path, host, key: calls.__setitem__("save", calls["save"] + 1)
    )

    prompter = _prompter(store, overrides=_answers(), key_dir=str(tmp_path / "k"))
    prompter.set_ssh_context(host="10.0.0.50", user="diagbot", known_hosts_path="config/kh")
    material = prompter.prompt_now("secret/harness/ssh/10.0.0.50")
    assert len(material) > 32
    assert "installed" in capsys.readouterr().out
    assert calls["save"] == 1


def test_prompt_ssh_file_import(tmp_path, monkeypatch):
    key_file = tmp_path / "mykey"
    key_file.write_bytes(_key_bytes())
    store = MemorySecretStore()

    def ask(prompt, default=""):
        if "Generate a new" in prompt:
            return "file"
        if prompt.startswith("Private key FILE"):
            return str(key_file)
        return default or ""

    prompter = _prompter(store, overrides=dict(_answers(), ask=ask))
    material = prompter.prompt_now(DIAGBOT_SSH_VAULT)
    assert material == key_file.read_bytes()


def test_prompt_ssh_no_host_prints_manual(tmp_path, capsys):
    store = MemorySecretStore()

    def fake_generate(private_path):
        private_path.parent.mkdir(parents=True, exist_ok=True)
        private_path.write_bytes(_key_bytes())
        private_path.with_suffix(".pub").write_text("ssh-ed25519 AAAA fake", encoding="utf-8")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod, "_generate_ed25519", fake_generate)
    try:
        prompter = _prompter(store, overrides=_answers(), key_dir=str(tmp_path / "k"))
        prompter.prompt_now(DIAGBOT_SSH_VAULT)
        out = capsys.readouterr().out
        assert "authorized_keys" in out
        assert "echo" in out
    finally:
        monkeypatch.undo()


def test_prompt_ssh_declined_raises(tmp_path):
    store = MemorySecretStore()
    prompter = _prompter(store, overrides=dict(_answers(), confirm=lambda p, default=False: False))
    with pytest.raises(KeyError):
        prompter.prompt_now(DIAGBOT_SSH_VAULT)
    assert store.keys() == []


# ---- _make_store integration ----


def test_make_store_wraps_when_prompting(tmp_path):
    class Args:
        secret_dir = str(tmp_path / "secrets")

    store = _make_store(Args(), prompt=True)
    assert hasattr(store, "prompter")
    store.put("secret/harness/passwords/x", b"v")
    assert store.get("secret/harness/passwords/x") == b"v"


def test_make_store_plain_when_not_prompting(tmp_path):
    class Args:
        secret_dir = str(tmp_path / "secrets")

    store = _make_store(Args(), prompt=False)
    assert not hasattr(store, "prompter")


def test_make_store_wraps_on_tty(tmp_path, monkeypatch):
    class Args:
        secret_dir = str(tmp_path / "secrets")

    class FakeStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", FakeStdin())
    monkeypatch.delenv("HARNESS_NO_PROMPT", raising=False)
    store = _make_store(Args(), prompt=None)
    assert hasattr(store, "prompter")


def test_make_store_honours_no_prompt_env(tmp_path, monkeypatch):
    class Args:
        secret_dir = str(tmp_path / "secrets")

    class FakeStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", FakeStdin())
    monkeypatch.setenv("HARNESS_NO_PROMPT", "1")
    store = _make_store(Args(), prompt=None)
    assert not hasattr(store, "prompter")


# ---- apply_ssh_context ----


def test_apply_ssh_context_sets_host_for_ssh_target(tmp_path):
    body = "trust_level: lab\nllm:\n  provider: stub\nhosts: []\n"
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text(body, encoding="utf-8")
    inv = load_inventory(inv_path)
    store = MemorySecretStore()
    store.put("secret/harness/ssh/10.0.0.9", _key_bytes())
    prompter = _prompter(store)
    wrapped = OnDemandSecretStore(store, prompter)
    target = resolve_target(
        TargetSpec(ip="10.0.0.9"),
        inv,
        wrapped,
        targets_path=str(tmp_path / "t.yaml"),
        identity_vault_path="secret/harness/ssh/10.0.0.9",
    )
    apply_ssh_context(wrapped, target, ssh_user="diagbot")
    assert prompter._ssh["host"] == "10.0.0.9"


def test_apply_ssh_context_plain_store_is_noop(tmp_path):
    inv_path = tmp_path / "inventory.yaml"
    inv_path.write_text("trust_level: lab\nhosts: []\n", encoding="utf-8")
    inv = load_inventory(inv_path)
    store = MemorySecretStore()
    store.put("secret/harness/ssh/10.0.0.9", _key_bytes())
    target = resolve_target(
        TargetSpec(ip="10.0.0.9"),
        inv,
        store,
        targets_path=str(tmp_path / "t.yaml"),
        identity_vault_path="secret/harness/ssh/10.0.0.9",
    )
    apply_ssh_context(store, target, ssh_user="diagbot")  # must not raise


def test_append_pubkey_rejects_bad_host(tmp_path, capsys):
    # A real connect to a closed port fails fast and returns (False, message).
    ok, message, host_key = append_pubkey_to_target(
        "127.0.0.1", "nobody", 1, "ssh-ed25519 AAA fake", "pw", confirm_host=lambda fp: True
    )
    assert ok is False
    assert message  # 'connect failed' or 'ssh failed'
    assert host_key is None
