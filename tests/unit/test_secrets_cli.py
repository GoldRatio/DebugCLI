"""Non-agent credential CLI (``harness secrets``): registration, listing,
removal, checks, and redacted auditing. Credentials never appear in prompts
or audit files."""

import json

import pytest

from harness.audit.redact import Redactor
from harness.operator.cli import build_parser

_KEY = b"-----BEGIN OPENSSH PRIVATE KEY-----\nZm9vYmFy\n-----END OPENSSH PRIVATE KEY-----\n"
_PASSWORD = "0penBmcR0cks!"


def _invoke(argv: list[str]):
    args = build_parser().parse_args(["secrets", *argv])
    return args.func(args), args


def test_add_ssh_stores_key_and_audits_redacted(tmp_path, capsys):
    key_file = tmp_path / "id_ed25519"
    key_file.write_bytes(_KEY)
    secret_dir = tmp_path / "secrets"
    audit_dir = tmp_path / "audit"
    rc, _ = _invoke(["add-ssh", "10.0.0.50", "--key-file", str(key_file),
                     "--secret-dir", str(secret_dir), "--audit-dir", str(audit_dir)])
    assert rc == 0
    assert (secret_dir / "secret/harness/ssh/10.0.0.50").read_bytes() == _KEY
    assert "stored SSH identity at secret/harness/ssh/10.0.0.50" in capsys.readouterr().out

    raw = (audit_dir / "audit.jsonl").read_text(encoding="utf-8")
    assert "secret_registered" in raw
    assert _KEY.decode() not in raw  # never persisted
    payload = json.loads(raw.splitlines()[-1])["payload"]
    assert payload["kind"] == "ssh"
    assert payload["vault_path"] == "secret/harness/ssh/10.0.0.50"
    assert len(payload["sha256"]) == 64


def test_add_ssh_custom_vault_path(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_bytes(_KEY)
    rc, _ = _invoke(["add-ssh", "svc", "--key-file", str(key_file),
                     "--vault-path", "secret/custom", "--secret-dir", str(tmp_path / "s")])
    assert rc == 0
    assert (tmp_path / "s/secret/custom").read_bytes() == _KEY


def test_add_ssh_missing_key_file(tmp_path, capsys):
    rc, _ = _invoke(["add-ssh", "x", "--key-file", str(tmp_path / "nope"),
                     "--secret-dir", str(tmp_path / "s")])
    assert rc == 2
    assert "key file not found" in capsys.readouterr().err


def test_add_ssh_requires_secret_dir(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_bytes(_KEY)
    with pytest.raises(SystemExit, match="requires --secret-dir"):
        _invoke(["add-ssh", "x", "--key-file", str(key_file)])


def test_set_password_interactive_and_redacted(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("harness.operator.secrets_cli.getpass.getpass",
                        lambda prompt: _PASSWORD)
    secret_dir = tmp_path / "secrets"
    audit_dir = tmp_path / "audit"
    rc, _ = _invoke(["set-password", "bmc-ro", "--secret-dir", str(secret_dir),
                     "--audit-dir", str(audit_dir)])
    assert rc == 0
    assert (secret_dir / "secret/harness/passwords/bmc-ro").read_bytes() == _PASSWORD.encode()
    raw = (audit_dir / "audit.jsonl").read_text(encoding="utf-8")
    assert _PASSWORD not in raw
    # material never reached the log, so there is nothing to redact
    assert Redactor([_PASSWORD]).redact(raw) == raw


def test_set_password_mismatch(tmp_path, monkeypatch, capsys):
    calls = {"n": 0}

    def fake_getpass(prompt):
        calls["n"] += 1
        return "first" if calls["n"] == 1 else "second"

    monkeypatch.setattr("harness.operator.secrets_cli.getpass.getpass", fake_getpass)
    rc, _ = _invoke(["set-password", "x", "--secret-dir", str(tmp_path / "s")])
    assert rc == 2
    assert "do not match" in capsys.readouterr().err


def test_list_and_rm_roundtrip(tmp_path, capsys):
    key_file = tmp_path / "key"
    key_file.write_bytes(_KEY)
    secret_dir = tmp_path / "secrets"
    rc, _ = _invoke(["add-ssh", "10.0.0.50", "--key-file", str(key_file),
                     "--secret-dir", str(secret_dir), "--audit-dir", str(tmp_path / "a")])
    assert rc == 0
    capsys.readouterr()
    rc, _ = _invoke(["add-ssh", "10.0.0.60", "--key-file", str(key_file),
                     "--secret-dir", str(secret_dir), "--audit-dir", str(tmp_path / "a")])
    assert rc == 0
    capsys.readouterr()

    rc, _ = _invoke(["list", "--secret-dir", str(secret_dir)])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["secret/harness/ssh/10.0.0.50", "secret/harness/ssh/10.0.0.60"]

    rc, _ = _invoke(["rm", "10.0.0.50", "--secret-dir", str(secret_dir),
                     "--audit-dir", str(tmp_path / "a")])
    assert rc == 0
    assert not (secret_dir / "secret/harness/ssh/10.0.0.50").exists()
    assert (secret_dir / "secret/harness/ssh/10.0.0.60").exists()


def test_rm_missing_returns_1(tmp_path, capsys):
    rc, _ = _invoke(["rm", "ghost", "--secret-dir", str(tmp_path / "s")])
    assert rc == 1
    assert "no secret at" in capsys.readouterr().err


def test_check_ok_and_missing(tmp_path, capsys):
    key_file = tmp_path / "key"
    key_file.write_bytes(_KEY)
    secret_dir = tmp_path / "secrets"
    _invoke(["add-ssh", "10.0.0.50", "--key-file", str(key_file),
             "--secret-dir", str(secret_dir), "--audit-dir", str(tmp_path / "a")])
    rc, _ = _invoke(["check", "10.0.0.50", "--secret-dir", str(secret_dir)])
    assert rc == 0
    assert "ok:" in capsys.readouterr().out

    rc, _ = _invoke(["check", "ghost", "--secret-dir", str(secret_dir)])
    assert rc == 1
    assert "add-ssh" in capsys.readouterr().err


def test_empty_list(tmp_path, capsys):
    rc, _ = _invoke(["list", "--secret-dir", str(tmp_path / "s")])
    assert rc == 0
    assert "no secrets registered" in capsys.readouterr().out


def test_audit_chain_verifies_clean(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_bytes(_KEY)
    audit_dir = tmp_path / "audit"
    _invoke(["add-ssh", "10.0.0.50", "--key-file", str(key_file),
             "--secret-dir", str(tmp_path / "s"), "--audit-dir", str(audit_dir)])

    from harness.audit.auditlog import AuditLog
    from harness.audit.redact import Redactor

    log = AuditLog(audit_dir / "audit.jsonl")
    entries = list(log.read())
    assert len(entries) == 1
    assert log.verify() == []  # chain intact
    raw = (audit_dir / "audit.jsonl").read_text(encoding="utf-8")
    assert Redactor([_KEY.decode(errors="replace")]).redact(raw) == raw


def test_key_material_never_in_audit(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_bytes(_KEY)
    audit_dir = tmp_path / "audit"
    _invoke(["add-ssh", "10.0.0.50", "--key-file", str(key_file),
             "--secret-dir", str(tmp_path / "s"), "--audit-dir", str(audit_dir)])
    _invoke(["check", "10.0.0.50", "--secret-dir", str(tmp_path / "s"),
             "--audit-dir", str(audit_dir)])
    raw = (audit_dir / "audit.jsonl").read_text(encoding="utf-8")
    assert "Zm9vYmFy" not in raw
    assert "OPENSSH" not in raw
