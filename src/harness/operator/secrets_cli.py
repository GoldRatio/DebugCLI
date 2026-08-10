"""Non-agent credential CLI (``harness secrets``).

Credentials are registered HERE, outside the agent prompt, so prompts only ever
carry identifiers and vault paths. Key material is accepted only as a *file
path* (never a literal argument -- shell history / process list would leak it);
passwords are read interactively via ``getpass`` and never echoed.

Every registration/removal/check appends to an audit log (redacted with the
secret value, so even an accidental echo cannot persist).
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import sys
from pathlib import Path

from ..audit.auditlog import AuditLog
from ..audit.redact import Redactor
from ..config.vault import DirSecretStore, SecretStore

SSH_VAULT = "secret/harness/ssh"
PASSWORD_VAULT = "secret/harness/passwords"


def _store_for(args) -> SecretStore:
    if not getattr(args, "secret_dir", None):
        raise SystemExit(
            "harness secrets requires --secret-dir <dir> (file-backed lab store) "
            "or a configured vault backend")
    return DirSecretStore(args.secret_dir)


def _audit(args, session_id: str, kind: str, payload: dict, secret: str | None = None) -> None:
    redactor = Redactor([secret] if secret else None)
    out = Path(getattr(args, "audit_dir", None) or "harness_runs/secrets")
    AuditLog(out / "audit.jsonl", redactor).append(session_id, kind, payload)


def _vault_path(name: str, explicit: str | None, kind: str) -> str:
    if explicit is not None:
        return explicit
    base = SSH_VAULT if kind == "ssh" else PASSWORD_VAULT
    return f"{base}/{name}"


def run_add_ssh(args) -> int:
    key_file = Path(args.key_file)
    if not key_file.is_file():
        print(f"error: key file not found: {key_file}", file=sys.stderr)
        return 2
    store = _store_for(args)
    vault_path = _vault_path(args.name, args.vault_path, "ssh")
    material = key_file.read_bytes()
    if not material.strip():
        print(f"error: key file {key_file} is empty", file=sys.stderr)
        return 2
    store.put(vault_path, material)
    _audit(args, "secrets", "secret_registered", {
        "kind": "ssh", "name": args.name, "vault_path": vault_path,
        "key_file": str(key_file), "sha256": hashlib.sha256(material).hexdigest(),
    }, secret=material.decode(errors="replace"))
    print(f"stored SSH identity at {vault_path} (from {key_file})")
    return 0


def run_set_password(args) -> int:
    store = _store_for(args)
    password = getpass.getpass(f"Password for {args.name}: ")
    if not password:
        print("error: empty password", file=sys.stderr)
        return 2
    confirm = getpass.getpass("Confirm: ")
    if password != confirm:
        print("error: passwords do not match", file=sys.stderr)
        return 2
    vault_path = _vault_path(args.name, args.vault_path, "password")
    store.put(vault_path, password.encode("utf-8"))
    _audit(args, "secrets", "secret_registered", {
        "kind": "password", "name": args.name, "vault_path": vault_path,
    }, secret=password)
    print(f"stored password at {vault_path}")
    return 0


def run_secrets_list(args) -> int:
    store = _store_for(args)
    keys = sorted(store.keys())
    if not keys:
        print("(no secrets registered)")
        return 0
    for key in keys:
        print(key)
    return 0


def run_secrets_rm(args) -> int:
    store = _store_for(args)
    vault_path = _vault_path(args.name, args.vault_path, "ssh")
    try:
        store.delete(vault_path)
    except KeyError:
        print(f"error: no secret at {vault_path}", file=sys.stderr)
        return 1
    _audit(args, "secrets", "secret_removed", {
        "name": args.name, "vault_path": vault_path})
    print(f"removed {vault_path}")
    return 0


def run_secrets_check(args) -> int:
    store = _store_for(args)
    vault_path = _vault_path(args.name, args.vault_path, "ssh")
    try:
        value = store.get(vault_path)
    except KeyError:
        print(f"missing: {vault_path} (register with 'harness secrets add-ssh "
              f"{args.name} --key-file <path>' or 'set-password')", file=sys.stderr)
        return 1
    _audit(args, "secrets", "secret_checked", {
        "name": args.name, "vault_path": vault_path})
    print(f"ok: {vault_path} ({len(value)} bytes)")
    return 0


def run_secrets(args) -> int:
    if args.secret_action == "add-ssh":
        return run_add_ssh(args)
    if args.secret_action == "set-password":
        return run_set_password(args)
    if args.secret_action == "list":
        return run_secrets_list(args)
    if args.secret_action == "rm":
        return run_secrets_rm(args)
    if args.secret_action == "check":
        return run_secrets_check(args)
    return 2


def build_secrets_parser(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="secret_action", required=True)
    a = sub.add_parser("add-ssh", help="store an SSH private key from a FILE (never a literal)")
    a.add_argument("name", help="name (use the target IP for --address targeting)")
    a.add_argument("--key-file", required=True, help="path to the private key file")
    a.add_argument("--vault-path", default=None, help="vault path (default secret/harness/ssh/<name>)")
    a.add_argument("--secret-dir", default=None)
    a.add_argument("--audit-dir", default=None)
    a.set_defaults(func=run_secrets)

    s = sub.add_parser("set-password", help="store a password interactively (never echoed)")
    s.add_argument("name")
    s.add_argument("--vault-path", default=None, help="vault path (default secret/harness/passwords/<name>)")
    s.add_argument("--secret-dir", default=None)
    s.add_argument("--audit-dir", default=None)
    s.set_defaults(func=run_secrets)

    l = sub.add_parser("list", help="list registered vault paths")
    l.add_argument("--secret-dir", default=None)
    l.add_argument("--audit-dir", default=None)
    l.set_defaults(func=run_secrets)

    r = sub.add_parser("rm", help="remove a registered secret")
    r.add_argument("name")
    r.add_argument("--vault-path", default=None)
    r.add_argument("--secret-dir", default=None)
    r.add_argument("--audit-dir", default=None)
    r.set_defaults(func=run_secrets)

    c = sub.add_parser("check", help="verify a vault path resolves")
    c.add_argument("name")
    c.add_argument("--vault-path", default=None)
    c.add_argument("--secret-dir", default=None)
    c.add_argument("--audit-dir", default=None)
    c.set_defaults(func=run_secrets)


def _secret_audit_dump(path: str) -> list[dict]:
    return [e.payload for e in AuditLog(path).read()]
