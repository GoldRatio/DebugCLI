"""On-demand credential prompting for zero-setup starts.

New users can launch ``harness menu`` / ``harness session`` with an EMPTY secret
store. ``OnDemandSecretStore`` wraps the concrete store and, the first time a
vault path is actually needed (LLM API key at first LLM call, SSH identity at
session open, BMC/sudo password at console probe time), pauses and asks the
*operator* interactively -- never the agent/LLM -- then stores the material for
every later run.

Security model (same guarantees as ``operator.setup_cli`` / ``operator.secrets_cli``):

- Key material is accepted only as a *generated keypair* or a *file path*, never
  a literal; passwords are read via ``getpass`` and never echoed.
- The value goes straight into the wrapped store (0o600 files under
  ``--secret-dir``); the agent's prompts, transcripts and audit logs only ever
  see vault paths and identifiers, never the material (post-facto safe).
- The SSH flow can *also* install the generated public key onto the target:
  one-time password auth through paramiko, host-key fingerprint shown for
  operator confirmation, and the pinned ``known_hosts`` entry recorded only with
  explicit confirmation. The install password is used once and never stored.
- Interactive only: when stdin is not a tty (automation/CI) or
  ``HARNESS_NO_PROMPT=1``, no prompt happens -- the original ``KeyError`` is
  re-raised so the existing hard-error paths keep their actionable messages.

Answers are injectable via ``overrides`` (``ask``/``confirm``/``secret`` fakes)
so tests drive the gate deterministically without a tty.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import io
import re
import shlex
import sys
from pathlib import Path

import paramiko

from ..config.vault import SecretStore
from .secrets_cli import SSH_VAULT
from .setup_cli import (
    BMC_PASSWORD_VAULT,
    BMC_SUDO_VAULT,
    DIAGBOT_SSH_VAULT,
    LLM_VAULT,
    RACKMGR_SSH_VAULT,
    SetupError,
    _generate_ed25519,
)

_IP4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$"
)


def classify(vault_path: str) -> str:
    """Best-effort classification of a vault path to a prompt kind.

    Paths ending in ``-password`` (or ``password``) are SSH/login passwords
    even when they live under the ``llm/`` tree (e.g. the rack-manager SSH
    password for the LLM hop); sudo passwords (e.g. the per-rack node-sudo
    entry) prompt as passwords too; ``llm`` paths otherwise ask for an API
    key, ``ssh`` paths ask for an identity key (generate or file, optionally
    installed onto the target), everything else is treated as a password.
    """
    name = vault_path.rsplit("/", 1)[-1]
    if name.endswith("password") or "sudo" in name:
        return "password"
    if vault_path == LLM_VAULT or vault_path.startswith(LLM_VAULT + "/"):
        return "llm"
    if (
        vault_path == DIAGBOT_SSH_VAULT
        or vault_path == RACKMGR_SSH_VAULT
        or vault_path == SSH_VAULT
        or vault_path.startswith(SSH_VAULT + "/")
        or vault_path.endswith("id_ed25519")
        or "rackmgr" in vault_path
    ):
        return "ssh"
    return "password"


def label_for(vault_path: str) -> str:
    """Human label for a password-style vault path."""
    if vault_path == BMC_SUDO_VAULT:
        return "BMC sudo password"
    if "sudo" in vault_path:
        name = vault_path.rsplit("/", 1)[-1]
        return f"{name} password"
    if vault_path.startswith("secret/harness/bmc/") or vault_path == BMC_PASSWORD_VAULT:
        return "BMC password"
    name = vault_path.rsplit("/", 1)[-1]
    return f"{name} password" if name else "password"


def _ssh_ip_from_vault(vault_path: str) -> str | None:
    """``secret/harness/ssh/<ip>`` carries the target address for --address runs."""
    prefix = "secret/harness/ssh/"
    if vault_path.startswith(prefix):
        candidate = vault_path[len(prefix) :]
        if "/" not in candidate and _IP4_RE.match(candidate):
            return candidate
    return None


def derive_public_key_line(private_bytes: bytes) -> str:
    """Derive the OpenSSH public-key line for a private key via paramiko.

    Accepts OpenSSH/PEM private keys for the ed25519 / rsa / ecdsa families.
    """
    buf = io.StringIO(private_bytes.decode("utf-8", errors="strict"))
    key = None
    for cls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            key = cls.from_private_key(buf)
            break
        except paramiko.SSHException:
            buf.seek(0)
            continue
    if key is None:
        raise ValueError("unsupported private key format")
    return f"{key.get_name()} {key.get_base64()}"


def _key_fingerprint_sha256(key) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return base64.b64encode(digest).decode().rstrip("=")


def append_pubkey_to_target(
    host: str, user: str, port: int, publine: str, password: str, confirm_host=None
) -> tuple[bool, str, object | None]:
    """One-shot append of ``publine`` to the target's ``~/.ssh/authorized_keys``.

    Password-authenticated (the operator's one-time password; never stored). The
    presented host-key fingerprint is shown for operator confirmation BEFORE the
    append is applied. Returns ``(ok, message, host_key_or_None)``.
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=user,
            password=password,
            allow_agent=False,
            look_for_keys=False,
            timeout=20.0,
        )
    except paramiko.AuthenticationException:
        return False, "authentication failed (wrong password or user?)", None
    except paramiko.SSHException as exc:
        return False, f"ssh failed: {exc}", None
    except OSError as exc:
        return False, f"connect failed: {exc}", None
    host_key = client.get_transport().get_remote_server_key()
    fingerprint = _key_fingerprint_sha256(host_key)
    if confirm_host is not None and not confirm_host(fingerprint):
        client.close()
        return False, "host key not confirmed; key not installed", None
    cmd = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        f"grep -qF {shlex.quote(publine)} ~/.ssh/authorized_keys 2>/dev/null || "
        f"(echo {shlex.quote(publine)} >> ~/.ssh/authorized_keys "
        "&& chmod 600 ~/.ssh/authorized_keys)"
    )
    try:
        _stdin, stdout, stderr = client.exec_command(cmd, timeout=30.0)
        err = stderr.read().decode(errors="replace")
        code = stdout.channel.recv_exit_status()
    finally:
        client.close()
    if code != 0:
        return False, (f"could not install key (exit {code}): {err.strip()[:200]}"), host_key
    return (
        True,
        (f"public key installed on {user}@{host} (fingerprint SHA256:{fingerprint})"),
        host_key,
    )


def save_host_key(known_hosts_path: str | Path, host: str, key) -> None:
    """Persist a verified host key into the pinned ``known_hosts`` file."""
    path = Path(known_hosts_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    client = paramiko.SSHClient()
    client.load_host_keys(str(path))
    client.get_host_keys().add(host, key.get_name(), key)
    client.get_host_keys().save(str(path))


# ---- interactive helpers (overrides-injectable, mirrors setup_cli) ----


def _ask(overrides: dict, prompt: str, default: str = "") -> str:
    ask = overrides.get("ask")
    if ask is None:
        from .menu import ask_text

        ask = ask_text
    return (ask(prompt, default=default) or "").strip() or default


def _confirm(overrides: dict, prompt: str, default: bool = False) -> bool:
    confirm = overrides.get("confirm")
    if confirm is None:
        from .menu import confirm as _confirm_fn

        confirm = _confirm_fn
    return confirm(prompt, default=default)


def _secret(overrides: dict, prompt: str) -> str:
    secret = overrides.get("secret")
    if secret is None:
        secret = getpass.getpass
    return secret(prompt) or ""


class CredentialPrompter:
    """Operator-only, interactive capture for a single missing vault path.

    Triggered by ``OnDemandSecretStore``; never runs inside an agent/LLM call.
    ``request`` returns the material bytes (already valid); it raises ``KeyError``
    when the operator declines or the environment is non-interactive.
    """

    def __init__(
        self,
        store: SecretStore,
        *,
        interactive: bool | None = None,
        overrides: dict | None = None,
        ssh_context: dict | None = None,
        key_dir: str | Path | None = None,
    ) -> None:
        self._store = store
        self._interactive = sys.stdin.isatty() if interactive is None else interactive
        self._overrides = overrides or {}
        self._ssh = dict(ssh_context or {})
        self._key_dir = Path(key_dir) if key_dir else Path("secrets")
        self._bridge = None  # installed by the REPL: prompts via the main thread

    # ---- context ----
    def set_ssh_context(
        self,
        *,
        host: str | None = None,
        user: str | None = None,
        port: int | None = None,
        known_hosts_path: str | None = None,
    ) -> None:
        if host is not None:
            self._ssh["host"] = host
        if user is not None:
            self._ssh["user"] = user
        if port is not None:
            self._ssh["port"] = port
        if known_hosts_path is not None:
            self._ssh["known_hosts_path"] = known_hosts_path

    def set_bridge(self, bridge) -> None:
        """Route prompts through a main-thread handler (REPL background runs)."""
        self._bridge = bridge

    # ---- entry ----
    def request(self, vault_path: str) -> bytes:
        if not self._interactive:
            raise KeyError(f"secret not found: {vault_path!r}")
        if self._bridge is not None:
            material = self._bridge(vault_path)
            if material is None:
                raise KeyError(f"secret not found: {vault_path!r}")
            return material
        return self.prompt_now(vault_path)

    def prompt_now(self, vault_path: str) -> bytes:
        """Synchronous interactive prompt (main thread); raises KeyError on decline."""
        kind = classify(vault_path)
        if kind == "llm":
            return self._prompt_llm(vault_path)
        if kind == "ssh":
            return self._prompt_ssh(vault_path)
        return self._prompt_password(vault_path)

    # ---- llm ----
    def _prompt_llm(self, vault_path: str) -> bytes:
        provider = vault_path.rsplit("/", 1)[-1].removesuffix("-key") or "llm"
        key = _secret(self._overrides, f"LLM API key for {provider}: ")
        if not key:
            print(f"  llm: skipped (no key) -- {vault_path} stays unregistered")
            raise KeyError(vault_path)
        print(f"  llm: API key stored at {vault_path} (never echoed)")
        return key.encode("utf-8")

    # ---- password ----
    def _prompt_password(self, vault_path: str) -> bytes:
        label = label_for(vault_path)
        first = _secret(self._overrides, f"{label}: ")
        if not first:
            print(f"  {label.lower()}: skipped -- {vault_path} stays unregistered")
            raise KeyError(vault_path)
        second = _secret(self._overrides, f"Confirm {label}: ")
        if first != second:
            print(f"  {label.lower()}: entries did not match -- not stored")
            raise KeyError(vault_path)
        print(f"  {label.lower()}: stored at {vault_path} (never echoed)")
        return first.encode("utf-8")

    # ---- ssh ----
    def _prompt_ssh(self, vault_path: str) -> bytes:
        if not _confirm(
            self._overrides, "No SSH identity registered -- set one up now?", default=True
        ):
            print("  ssh: skipped -- SSH access will fail to connect")
            raise KeyError(vault_path)
        choice = _ask(
            self._overrides,
            "Generate a new ed25519 key or use an existing file? (generate|file)",
            default="generate",
        ).lower()
        if choice.startswith("file"):
            material = self._read_key_file()
        else:
            private = self._key_dir / "id_ed25519"
            _generate_ed25519(private)
            material = private.read_bytes()
            for leftover in (private, private.with_suffix(".pub")):
                try:
                    leftover.unlink()
                except OSError:
                    pass
        publine = ""
        try:
            publine = derive_public_key_line(material)
        except ValueError as exc:
            print(f"  ssh: warning: {exc} -- key registered without install")
        if publine:
            print(f"  ssh: public key: {publine[:76]}...")
            self._offer_install(vault_path, publine)
        print(f"  ssh: identity stored at {vault_path} (never echoed)")
        return material

    def _read_key_file(self) -> bytes:
        key_file = Path(_ask(self._overrides, "Private key FILE path"))
        if not key_file.is_file():
            raise SetupError(f"key file not found: {key_file}")
        material = key_file.read_bytes()
        if not material.strip():
            raise SetupError(f"key file {key_file} is empty")
        return material

    def _offer_install(self, vault_path: str, publine: str) -> None:
        host = self._ssh.get("host") or _ssh_ip_from_vault(vault_path)
        user = self._ssh.get("user") or "diagbot"
        port = int(self._ssh.get("port") or 22)
        if not host:
            self._print_manual(publine, user)
            return
        target = f"{user}@{host}:{port}"
        if not _confirm(
            self._overrides,
            f"Install this public key on {target} now? (one-time "
            "password auth; the password is never stored)",
            default=True,
        ):
            self._print_manual(publine, user)
            return
        password = _secret(self._overrides, f"Password for {target}: ")
        if not password:
            self._print_manual(publine, user)
            return

        def confirm_host(fingerprint: str) -> bool:
            return _confirm(
                self._overrides,
                f"Host key fingerprint for {host}:\n    SHA256:{fingerprint}\n"
                "Verify and trust this key? (y/N)",
                default=False,
            )

        ok, message, host_key = append_pubkey_to_target(
            host, user, port, publine, password, confirm_host=confirm_host
        )
        print(f"  ssh: {message}")
        if ok and host_key is not None and self._ssh.get("known_hosts_path"):
            if _confirm(
                self._overrides,
                f"Record this host key in {self._ssh['known_hosts_path']} so pinned sessions work?",
                default=True,
            ):
                save_host_key(self._ssh["known_hosts_path"], host, host_key)
                print(f"  ssh: host key recorded at {self._ssh['known_hosts_path']}")
        elif not ok:
            self._print_manual(publine, user)

    @staticmethod
    def _print_manual(publine: str, user: str) -> None:
        print("  ssh: grant access from the remote machine (append once, per machine):")
        print(f'    echo "{publine}" >> ~/.ssh/authorized_keys')


class OnDemandSecretStore(SecretStore):
    """SecretStore that prompts the operator interactively for missing material.

    ``get`` on a missing vault path asks the operator (via the prompter), writes
    the answer into the wrapped store and returns it. Non-interactive contexts
    re-raise the original ``KeyError`` unchanged, preserving the existing
    actionable hard errors (e.g. "register one with: harness secrets add-ssh ...").
    """

    def __init__(self, inner: SecretStore, prompter: CredentialPrompter) -> None:
        self._inner = inner
        self.prompter = prompter

    def get(self, vault_path: str) -> bytes:
        if not vault_path:
            raise KeyError("secret not found: ''")
        try:
            return self._inner.get(vault_path)
        except KeyError:
            material = self.prompter.request(vault_path)
            self._inner.put(vault_path, material)
            return material

    def put(self, vault_path: str, value: bytes) -> None:
        self._inner.put(vault_path, value)

    def keys(self) -> list[str]:
        return self._inner.keys()

    def delete(self, vault_path: str) -> None:
        self._inner.delete(vault_path)

    def set_ssh_context(self, **kw) -> None:
        self.prompter.set_ssh_context(**kw)


def apply_ssh_context(store: SecretStore, target, *, ssh_user: str = "diagbot") -> None:
    """Push the resolved target's connection facts onto an on-demand store so a
    freshly generated SSH identity can be installed onto the right target."""
    if not hasattr(store, "set_ssh_context"):
        return
    if target.kind == "ssh":
        store.set_ssh_context(
            host=target.ip, user=ssh_user, known_hosts_path=target.host.ssh.known_hosts_path
        )
    elif target.kind == "console":
        store.set_ssh_context(
            host=target.console.address,
            user=target.console.user,
            known_hosts_path=target.console.known_hosts_path,
        )
    elif target.kind == "named":
        store.set_ssh_context(
            host=target.host.address,
            user=target.host.ssh.user,
            known_hosts_path=target.host.ssh.known_hosts_path,
        )
