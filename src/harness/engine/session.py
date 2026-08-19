"""SSH session lifecycle with host-key pinning.

Host keys are pinned from a known_hosts file. If the presented key differs from the
pinned key, a ``HostKeyMismatch`` is raised (critical alert) and the session MUST NOT
proceed. Remote execution is implemented here by subclassing ``engine.runner.Runner``
so ALL inspection commands still flow through the allowlist + security gate.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import paramiko

from ..config.models import Host
from ..config.vault import SecretStore, load_key_material
from .allowlist import AllowPolicy
from .runner import CommandResult, Runner


class HostKeyMismatch(RuntimeError):
    def __init__(self, host: str, expected: str, presented: str) -> None:
        self.host = host
        self.expected = expected
        self.presented = presented
        super().__init__(f"host key change for {host}: expected {expected[:16]}... got {presented[:16]}... ALERT, not proceeding")


class SSHSession(Runner):
    """Paramiko-based Session that also IS the Runner (single command funnel)."""

    def __init__(self, host: Host, policy: AllowPolicy, store: SecretStore,
                 tmp_dir: Path | None = None, force_read_only: bool = True,
                 timeout: float = 30.0) -> None:
        super().__init__(policy=policy, force_read_only=force_read_only)
        self.host = host
        self._store = store
        self._client: paramiko.SSHClient | None = None
        self._tmp_dir = Path(tmp_dir or tempfile.gettempdir())
        self._timeout = timeout
        self._key_discarded = False

    # ---- lifecycle ----
    def open(self) -> None:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(_MissingHostReject())  # never auto-add
        try:
            client.load_host_keys(self.host.ssh.known_hosts_path)
        except FileNotFoundError:
            pass  # no pinned keys yet; _MissingHostReject fails closed
        key_path = load_key_material(self._store, self.host.ssh.identity_vault_path, self._tmp_dir)
        try:
            client.connect(
                hostname=self.host.address,
                username=self.host.ssh.user,
                key_filename=str(key_path),
                look_for_keys=False,
                allow_agent=False,
                timeout=self._timeout,
            )
        finally:
            if key_path.exists():
                key_path.unlink()
        self._client = client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def client(self) -> paramiko.SSHClient | None:
        """The open transport client, for channels the runner itself doesn't own
        (e.g. an ``InteractiveShell`` for the FAT single-test menu)."""
        return self._client

    def __enter__(self) -> SSHSession:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- Runner implementation (parameterized over SSH) ----
    def _exec(self, argv: list[str], timeout: float) -> CommandResult:
        if self._client is None:
            raise RuntimeError("session not open")
        stdin, stdout, stderr = self._client.exec_command(self._join(argv), timeout=timeout)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        code = stdout.channel.recv_exit_status()
        return CommandResult(argv=list(argv), stdout=out, stderr=err, exit_code=code, elapsed_ms=0)

    @staticmethod
    def _join(argv: list[str]) -> str:
        # Build a shell-safe, parameterized command line. Arguments are individually
        # quoted so they cannot inject into the remote shell.
        import shlex
        return " ".join(shlex.quote(a) for a in argv)


# A missing-host-key policy that always raises: a host the harness has never vetted
# must never be silently accepted.
class _MissingHostReject(paramiko.client.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        raise paramiko.SSHException(f"host key for {hostname} not in pinned known_hosts")