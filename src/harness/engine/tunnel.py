"""Per-run SSH local port-forward through the rack manager.

The fleet's compute nodes have NO directly routable IP from an operator
workstation: the sanctioned access path is SSH to the rack manager, then the
``jumpin``/serial-console CLI (see ``engine.sol``). When an LLM is served on a
node's OS port (e.g. vLLM on ``:8000``) no workstation can open TCP to it, so
HTTP must be carried THROUGH the rack-manager SSH hop.

``LLMForward`` opens that hop exactly the way ``SerialConsole`` does -- pinned
host keys, vault-path identity materialized to a temp file, lab/qa gate -- and
then exposes ``<target>:<port>`` on a local listener
(``127.0.0.1:<ephemeral>``) over a paramiko ``direct-tcpip`` channel per
connection (the same primitive as ``engine.bastion``). The caller points the
OpenAI-compatible adapter at ``forward.url`` and tears the forward down with
the run (context manager).

Failure is staged, never silent: ``TunnelError.stage`` names the leg that
failed -- ``auth`` (SSH to the rack manager), ``bind`` (local listen), or
``forward`` (the rack manager refused/cannot route the channel, i.e. TCP
forwarding disabled or no route to the node). ``harness llm check`` renders
those stages plus the reverse-tunnel fallback recipe.
"""

from __future__ import annotations

import socket
import tempfile
import threading
from pathlib import Path
from typing import Self

import paramiko

from ..config.models import ConsoleDomain
from ..config.vault import SecretStore, load_key_material


class TunnelError(RuntimeError):
    """A leg of the rack-manager forward failed (see ``TunnelError.stage``)."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class _MissingHostReject(paramiko.client.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        raise paramiko.SSHException(f"host key for {hostname} not in pinned known_hosts")


class LLMForward:
    """Context-managed local forward ``127.0.0.1:<ephemeral> -> <target>:<port>``
    carried over the rack-manager SSH connection.

    When the rack manager is unroutable from the workstation (``bastion`` set
    to the jump host's :class:`ConsoleDomain`), the SSH connection opens
    THROUGH the bastion: paramiko -> bastion (key auth) -> ``direct-tcpip`` to
    ``rackmgr:22`` -> nested SSH (key first, then the vault-sourced password
    ``console.password_vault_path``) -> ``direct-tcpip`` to the node's vLLM
    port from the rack manager.

    Usage::

        with LLMForward("10.0.0.42", 8000, inv.console_defaults, store) as fwd:
            llm = OpenAICompatLLM(url=fwd.url, model="Qwen2.5-7B-Instruct")
            ...

    ``url`` is a ready-to-use OpenAI-compatible base URL (ends in ``/v1``).
    """

    def __init__(self, target_host: str, target_port: int,
                 console: ConsoleDomain, store: SecretStore,
                 tmp_dir: Path | None = None, timeout: float = 30.0,
                 bastion: ConsoleDomain | None = None) -> None:
        if console.trust_level not in ("lab", "qa"):
            raise TunnelError("trust", (
                f"rack-manager forwarding allowed only at lab/qa, "
                f"not {console.trust_level}"))
        self.target_host = target_host
        self.target_port = int(target_port)
        self.console = console
        self.bastion = bastion
        self._store = store
        self._tmp_dir = Path(tmp_dir or tempfile.gettempdir())
        self._timeout = timeout
        self.url: str | None = None
        self._client: paramiko.SSHClient | None = None
        self._bastion_client: paramiko.SSHClient | None = None
        self._server: socket.socket | None = None
        self._stop = threading.Event()
        # Last direct-tcpip failure, surfaced by staged diagnostics.
        self.forward_error: str | None = None

    # ---- lifecycle ----

    def _connect_client(self, client: paramiko.SSHClient, hostname: str,
                        user: str, key_path: Path, password: str | None,
                        sock: socket.socket | None = None,
                        password_vault_path: str | None = None) -> None:
        """One SSH connect with staged failure (never a raw traceback)."""
        try:
            client.connect(
                hostname=hostname,
                username=user,
                key_filename=str(key_path) if key_path else None,
                look_for_keys=False,
                allow_agent=False,
                timeout=min(self._timeout, 30.0),
                sock=sock,
                password=password,
            )
        except Exception as exc:
            hint = ""
            if "Authentication failed" in str(exc):
                hint = (f" -- both auth methods rejected for {user}@{hostname}"
                        + (f" (fix the material at {password_vault_path} or "
                           "install the harness key on the host)"
                           if password_vault_path else
                           " (install the harness key on the host)"))
            raise TunnelError(
                "auth", f"ssh to {hostname} failed: {exc}{hint}") from exc

    def start(self) -> str:
        """Connect, bind the local listener, start the accept loop. Idempotent;
        returns the local base URL (``http://127.0.0.1:<port>/v1``)."""
        if self.url is not None:
            return self.url
        sock: socket.socket | None = None
        if self.bastion is not None:
            self._bastion_client = paramiko.SSHClient()
            self._bastion_client.set_missing_host_key_policy(_MissingHostReject())
            try:
                self._bastion_client.load_host_keys(self.bastion.known_hosts_path)
            except FileNotFoundError:
                pass
            b_key = load_key_material(self._store, self.bastion.identity_vault_path,
                                      self._tmp_dir)
            try:
                self._connect_client(self._bastion_client,
                                     self.bastion.address_for_rack(),
                                     self.bastion.user, b_key, None)
            finally:
                if b_key.exists():
                    b_key.unlink()
            rackmgr_addr = self.console.address_for_rack()
            try:
                sock = self._bastion_client.get_transport().open_channel(
                    "direct-tcpip", (rackmgr_addr, 22), ("127.0.0.1", 0),
                    timeout=min(self._timeout, 30.0))
            except Exception as exc:
                raise TunnelError(
                    "auth", f"bastion {self.bastion.address_for_rack()} could "
                    f"not open a channel to {rackmgr_addr}: {exc}") from exc

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(_MissingHostReject())
        try:
            client.load_host_keys(self.console.known_hosts_path)
        except FileNotFoundError:
            pass  # no pinned keys yet; _MissingHostReject fails closed
        key_path = load_key_material(
            self._store, self.console.identity_vault_path, self._tmp_dir)
        password = None
        if self.console.password_vault_path is not None:
            try:
                password = self._store.get(
                    self.console.password_vault_path).decode(
                    errors="strict").rstrip("\r\n")
            except KeyError:
                raise TunnelError(
                    "auth", "rack-manager password missing from vault: "
                    f"{self.console.password_vault_path!r}") from None
        try:
            self._connect_client(client, self.console.address_for_rack(),
                                 self.console.user, key_path, password,
                                 sock=sock,
                                 password_vault_path=self.console.password_vault_path)
        finally:
            if key_path.exists():
                key_path.unlink()
        self._client = client
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen(16)
            server.settimeout(0.5)
        except OSError as exc:
            self.close()
            raise TunnelError("bind", f"local listener failed: {exc}") from exc
        self._server = server
        self.url = f"http://127.0.0.1:{server.getsockname()[1]}/v1"
        threading.Thread(target=self._accept_loop, daemon=True,
                         name="harness-llm-forward").start()
        return self.url

    def close(self) -> None:
        self._stop.set()
        server, self._server = self._server, None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        client, self._client = self._client, None
        if client is not None:
            client.close()
        bastion, self._bastion_client = self._bastion_client, None
        if bastion is not None:
            bastion.close()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- plumbing ----

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                sock, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                break  # listener closed
            threading.Thread(target=self._handle, args=(sock,),
                             daemon=True).start()

    def _handle(self, sock: socket.socket) -> None:
        try:
            channel = self._client.get_transport().open_channel(
                "direct-tcpip",
                (self.target_host, self.target_port),
                ("127.0.0.1", sock.getsockname()[1]),
                timeout=self._timeout,
            )
        except Exception as exc:  # noqa: BLE001 - paramiko raises assorted types
            # refused / no route / forwarding disabled: recorded for diagnostics
            self.forward_error = str(exc)
            try:
                sock.close()
            except OSError:
                pass
            return
        p1 = threading.Thread(target=self._pipe, args=(sock, channel),
                              daemon=True)
        p2 = threading.Thread(target=self._pipe, args=(channel, sock),
                              daemon=True)
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        for closer in (channel, sock):
            try:
                closer.close()
            except OSError:
                pass

    @staticmethod
    def _pipe(src, dst) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except Exception:  # noqa: BLE001, S110 - peer died mid-stream; drop the pair
            pass
        finally:
            try:
                dst.shutdown_write()
            except Exception:  # noqa: BLE001, S110 - already closing
                pass


def parse_tunnel_spec(spec: str) -> tuple[str, int]:
    """``HOST:PORT`` -> (host, port); identifier-safe (regex-validated)."""
    import re
    m = re.match(r"^([A-Za-z0-9._-]+):(\d{1,5})$", spec.strip())
    if not m:
        raise ValueError(f"invalid tunnel target {spec!r} (expected HOST:PORT)")
    port = int(m.group(2))
    if not 1 <= port <= 65535:
        raise ValueError(f"tunnel port out of range: {port}")
    return m.group(1), port
