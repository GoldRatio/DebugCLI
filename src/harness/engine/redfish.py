"""Rack-level Redfish GET client: read-only evidence without a serial session.

The rack manager exposes a per-node Redfish service on its own HTTPS port:
``https://<rack_ip>/<cable>/amc1/redfish/v1`` (``<cable>`` is the node's
eth_id). Rack-side tooling (the operator's ``get_event_logs`` /
``get_service_conditions`` aliases) uses it to fetch event logs and service
conditions without jumping into the server, and so does this harness: the
debug step collects that evidence straight over HTTPS while the serial console
stays free for the BMC probes.

Safety mirrors the console probe gate:

- The client surface exposes ONLY ``get`` -- no write verb is expressible, so
  firmware/power-control endpoints (e.g. ``update-multipart``) are unreachable
  by construction, not by policy.
- Paths are regex-validated (no query strings, no traversal).
- The credential lives in the vault; basic auth is built per request and never
  logged.
- TLS is unverified (self-signed BMC/service certs; the ``curl -k`` parity the
  rack tooling itself uses) -- recorded here so nobody "fixes" it silently.
- Trust gate: lab/qa only, same as the console and the LLM forward.

Transport: the rack IP is often routable only from the rack side, so ``get``
first tries a direct HTTPS request and, on a connect-type failure, retries
through a local port-forward carried over the rack-manager SSH hop (the same
``direct-tcpip`` primitive as ``engine.tunnel``). Which transport served a
response is reported per fetch and recorded in the collector dumps.
"""

from __future__ import annotations

import base64
import http.client
import re
import ssl
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from ..config.models import ConsoleDomain
from ..config.vault import SecretStore
from .sol import SerialProbeDenied, validate_identifier

# Service path (after the leading "/<cable>/amc1/redfish/v1"): segments of
# identifier characters only. Empty = the service root document (the preflight
# target). No query strings, no "..": a GET list is all this client can say.
_REDFISH_PATH = re.compile(r"^(?:/[A-Za-z0-9._-]+)*/?$")

DEFAULT_REDFISH_USER = "root"


class RedfishError(RuntimeError):
    """A Redfish fetch failed; ``stage`` names the leg (see ``RedfishClient``)."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class RedfishGet:
    """One GET response: HTTP status + body plus how it was served."""

    status: int
    body: str
    elapsed_ms: int
    transport: str          # "direct" | "tunnel"
    truncated: bool = False


class RedfishClient:
    """GET-only Redfish client for one node (rack + cable).

    Usage::

        client = RedfishClient(console, console.cable, store)
        result = client.get("/ServiceConditions")   # status/body/transport
        ...
        client.close()

    ``get`` raises :class:`RedfishError` only when the request could not be
    delivered (auth material missing, tunnel refused); HTTP error statuses are
    returned as data so the collector can record them honestly.
    """

    def __init__(self, console: ConsoleDomain, cable: str, store: SecretStore,
                 *, user: str | None = None, timeout: float = 30.0,
                 max_body: int = 512 * 1024, port: int = 443,
                 forward_factory=None, tmp_dir: Path | None = None) -> None:
        if console.trust_level not in ("lab", "qa"):
            raise RedfishError("trust", (
                f"redfish collection allowed only at lab/qa, "
                f"not {console.trust_level}"))
        try:
            validate_identifier(cable, "cable")
        except SerialProbeDenied as exc:
            raise RedfishError("path", str(exc)) from exc
        self.console = console
        self.cable = cable
        self.host = console.address_for_rack()
        self.user = user or console.redfish_user or DEFAULT_REDFISH_USER
        self.timeout = timeout
        self.max_body = max_body
        self.port = port
        self._store = store
        self._forward_factory = forward_factory or self._default_forward
        self._tmp_dir = Path(tmp_dir or tempfile.gettempdir())
        self._auth_header: str | None = None
        self._forward = None
        self._tunnel_port: int | None = None

    # ---- URL surface (derived, never configured) ----

    @property
    def base_url(self) -> str:
        """``https://<rack_ip>/<cable>/amc1/redfish/v1`` (the service root)."""
        return f"https://{self.host}/{self.cable}/amc1/redfish/v1"

    def url_for(self, path: str) -> str:
        return f"{self.base_url}{self._request_uri(path)}"

    @property
    def _request_prefix(self) -> str:
        return f"/{self.cable}/amc1/redfish/v1"

    def _request_uri(self, path: str) -> str:
        if not path:
            return ""
        if not path.startswith("/") or ".." in path.split("/") \
                or not _REDFISH_PATH.match(path):
            raise RedfishError("path", f"unsafe redfish path: {path!r}")
        return path

    # ---- public GET surface ----

    def preflight(self) -> RedfishGet:
        """GET the service root; the harness parity of ``redfish_init``."""
        return self.get("")

    def get(self, path: str) -> RedfishGet:
        uri = self._request_prefix + self._request_uri(path)
        started = time.monotonic()
        try:
            status, body, truncated = self._request_direct(uri)
            transport = "direct"
        except (OSError, http.client.HTTPException):
            status, body, truncated = self._request_tunneled(uri)
            transport = "tunnel"
        return RedfishGet(
            status=status, body=body, transport=transport,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            truncated=truncated,
        )

    # ---- lifecycle ----

    def close(self) -> None:
        forward, self._forward = self._forward, None
        self._tunnel_port = None
        if forward is not None:
            forward.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- transports ----

    def _tls_context(self) -> ssl.SSLContext:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False      # self-signed BMC certs (curl -k parity)
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    def _headers(self, host_header: str) -> dict[str, str]:
        return {
            "Host": host_header,
            "Authorization": self._basic_auth(),
            "Accept": "application/json",
            "Connection": "close",
        }

    def _basic_auth(self) -> str:
        if self._auth_header is None:
            try:
                password = self._store.get(
                    self.console.redfish_password_vault_path or ""
                ).decode().rstrip("\r\n")
            except KeyError as exc:
                raise RedfishError("auth",
                                   f"redfish password missing from vault: "
                                   f"{self.console.redfish_password_vault_path!r}") from exc
            token = base64.b64encode(
                f"{self.user}:{password}".encode()).decode()
            self._auth_header = f"Basic {token}"
        return self._auth_header

    def _request_direct(self, uri: str) -> tuple[int, str, bool]:
        conn = http.client.HTTPSConnection(
            self.host, self.port, timeout=self.timeout,
            context=self._tls_context())
        try:
            return self._execute(conn, uri, self.host)
        finally:
            conn.close()

    def _request_tunneled(self, uri: str) -> tuple[int, str, bool]:
        port = self._tunnel_port
        if port is None:
            try:
                self._forward = self._forward_factory(self.host, self.port)
                url = self._forward.start()   # staged failure on refusal
            except Exception as exc:
                raise RedfishError(
                    "forward", f"rack-manager forward failed: {exc}") from exc
            port = self._tunnel_port = int(url.rsplit(":", 1)[1].split("/", 1)[0])
        conn = http.client.HTTPSConnection(
            "127.0.0.1", port, timeout=self.timeout,
            context=self._tls_context())
        try:
            # The rack IP stays in the Host header: the hop carries raw bytes,
            # the service still sees the name it was addressed by.
            return self._execute(conn, uri, self.host)
        except (OSError, http.client.HTTPException) as exc:
            # The tunnel is the last resort: a failure here must surface as
            # RedfishError so collectors can stage it like any other fetch.
            raise RedfishError(
                "forward", f"tunneled request failed: {exc}") from exc
        finally:
            conn.close()

    def _execute(self, conn: http.client.HTTPSConnection, uri: str,
                 host_header: str) -> tuple[int, str, bool]:
        conn.request("GET", uri, headers=self._headers(host_header))
        resp = conn.getresponse()
        body = resp.read(self.max_body + 1)
        truncated = len(body) > self.max_body
        text = body[:self.max_body].decode("utf-8", errors="replace")
        return resp.status, text, truncated

    def _default_forward(self, target_host: str, target_port: int):
        from .tunnel import LLMForward
        return LLMForward(target_host, target_port, self.console,
                          self._store, tmp_dir=self._tmp_dir,
                          timeout=self.timeout)
