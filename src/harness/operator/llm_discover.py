"""Node-side vLLM endpoint discovery over the rack-and-cable console path.

The model often runs on the golden server itself -- the rack/cable debug
target. Rack/cable addressing yields a jumpin serial console (never a
routable address), so the tunnel ``HOST:PORT`` cannot be derived from the
target spec alone. This module runs a small batch of read-only probes ON the
node through that same console (``hostname -I``, ``ss -l -t -n`` and, when
the sudo password is configured, ``sudo -S docker ps``) and parses the
candidates: node addresses, listening ports, container port mappings. The
operator picks/overrides; the resulting ``HOST:PORT`` is verified end-to-end
by ``LLMForward`` (``harness llm check --tunnel``) before anything is saved.

When the rack managers sit behind a bastion (``llm_console.bastion``),
``pin_llm_host_key`` fetches a manager's host key THROUGH that bastion (a
plain client-hello, no auth -- the ``ssh-keyscan`` pattern) and pins it into
the known_hosts file, so the two-hop chain works without manual key bookkeeping.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from pathlib import Path

import paramiko

from ..config.models import ConsoleDefaults, ConsoleDomain, Inventory
from ..config.vault import SecretStore, load_key_material
from ..engine.runner import CommandResult
from ..engine.sol import ConsoleRunner, SerialConsole, SerialConsoleError, _MissingHostReject
from ..targets.resolver import Target, TargetError, TargetSpec, resolve_target

# One console session runs every probe in a single batch (one rackmgr hop +
# one serial-session start). Separate flags: the probe validator matches
# tokens literally, so bundled short flags ("-ltnp") are not expressible.
_DISCOVERY_PROBES = ("hostname -I", "ss -l -t -n")
_DOCKER_PROBE = "sudo -S docker ps"

_IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
_DOCKER_PORT_RE = re.compile(r"(?:0\.0\.0\.0|\[::\]|::):(\d+)->(\d+)/tcp")


@dataclass
class DiscoveryResult:
    """Candidates parsed from the node probes (empty lists are honest)."""

    addresses: list[str] = field(default_factory=list)     # hostname -I
    ports: list[int] = field(default_factory=list)         # ss LISTEN ports
    containers: dict[str, int] = field(default_factory=dict)  # docker name -> host port
    notes: list[str] = field(default_factory=list)         # probe-level problems

    def suggested_ports(self) -> list[int]:
        """Container-mapped ports first (exact), then other listening ports."""
        return list(dict.fromkeys([*self.containers.values(), *self.ports]))


def parse_node_addresses(output: str) -> list[str]:
    """``hostname -I`` -> IPv4 dotted quads (link-local v6 and noise dropped)."""
    return [t for t in output.split() if _IPV4_RE.fullmatch(t)]


def parse_listening_ports(output: str) -> list[int]:
    """``ss -l -t -n`` rows -> local ports of LISTEN sockets."""
    ports: list[int] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "LISTEN":
            try:
                ports.append(int(fields[3].rsplit(":", 1)[1]))
            except (ValueError, IndexError):
                continue
    return sorted(set(ports))


def parse_docker_ports(output: str) -> dict[str, int]:
    """``docker ps`` rows -> {container name: first published host port}.

    Only ``host->container`` tcp mappings count (``PORTS`` column); rows
    without a published port are skipped.
    """
    out: dict[str, int] = {}
    for line in output.splitlines():
        if "->" not in line:
            continue
        mapping = None
        for tok in line.split():
            if "->" in tok and "/tcp" in tok:
                mapping = tok
                break
        if mapping is None:
            continue
        m = _DOCKER_PORT_RE.search(mapping)
        if m:
            out[line.rsplit(None, 1)[-1]] = int(m.group(2))
    return out


def _digits(value) -> str:
    m = re.search(r"\d+", str(value))
    return m.group() if m else ""


def _inventory_console_address(inv: Inventory, rack: str, cable: str) -> str | None:
    """Named hosts may pin a console rack/cable AND a routable address: when
    they match, the tunnel HOST is known without opening the console. Rack
    spellings vary (``Q61``/``q61``/``61``/``03``) -- compared numerically."""
    want_rack, want_cable = _digits(rack), _digits(cable)
    for host in inv.hosts:
        console = getattr(host, "console", None)
        if (console is not None and _digits(console.rack) == want_rack
                and _digits(console.cable) == want_cable and host.address):
            return host.address
    return None


def llm_console(inv: Inventory) -> ConsoleDefaults | None:
    """The console config for LLM access: the inventory's ``llm_console``
    block when present (a golden server often reached with a different
    tool/port/account than the debug console), else ``console_defaults``.
    The DEBUG console path always uses ``console_defaults`` unchanged."""
    return inv.llm_console or inv.console_defaults


def llm_console_domain(inv: Inventory, rack: str = "", cable: str = "",
                       ) -> ConsoleDomain | None:
    """``llm_console`` as a ready :class:`ConsoleDomain` with the runtime
    rack/cable layered in (so per-rack manager addresses resolve)."""
    defaults = llm_console(inv)
    return defaults.for_rack(rack, cable) if defaults else None


def llm_bastion_domain(inv: Inventory, rack: str = "", cable: str = "",
                       ) -> ConsoleDomain | None:
    """The bastion hop for the LLM console when configured: the
    ``llm_console.bastion`` address is reached with the DEBUG console's
    credentials (``console_defaults`` -- the proven workstation route)."""
    defaults = inv.llm_console
    if defaults is None or not defaults.bastion or inv.console_defaults is None:
        return None
    return inv.console_defaults.for_rack(rack, cable)


def _node_sudo_path(rack: str) -> str:
    """Per-rack vault path of the node user's sudo password -- the wizard
    captures it fresh at every model setup and stores it here."""
    m = re.search(r"\d+", str(rack))
    return f"secret/harness/llm/node-sudo-{m.group() if m else rack}"


def discover_domain(inv: Inventory, rack: str, cable: str,
                    store: SecretStore, node_user: str | None = None,
                    ) -> ConsoleDomain:
    """The llm console domain for discovery, with the per-rack node sudo
    path layered in when the config does not pin one, and the node login
    identity when the caller captured it (the wizard prompts fresh at every
    setup)."""
    target: Target = resolve_target(TargetSpec(rack=rack, cable=cable), inv, store,
                                    console_defaults=llm_console(inv))
    if target.console is None:  # pragma: no cover - resolve_target guarantees console
        raise TargetError("rack/cable did not resolve to a console session")
    domain = target.console
    if not domain.sudo_vault_path:
        # Probes run as the node's logged-in user; docker needs that user's
        # sudo. The wizard captures the password fresh at every setup and
        # stores it here; standalone discover prompts on demand if absent.
        domain = _dc_replace(domain, sudo_vault_path=_node_sudo_path(rack))
    if node_user and not domain.node_user:
        domain = _dc_replace(domain, node_user=node_user,
                             node_password_vault_path=_node_sudo_path(rack))
    return domain


def discover(rack: str, cable: str, inv: Inventory, store: SecretStore, *,
             node_user: str | None = None, console_factory=None,
             ) -> DiscoveryResult:
    """Probe the golden server for its vLLM endpoint via the jumpin console.

    Targeting errors (no console block, wrong trust level, bad identifiers)
    raise :class:`TargetError` -- the caller shows them as-is. Probe-level
    problems (tool missing, no sudo password, unparseable output) never fail
    the discovery: they are recorded in ``DiscoveryResult.notes`` so the
    wizard can fall back to manual entry.
    """
    # Inventory fast path: rack/cable pinned on a named host with an address.
    address = _inventory_console_address(inv, rack, cable)
    if address is not None:
        return DiscoveryResult(
            addresses=[address],
            notes=[("HOST resolved from the inventory (no console hop needed); "
                    "enter the vLLM port manually")])

    domain = discover_domain(inv, rack, cable, store, node_user=node_user)

    probes = list(_DISCOVERY_PROBES)
    if domain.sudo_vault_path:
        # SerialConsole only performs the sudo password handshake when a
        # vault path is configured; without it `sudo -S` would hang the
        # expect session on a password prompt.
        probes.append(_DOCKER_PROBE)

    bastion = llm_bastion_domain(inv, rack, cable)
    factory = console_factory or (
        lambda: ConsoleRunner(SerialConsole(domain, store, bastion=bastion)))
    runner = factory()
    try:
        results: list[CommandResult] = runner.batch_execute(probes)
    except Exception as exc:  # noqa: BLE001 - discovery never crashes the wizard
        return DiscoveryResult(
            notes=[(f"console hop failed ({exc}; tried via {domain.tool} @ "
                    f"{domain.address_for_rack()})")])

    result = DiscoveryResult()
    login_noted = False
    for probe, res in zip(probes, results):
        if res.exit_code != 0:
            result.notes.append(f"{probe}: {res.stderr.strip()[:200] or 'failed'}")
            continue
        if not login_noted and "login:" in (res.stdout or ""):
            # The serial console reattached at a getty login prompt (e.g.
            # after a node reboot): the probes were typed at the login header,
            # not a shell. Setup captures node credentials for the automatic
            # login handshake; a manual sol() session works too (the login
            # then persists on the console).
            login_noted = True
            result.notes.append(
                "node console at a login prompt -- rerun setup so the node "
                "login handshake runs, or log in once via your manual sol() "
                "session (the login persists on the console)")
        if probe.startswith("hostname"):
            result.addresses = parse_node_addresses(res.stdout)
        elif probe.startswith("ss"):
            result.ports = parse_listening_ports(res.stdout)
        elif probe.startswith("sudo"):
            result.containers = parse_docker_ports(res.stdout)
    if not result.addresses:
        result.notes.append("no node addresses parsed from `hostname -I`")
    if not result.suggested_ports():
        result.notes.append("no listening ports parsed from `ss`")
    return result

# ---- host-key pinning through the bastion ----

def pin_llm_host_key(rack: str, cable: str, inv: Inventory, store: SecretStore,
                     ) -> str:
    """Fetch the per-rack manager's SSH host key THROUGH the bastion and pin
    it into ``llm_console.known_hosts_path``.

    The fetch is the ``ssh-keyscan`` pattern: connect to the bastion (key
    auth, the proven workstation route), open a ``direct-tcpip`` channel to
    ``rackmgr:22``, run the SSH client-hello, and take the key the server
    offers -- no credentials ever reach the rack manager. Trust-gated to
    lab/qa like every console path. Returns a short summary of what was
    pinned; raises :class:`SerialConsoleError` (staged) on any failure.
    """
    defaults = llm_console(inv)
    if defaults is None or not defaults.bastion:
        raise SerialConsoleError(
            "host-key pinning through a bastion needs an llm_console block "
            "with bastion: set")
    if defaults.trust_level not in ("lab", "qa"):
        raise SerialConsoleError(
            f"host-key pinning allowed only at lab/qa, not "
            f"{defaults.trust_level!r}")
    bastion = llm_bastion_domain(inv, rack, cable)
    inner = defaults.for_rack(rack, cable)
    addr = inner.address_for_rack()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(_MissingHostReject())
    try:
        client.load_host_keys(bastion.known_hosts_path)
    except FileNotFoundError:
        pass
    key_path = load_key_material(store, bastion.identity_vault_path,
                                 Path(tempfile.gettempdir()))
    try:
        try:
            client.connect(
                hostname=bastion.address_for_rack(), username=bastion.user,
                key_filename=str(key_path), look_for_keys=False,
                allow_agent=False, timeout=30.0)
        except Exception as exc:
            raise SerialConsoleError(
                f"ssh to bastion {bastion.address_for_rack()} failed: {exc}"
            ) from exc
        try:
            sock = client.get_transport().open_channel(
                "direct-tcpip", (addr, 22), ("127.0.0.1", 0), timeout=30.0)
        except Exception as exc:
            raise SerialConsoleError(
                f"bastion {bastion.address_for_rack()} could not open a "
                f"channel to {addr}: {exc}") from exc
        try:
            transport = paramiko.Transport(sock=sock)
            try:
                transport.start_client(timeout=30.0)
                key = transport.get_remote_server_key()
            finally:
                transport.close()
        except Exception as exc:
            raise SerialConsoleError(
                f"could not read the host key of {addr}: {exc}") from exc
    finally:
        if key_path.exists():
            key_path.unlink()
        client.close()

    from .credential_gate import save_host_key

    save_host_key(inner.known_hosts_path, addr, key)
    return (f"{addr} {key.get_name()} "
            f"SHA256:{key.get_fingerprint().hex()}")
