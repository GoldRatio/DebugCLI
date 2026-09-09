"""Types for the host inventory, modeled on ``inventory.yaml``.

These are plain dataclasses (no pydantic) so ``config`` stays dependency-light and
everything below H-level in the dependency order can rely on them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from typing import Literal

TrustLevel = Literal["lab", "qa", "prod"]


def _rack_digits(value: object) -> str:
    """Numeric core of a rack id: ``Q61``/``q61``/``61`` all -> ``61``."""
    m = re.search(r"\d+", str(value))
    return m.group() if m else str(value)


@dataclass(frozen=True)
class SSHDomain:
    """OS-level access. Key-based only; identity is a vault path, never a secret."""

    user: str
    identity_vault_path: str
    known_hosts_path: str
    bastion: str | None = None


@dataclass(frozen=True)
class BMCDomain:
    """BMC/IPMI access. Separate rotated user; never reuses the OS identity."""

    address: str
    username: str
    password_vault_path: str


# Probe-program routing tables for multi-service console targets (one node, one
# serial session per service port). BMC-shell programs run on the BMC access
# port (OpenBMC BusyBox; ``dmesg -r`` is the BMC ring buffer, as today); the
# host-SOL set runs on the node Linux console. A service entry overrides its
# table with ``programs:``.
BMC_SERVICE_PROGRAMS = ("ipmitool", "i2cdump", "i2cget", "i2cdetect", "i2ctransfer", "ls", "dmesg")
HOST_SERVICE_PROGRAMS = ("lspci", "dmidecode", "nvme", "smartctl",
                         "lsblk", "cat", "grep", "ss", "ip", "docker", "hostname")
# Collector names served by each service class when not configured explicitly.
BMC_SERVICE_SUBSYSTEMS = ("cpu_msr", "kernel", "ipmi")
HOST_SERVICE_SUBSYSTEMS = ("pcie", "storage")


def is_bmc_service_name(name: str) -> bool:
    """Name convention: a service named (or containing) ``bmc`` serves the
    BMC-shell program set unless it declares explicit ``programs``."""
    return "bmc" in name.lower()


def default_programs_for_service(name: str) -> tuple[str, ...]:
    return BMC_SERVICE_PROGRAMS if is_bmc_service_name(name) else HOST_SERVICE_PROGRAMS


def default_subsystems_for_service(name: str) -> tuple[str, ...]:
    return BMC_SERVICE_SUBSYSTEMS if is_bmc_service_name(name) else HOST_SERVICE_SUBSYSTEMS


@dataclass(frozen=True)
class ConsoleService:
    """One named node console service (a port) under ``console_defaults.services``.

    One debug session on one node then reaches several console services: the
    BMC access port (2200, OpenBMC BusyBox shell) and the host SOL port (22)
    are routed PER PROBE by program/command, never by the agent. ``port``/
    ``prompts``/``sudo_vault_path``/``node_user``/``node_password_vault_path``
    override the fleet defaults for this service; ``programs`` replaces the
    name-convention routing table (BMC set vs host set); ``subsystems`` moves
    collector names to this service. Vault paths only -- never inline secrets.
    """

    name: str  # identifier-stable label (lint enforces [A-Za-z0-9_-]+)
    port: int | None = None
    prompts: tuple[str, str] | None = None
    sudo_vault_path: str | None = None
    node_user: str | None = None
    node_password_vault_path: str | None = None
    programs: tuple[str, ...] = ()
    subsystems: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsoleDomain:
    """Serial-over-LAN console access via a rack manager.

    The primary path for rack units: SSH to the rack manager, then an interactive
    console (``expect`` + ``start serial session -i <cable>``) to reach each
    server. ``rack``/``cable``/``tool`` are identifier-validated. Console access is
    gated by ``trust_level`` -- allowed only at ``lab``/``qa`` (never ``prod``).

    ``tool`` selects how the session starts: ``jumpin`` spawns the jump CLI
    (``jumpin q<rack>-1 rm``) and waits for the rack-manager CLI prompt;
    ``direct`` sends ``start serial session`` straight in the SSH shell (no
    spawn, no prompt wait) -- the pattern of fleet managers that run the start
    command as a plain shell builtin.

    ``port`` selects the console service on the node: none = server node
    console, ``2200`` = the BMC access port (BMC shell; ``i2cdump``/``i2cget``
    etc. run there, usually under ``sudo -S``), ``22`` = host SOL (the host
    Linux shell where ``docker``/``ss`` live). ``sudo_vault_path`` is the vault
    path of the sudo password -- never an inline secret.

    ``rack_addresses`` optionally maps rack id -> rack-manager address for
    fleets where each rack has its own manager; ``address_for_rack`` falls
    back to ``address`` for racks absent from the map.

    ``bastion`` (address of a jump host) handles fleets where the rack
    managers sit on a network the workstation cannot route: the connection
    opens THROUGH the bastion (``direct-tcpip``) and nests a second SSH to
    the rack manager. ``password_vault_path`` is the vault path of the
    rack-manager SSH password for that nested hop (key auth is tried first;
    the password covers fleets without keys on the rack managers) -- never an
    inline secret.

    Redfish read-only collection (debug-step evidence fetched straight from the
    rack manager over HTTPS -- no serial session, no server jumpin) is enabled
    when ``redfish_password_vault_path`` is set. The per-node base URL is
    derived, never configured: ``https://<rack_ip>/<cable>/amc1/redfish/v1``.
    Only GET endpoints are ever contacted (event logs, service conditions);
    write endpoints (firmware update, power control) are unreachable by design.
    """

    address: str
    user: str
    identity_vault_path: str
    known_hosts_path: str
    rack: str
    cable: str
    tool: str = "jumpin"
    trust_level: TrustLevel = "prod"
    prompts: tuple[str, str] = ("RScmCli#", "~#")  # (rack_manager, node)
    port: int | None = None        # node console service port (e.g. 2200 BMC, 22 host SOL)
    sudo_vault_path: str | None = None  # vault path of the sudo password
    rack_addresses: dict[str, str] | None = None  # optional per-rack manager IPs
    redfish_user: str | None = None              # Redfish basic-auth user (default "root")
    redfish_password_vault_path: str | None = None  # set = Redfish collection enabled
    bastion: str | None = None     # jump-host address when the manager is unroutable
    password_vault_path: str | None = None  # vault path of the rackmgr SSH password
    node_user: str | None = None   # user the node serial console is logged in as
    node_password_vault_path: str | None = None  # vault path of the node login/sudo password
    # Multi-service routing (from a ConsoleDefaults.services entry): which probe
    # programs this service accepts and which collector names it serves. Empty
    # = derive from the service name convention ("bmc" -> BMC shell set,
    # otherwise the host-SOL set); see default_programs_for_service.
    programs: tuple[str, ...] = ()
    subsystems: tuple[str, ...] = ()
    service_name: str = "default"

    def address_for_rack(self, rack: str | None = None) -> str:
        """The manager to SSH to for ``rack``: the per-rack map entry when
        present, else the fleet-level ``address``. Rack spellings are compared
        numerically (``Q61``/``q61``/``61`` match)."""
        want = _rack_digits(rack or self.rack)
        for key, addr in (self.rack_addresses or {}).items():
            if _rack_digits(key) == want and addr:
                return addr
        return self.address


@dataclass(frozen=True)
class Host:
    name: str
    address: str
    model: str
    ssh: SSHDomain
    bmc: BMCDomain
    collector_profile: str
    trust_level: TrustLevel = "prod"
    console: ConsoleDomain | None = None


@dataclass(frozen=True)
class LLMConfig:
    """LLM backend for reasoning. ``api_key_vault_path`` is a vault path, never
    inline; absent = fall back to env (e.g. ``GEMINI_API_KEY``)."""

    provider: Literal["openai", "gemini", "local", "stub"] = "openai"
    model: str | None = None
    url: str | None = None
    api_key_vault_path: str | None = None
    timeout: float = 120.0


@dataclass(frozen=True)
class ConsoleDefaults:
    """Fleet-level rack-manager console configuration (configured ONCE).

    Per-target ``rack``/``cable`` are runtime parameters layered over this block,
    so no per-server YAML entry is needed for the console/SOL path. ``tool``
    and ``port`` mirror :class:`ConsoleDomain` (``jumpin``/``direct``; node
    console service port). ``rack_addresses`` optionally maps rack id ->
    manager IP when each rack runs its own manager. The ``redfish_*`` fields
    mirror :class:`ConsoleDomain` and enable rack-level Redfish GET collection.

    The same shape doubles as the inventory's optional ``llm_console`` block:
    a SEPARATE console configuration used only for LLM endpoint discovery and
    the LLM tunnel hop, for fleets where the golden server is reached with a
    different tool/port/account than the debug console (e.g. ``direct`` +
    host-SOL port vs the ``jumpin`` debug path). Absent, the LLM paths fall
    back to ``console_defaults``.
    """

    address: str
    user: str
    identity_vault_path: str
    known_hosts_path: str = "config/known_hosts"
    tool: str = "jumpin"
    trust_level: TrustLevel = "lab"
    prompts: tuple[str, str] = ("RScmCli#", "~#")
    port: int | None = None
    sudo_vault_path: str | None = None
    rack_addresses: dict[str, str] | None = None
    redfish_user: str | None = None
    redfish_password_vault_path: str | None = None
    bastion: str | None = None
    password_vault_path: str | None = None
    node_user: str | None = None
    node_password_vault_path: str | None = None
    # Multi-service console access: one node, one serial session PER service
    # port (BMC shell 2200, host SOL 22, ...). None = single-service legacy
    # behavior from ``port``. Insertion order of the mapping fixes routing
    # precedence: the first service whose program table lists a command's
    # program serves it.
    services: dict[str, ConsoleService] | None = None
    # Optional BMC LAN credentials (address/username/password_vault_path) so
    # rack/cable targets get the ipmitool-over-LAN channel (IpmiCollector) in
    # the same run. Absent = BMC LAN stays unavailable for console targets.
    bmc: BMCDomain | None = None

    def address_for_rack(self, rack: str | None = None) -> str:
        """Per-rack manager address when mapped, else ``address``."""
        want = _rack_digits(rack or "")
        for key, addr in (self.rack_addresses or {}).items():
            if _rack_digits(key) == want and addr:
                return addr
        return self.address

    def for_rack(self, rack: str, cable: str) -> ConsoleDomain:
        """The same configuration as a ready :class:`ConsoleDomain` with the
        runtime ``rack``/``cable`` layered in (so per-rack addresses resolve).
        Identifier validation is the caller's job (the resolver validates the
        console path; the tunnel only uses the address)."""
        return ConsoleDomain(
            address=self.address, user=self.user,
            identity_vault_path=self.identity_vault_path,
            known_hosts_path=self.known_hosts_path,
            rack=rack, cable=cable, tool=self.tool,
            trust_level=self.trust_level, prompts=self.prompts,
            port=self.port, sudo_vault_path=self.sudo_vault_path,
            rack_addresses=self.rack_addresses,
            redfish_user=self.redfish_user,
            redfish_password_vault_path=self.redfish_password_vault_path,
            bastion=self.bastion,
            password_vault_path=self.password_vault_path,
            node_user=self.node_user,
            node_password_vault_path=self.node_password_vault_path)

    def consoles_for_rack(self, rack: str, cable: str) -> dict[str, ConsoleDomain]:
        """Per-service :class:`ConsoleDomain` map for one rack/cable target.

        Without ``services`` this is the legacy single ``{"default": ...}`` map
        built from ``port``. With ``services``, each entry becomes its own
        domain: the service's port/prompts/credential overrides are layered
        onto the fleet defaults, and the name convention fills empty
        ``programs``/``subsystems``."""
        if not self.services:
            return {"default": self.for_rack(rack, cable)}
        out: dict[str, ConsoleDomain] = {}
        for name, svc in self.services.items():
            base = self.for_rack(rack, cable)
            out[name] = _replace(
                base,
                port=svc.port if svc.port is not None else self.port,
                prompts=svc.prompts if svc.prompts is not None else self.prompts,
                sudo_vault_path=(svc.sudo_vault_path
                                 if svc.sudo_vault_path is not None
                                 else self.sudo_vault_path),
                node_user=(svc.node_user if svc.node_user is not None
                           else self.node_user),
                node_password_vault_path=(
                    svc.node_password_vault_path
                    if svc.node_password_vault_path is not None
                    else self.node_password_vault_path),
                programs=svc.programs or default_programs_for_service(name),
                subsystems=svc.subsystems or default_subsystems_for_service(name),
                service_name=name,
            )
        return out


@dataclass(frozen=True)
class Inventory:
    trust_level: TrustLevel
    hosts: list[Host] = field(default_factory=list)
    llm: LLMConfig | None = None
    console_defaults: ConsoleDefaults | None = None
    llm_console: ConsoleDefaults | None = None

    @property
    def host_names(self) -> frozenset[str]:
        return frozenset(h.name for h in self.hosts)

    def get(self, name: str) -> Host:
        for host in self.hosts:
            if host.name == name:
                return host
        raise KeyError(f"unknown host: {name!r}")