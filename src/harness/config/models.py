"""Types for the host inventory, modeled on ``inventory.yaml``.

These are plain dataclasses (no pydantic) so ``config`` stays dependency-light and
everything below H-level in the dependency order can rely on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TrustLevel = Literal["lab", "qa", "prod"]


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


@dataclass(frozen=True)
class ConsoleDomain:
    """Serial-over-LAN console access via a rack manager.

    The primary path for rack units: SSH to the rack manager, then an interactive
    console (``expect``/``jumpin`` + ``start serial session -i <cable>``) to reach each
    server. ``rack``/``cable``/``tool`` are identifier-validated. Console access is
    gated by ``trust_level`` -- allowed only at ``lab``/``qa`` (never ``prod``).

    ``port`` selects the console service: none = server node console, ``2200`` =
    the BMC access port (BMC shell; ``i2cdump``/``i2cget`` etc. run there, usually
    under ``sudo -S``). ``sudo_vault_path`` is the vault path of the BMC sudo
    password -- never an inline secret.
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
    port: int | None = None        # BMC access port for the serial session (e.g. 2200)
    sudo_vault_path: str | None = None  # vault path of the BMC sudo password


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

    provider: Literal["openai", "gemini", "stub"] = "openai"
    model: str | None = None
    url: str | None = None
    api_key_vault_path: str | None = None
    timeout: float = 120.0


@dataclass(frozen=True)
class ConsoleDefaults:
    """Fleet-level rack-manager console configuration (configured ONCE).

    Per-target ``rack``/``cable`` are runtime parameters layered over this block,
    so no per-server YAML entry is needed for the console/SOL path.
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


@dataclass(frozen=True)
class Inventory:
    trust_level: TrustLevel
    hosts: list[Host] = field(default_factory=list)
    llm: LLMConfig | None = None
    console_defaults: ConsoleDefaults | None = None

    @property
    def host_names(self) -> frozenset[str]:
        return frozenset(h.name for h in self.hosts)

    def get(self, name: str) -> Host:
        for host in self.hosts:
            if host.name == name:
                return host
        raise KeyError(f"unknown host: {name!r}")