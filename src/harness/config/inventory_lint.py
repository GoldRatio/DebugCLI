"""Lint the inventory YAML.

Guarantees the invariant: inventory holds *vault paths*, never credentials.
Any literal-looking secret (long token, base64 blob, private key material) is a
hard error so credentials cannot silently land in the repo.

Invoked by CI and by ``Session.open`` before connecting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import (
    BMCDomain,
    ConsoleDefaults,
    ConsoleDomain,
    Host,
    Inventory,
    LLMConfig,
    SSHDomain,
)

# Heuristic indicators of an inline secret. Kept intentionally simple; a real
# deployment would also call a secrets-scanner.
_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9+/]{64,}={0,2}\b"),  # >=64 char base64-ish blob
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),  # AWS access-key style
]

_PATH_KEYS = {
    "identity_vault_path",
    "password_vault_path",
    "known_hosts_path",
    "console_identity_vault_path",
    "sudo_vault_path",
    "api_key_vault_path",
    "redfish_password_vault_path",
}


@dataclass(frozen=True)
class InventoryIssue:
    host: str | None
    field: str
    message: str


def _looks_like_secret(value: str) -> bool:
    return any(p.search(value) for p in _SECRET_PATTERNS)


def _lint_value(host: str | None, field: str, value: object, issues: list[InventoryIssue]) -> None:
    if not isinstance(value, str):
        return
    if _looks_like_secret(value):
        issues.append(InventoryIssue(host, field, "value looks like an inline secret; use a vault path"))

    if field.rsplit(".", 1)[-1] in _PATH_KEYS and value.startswith(("secret/", "config/")):
        return
    if field.rsplit(".", 1)[-1] in _PATH_KEYS:
        issues.append(InventoryIssue(host, field, "expected a vault/config path, got non-path value"))


def lint_inventory(inv: Inventory) -> list[InventoryIssue]:
    issues: list[InventoryIssue] = []
    for host in inv.hosts:
        _lint_value(host.name, "address", host.address, issues)
        _lint_value(host.name, "ssh.identity_vault_path", host.ssh.identity_vault_path, issues)
        _lint_value(host.name, "ssh.user", host.ssh.user, issues)
        _lint_value(host.name, "bmc.address", host.bmc.address, issues)
        _lint_value(host.name, "bmc.username", host.bmc.username, issues)
        _lint_value(host.name, "bmc.password_vault_path", host.bmc.password_vault_path, issues)
        if host.console is not None:
            _lint_value(host.name, "console.identity_vault_path", host.console.identity_vault_path, issues)
            _lint_value(host.name, "console.user", host.console.user, issues)
            _lint_value(host.name, "console.rack", host.console.rack, issues)
            _lint_value(host.name, "console.cable", host.console.cable, issues)
            if host.console.sudo_vault_path is not None:
                _lint_value(host.name, "console.sudo_vault_path", host.console.sudo_vault_path, issues)
            if host.console.redfish_password_vault_path is not None:
                _lint_value(host.name, "console.redfish_password_vault_path",
                            host.console.redfish_password_vault_path, issues)
            if host.console.port is not None and not 1 <= host.console.port <= 65535:
                issues.append(InventoryIssue(host.name, "console.port",
                                             f"port {host.console.port} out of range 1-65535"))
    if inv.llm is not None and inv.llm.api_key_vault_path is not None:
        _lint_value(None, "llm.api_key_vault_path", inv.llm.api_key_vault_path, issues)
    if inv.console_defaults is not None:
        d = inv.console_defaults
        _lint_value(None, "console_defaults.identity_vault_path", d.identity_vault_path, issues)
        _lint_value(None, "console_defaults.known_hosts_path", d.known_hosts_path, issues)
        if d.sudo_vault_path is not None:
            _lint_value(None, "console_defaults.sudo_vault_path", d.sudo_vault_path, issues)
        if d.redfish_password_vault_path is not None:
            _lint_value(None, "console_defaults.redfish_password_vault_path",
                        d.redfish_password_vault_path, issues)
        if d.password_vault_path is not None:
            _lint_value(None, "console_defaults.password_vault_path",
                        d.password_vault_path, issues)
        if d.node_password_vault_path is not None:
            _lint_value(None, "console_defaults.node_password_vault_path",
                        d.node_password_vault_path, issues)
        if d.port is not None and not 1 <= d.port <= 65535:
            issues.append(InventoryIssue(None, "console_defaults.port",
                                         f"port {d.port} out of range 1-65535"))
    if inv.llm_console is not None:
        d = inv.llm_console
        _lint_value(None, "llm_console.identity_vault_path", d.identity_vault_path, issues)
        _lint_value(None, "llm_console.known_hosts_path", d.known_hosts_path, issues)
        if d.sudo_vault_path is not None:
            _lint_value(None, "llm_console.sudo_vault_path", d.sudo_vault_path, issues)
        if d.redfish_password_vault_path is not None:
            _lint_value(None, "llm_console.redfish_password_vault_path",
                        d.redfish_password_vault_path, issues)
        if d.password_vault_path is not None:
            _lint_value(None, "llm_console.password_vault_path",
                        d.password_vault_path, issues)
        if d.node_password_vault_path is not None:
            _lint_value(None, "llm_console.node_password_vault_path",
                        d.node_password_vault_path, issues)
        if d.port is not None and not 1 <= d.port <= 65535:
            issues.append(InventoryIssue(None, "llm_console.port",
                                         f"port {d.port} out of range 1-65535"))
    return issues


def load_inventory(path: str | Path) -> Inventory:
    """Load and lint a single-inventory YAML file; raise on any secret-like content."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    inv = _from_mapping(raw)
    issues = lint_inventory(inv)
    if issues:
        raise InventoryError(issues)
    return inv


class InventoryError(RuntimeError):
    def __init__(self, issues: list[InventoryIssue]) -> None:
        self.issues = issues
        super().__init__("inventory failed lint:\n" + "\n".join(f"  {i.host}.{i.field}: {i.message}" for i in issues))


def _from_mapping(raw: dict | None) -> Inventory:
    if not isinstance(raw, dict):
        raise InventoryError([InventoryIssue(None, "root", "inventory must be a mapping")])

    trust = raw.get("trust_level", "prod")
    hosts = []
    llm = _llm_from_mapping(raw.get("llm"))
    console_defaults = _console_defaults_from_mapping(raw.get("console_defaults"), trust)
    llm_console = _console_defaults_from_mapping(raw.get("llm_console"), trust)
    for entry in raw.get("hosts", []) or []:
        ssh = entry["ssh"]
        bmc = entry["bmc"]
        console_raw = entry.get("console")
        console = None
        if isinstance(console_raw, dict):
            prompts = console_raw.get("prompts", ("RScmCli#", "~#"))
            port_raw = console_raw.get("port")
            console = ConsoleDomain(
                address=console_raw["address"],
                user=console_raw["user"],
                identity_vault_path=console_raw["identity_vault_path"],
                known_hosts_path=console_raw.get("known_hosts_path", "config/known_hosts"),
                rack=console_raw["rack"],
                cable=console_raw["cable"],
                tool=console_raw.get("tool", "jumpin"),
                trust_level=console_raw.get("trust_level", trust),
                prompts=(prompts[0], prompts[1]) if prompts else ("RScmCli#", "~#"),
                port=int(port_raw) if port_raw is not None else None,
                sudo_vault_path=console_raw.get("sudo_vault_path"),
                redfish_user=console_raw.get("redfish_user"),
                redfish_password_vault_path=console_raw.get("redfish_password_vault_path"),
                rack_addresses={str(k): str(v) for k, v in
                                (console_raw.get("rack_addresses") or {}).items()}
                                if console_raw.get("rack_addresses") else None,
                bastion=console_raw.get("bastion"),
                password_vault_path=console_raw.get("password_vault_path"),
                node_user=console_raw.get("node_user"),
                node_password_vault_path=console_raw.get("node_password_vault_path"),
            )
        hosts.append(
            Host(
                name=entry["name"],
                address=entry["address"],
                model=entry["model"],
                collector_profile=entry.get("collector_profile", "default"),
                trust_level=entry.get("trust_level", trust),
                ssh=SSHDomain(
                    user=ssh["user"],
                    identity_vault_path=ssh["identity_vault_path"],
                    known_hosts_path=ssh.get("known_hosts_path", "config/known_hosts"),
                    bastion=ssh.get("bastion"),
                ),
                bmc=BMCDomain(
                    address=bmc["address"],
                    username=bmc["username"],
                    password_vault_path=bmc["password_vault_path"],
                ),
                console=console,
            )
        )
    return Inventory(trust_level=trust, hosts=hosts, llm=llm,
                     console_defaults=console_defaults, llm_console=llm_console)


def _console_defaults_from_mapping(raw: object, trust: str) -> ConsoleDefaults | None:
    if not isinstance(raw, dict):
        return None
    prompts = raw.get("prompts", ("RScmCli#", "~#"))
    port_raw = raw.get("port")
    return ConsoleDefaults(
        address=raw["address"],
        user=raw["user"],
        identity_vault_path=raw["identity_vault_path"],
        known_hosts_path=raw.get("known_hosts_path", "config/known_hosts"),
        tool=raw.get("tool", "jumpin"),
        trust_level=raw.get("trust_level", trust),
        prompts=(prompts[0], prompts[1]) if prompts else ("RScmCli#", "~#"),
        port=int(port_raw) if port_raw is not None else None,
        sudo_vault_path=raw.get("sudo_vault_path"),
        rack_addresses={str(k): str(v) for k, v in
                        (raw.get("rack_addresses") or {}).items()}
                        if raw.get("rack_addresses") else None,
        bastion=raw.get("bastion"),
        password_vault_path=raw.get("password_vault_path"),
        node_user=raw.get("node_user"),
        node_password_vault_path=raw.get("node_password_vault_path"),
    )


def _llm_from_mapping(raw: object) -> LLMConfig | None:
    if not isinstance(raw, dict):
        return None
    provider = raw.get("provider", "openai")
    if provider not in ("openai", "gemini", "stub"):
        raise InventoryError([InventoryIssue(
            None, "llm.provider", f"unknown provider {provider!r} "
            "(expected openai | gemini | stub)")])
    return LLMConfig(
        provider=provider,
        model=raw.get("model"),
        url=raw.get("url"),
        api_key_vault_path=raw.get("api_key_vault_path"),
        timeout=float(raw.get("timeout", 120.0)),
    )