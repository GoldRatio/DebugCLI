"""Target resolution: runtime spec -> connection plan, zero per-server YAML.

A ``TargetSpec`` is one of:

- ``name``: a named host already in the inventory (backward compat, unchanged).
- ``rack`` + ``cable``: a serial-console session through the fleet-level
  ``console_defaults`` block (no per-host YAML needed).
- ``ip``: a direct SSH session to that address; the identity key and
  known_hosts are resolved from the secret store.
- ``alias``: a short alias from ``harness targets`` (maps to rack/cable or IP).

``resolve_target`` always returns a ``Target`` carrying a ``Host`` (the real one
for named hosts, a synthetic one otherwise) so the rest of the pipeline --
session open, collectors, audit labels -- works unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..config.models import BMCDomain, ConsoleDefaults, ConsoleDomain, Host, Inventory, SSHDomain
from ..config.vault import SecretStore
from ..engine.sol import validate_identifier
from .aliases import load_targets

_DEFAULT_SSH_USER = "diagbot"
_DEFAULT_KNOWN_HOSTS = "config/known_hosts"
_LEGACY_IDENTITY = "secret/harness/diagbot/id_ed25519"


class TargetError(RuntimeError):
    pass


@dataclass(frozen=True)
class TargetSpec:
    """One runtime addressing spec; fields are mutually prioritised."""
    name: str | None = None
    rack: str | None = None
    cable: str | None = None
    ip: str | None = None
    alias: str | None = None


@dataclass(frozen=True)
class Target:
    kind: Literal["named", "console", "ssh"]
    label: str
    trust_level: str
    host: Host
    console: ConsoleDomain | None = None
    ip: str | None = None
    model_hint: str | None = None  # canonical model key from a target alias


def _dummy_bmc() -> BMCDomain:
    # A placeholder the pipeline never uses (empty vault path -> KeyError -> BMC
    # channel skipped). Dynamic targets have no BMC inventory entry.
    return BMCDomain(address="", username="", password_vault_path="")


def _synthetic_console_host(label: str, console: ConsoleDomain) -> Host:
    return Host(
        name=label,
        address="",
        model="unknown",
        ssh=SSHDomain(user="", identity_vault_path="", known_hosts_path=""),
        bmc=_dummy_bmc(),
        collector_profile="default",
        trust_level=console.trust_level,
        console=console,
    )


def _synthetic_ssh_host(ip: str, ssh: SSHDomain, trust_level: str) -> Host:
    return Host(
        name=ip,
        address=ip,
        model="unknown",
        ssh=ssh,
        bmc=_dummy_bmc(),
        collector_profile="default",
        trust_level=trust_level,
    )


def _canonical_rack_id(rack: str) -> str:
    """Canonicalize the fleet rack id so labels/audit stay consistent:
    ``61``/``q61``/``Q61`` all become ``Q61``; other ids pass through."""
    match = re.fullmatch(r"[qQ]?([0-9]+)", rack)
    if match:
        return f"Q{match.group(1)}"
    return rack


def _console_domain(inv: Inventory, rack: str, cable: str,
                    defaults: ConsoleDefaults | None = None) -> ConsoleDomain:
    defaults = defaults or inv.console_defaults
    if defaults is None:
        raise TargetError(
            "rack/cable targeting needs a fleet-level 'console_defaults:' block in "
            "the inventory (rack manager configured once, rack/cable per launch)")
    rack = _canonical_rack_id(validate_identifier(rack, "rack"))
    cable = validate_identifier(cable, "cable")
    if defaults.trust_level not in ("lab", "qa"):
        raise TargetError(
            f"console targeting blocked: console_defaults.trust_level is "
            f"{defaults.trust_level!r} (only lab/qa may dial the serial console)")
    return ConsoleDomain(
        address=defaults.address,
        user=defaults.user,
        identity_vault_path=defaults.identity_vault_path,
        known_hosts_path=defaults.known_hosts_path,
        rack=rack,
        cable=cable,
        tool=defaults.tool,
        trust_level=defaults.trust_level,
        prompts=defaults.prompts,
        port=defaults.port,
        sudo_vault_path=defaults.sudo_vault_path,
        rack_addresses=defaults.rack_addresses,
        bastion=defaults.bastion,
        password_vault_path=defaults.password_vault_path,
        node_user=defaults.node_user,
        node_password_vault_path=defaults.node_password_vault_path,
    )


def _identity_vault_path(store: SecretStore, ip: str, explicit: str | None) -> str:
    """Identity resolution for SSH-by-IP: explicit flag > per-IP path > legacy
    diagbot path; verified to exist in the store so failures are immediate and
    actionable (never a silent late SSH error)."""
    if explicit is not None:
        try:
            store.get(explicit)
        except KeyError:
            raise TargetError(
                f"identity vault path {explicit!r} not found in the secret store "
                "(register it with: harness secrets add-ssh <name> --key-file <path>)") from None
        return explicit
    for candidate in (f"secret/harness/ssh/{ip}", _LEGACY_IDENTITY):
        try:
            store.get(candidate)
            return candidate
        except KeyError:
            continue
    raise TargetError(
        f"no SSH identity for {ip!r} in the secret store (tried "
        f"'secret/harness/ssh/{ip}' and {_LEGACY_IDENTITY!r}); register one with: "
        f"harness secrets add-ssh {ip} --key-file <path>")


def resolve_target(
    spec: TargetSpec,
    inv: Inventory,
    store: SecretStore,
    *,
    targets_path: str | Path | None = None,
    ssh_user: str = _DEFAULT_SSH_USER,
    identity_vault_path: str | None = None,
    known_hosts_path: str = _DEFAULT_KNOWN_HOSTS,
    console_defaults: ConsoleDefaults | None = None,
) -> Target:
    """Resolve a runtime spec to a connection ``Target`` (never raises on bad
    credentials -- those surface as clear ``TargetError``s before any connection).

    ``console_defaults`` overrides the inventory's block (the LLM paths layer
    their own ``llm_console`` config over the same rack/cable addressing)."""
    if spec.name is not None:
        try:
            host = inv.get(spec.name)
        except KeyError:
            raise TargetError(
                f"unknown host {spec.name!r}; known: {', '.join(sorted(inv.host_names)) or '(none)'}") from None
        return Target(kind="named", label=host.name, trust_level=host.trust_level, host=host)

    if spec.alias is not None:
        aliases = load_targets(targets_path) if targets_path else {}
        entry = aliases.get(spec.alias)
        if entry is None:
            raise TargetError(
                f"unknown target alias {spec.alias!r} (see 'harness targets ls'; "
                f"file: {targets_path or 'none'})")
        alias_model = entry.model
        if entry.address is not None:
            spec = TargetSpec(ip=entry.address)
        else:
            spec = TargetSpec(rack=entry.rack, cable=entry.cable)
    else:
        alias_model = None

    if spec.rack is not None and spec.cable is not None:
        console = _console_domain(inv, spec.rack, spec.cable, defaults=console_defaults)
        label = f"{console.rack}-cable{console.cable}"
        return Target(
            kind="console", label=label, trust_level=console.trust_level,
            host=_synthetic_console_host(label, console), console=console,
            model_hint=alias_model,
        )

    if spec.ip is not None:
        identity = _identity_vault_path(store, spec.ip, identity_vault_path)
        ssh = SSHDomain(user=ssh_user, identity_vault_path=identity,
                        known_hosts_path=known_hosts_path)
        return Target(
            kind="ssh", label=spec.ip, trust_level=inv.trust_level,
            host=_synthetic_ssh_host(spec.ip, ssh, inv.trust_level), ip=spec.ip,
            model_hint=alias_model,
        )

    raise TargetError(
        "no target given: use --host <name>, --rack <r> --cable <n>, "
        "--address <ip>, or --target <alias>")
