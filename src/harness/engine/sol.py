"""Serial-over-LAN console access: SSH to a rack manager, then an interactive
console to the individual server (the ``jumpin``/``expect`` pattern).

The user's production script does exactly this: SSH to the rack manager IP, pipe an
``expect -c`` script that spawns a jump tool, ``start serial session -i <cable>``,
then run ``~#``-prompted commands on the server's console. That is the primary
access path for these rack units -- NOT a direct SSH to the server.

Security model (this is a distinct console path, not the allowlist ``Runner``):
  * The rack-manager SSH hop itself uses host-key pinning (never ``StrictHostKeyChecking=no``).
  * Every ``~#`` probe sent to the console is validated as read-only by
    ``validate_serial_probe`` before it is embedded in the script.
  * ``prod`` trust level disables SOL fallback entirely.
  * No raw writes are ever possible: probe validation forbids redirection,
    sub-shells, and destructive binaries.
"""

from __future__ import annotations

import re
import shlex
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import paramiko

from ..config.models import ConsoleDomain
from ..config.vault import SecretStore, load_key_material
from .runner import CommandResult
from .security_gate import check as security_check


class SerialProbeDenied(Exception):
    """A probe command was rejected as unsafe/not-read-only."""


class SerialConsoleError(RuntimeError):
    pass


# ---- probe validation (read-only, injection-safe) ----

# Constructs never allowed anywhere in a probe, even inside a pipeline segment or
# a value. `$` expansion is banned so a path cannot be rewritten to something
# outside the allowed set (e.g. `/sys/$X` -> `/etc/shadow`); `&` cannot background
# a second command; single `|` pipelines are allowed and validated per segment.
_DANGEROUS = re.compile(
    r"[><;&]|\|\||\$|`|\$\{|\n"
    r"|\b(?:sh|bash|python|perl|ruby|tee|dd|mkfs|shutdown|reboot|flashrom)\b"
)

# Value charset: a value is a single token -- no spaces, no shell metacharacters,
# no globbing, no expansion, no quotes. `|` is allowed inside a quoted token
# (a regex alternation for grep), never as a shell pipe.
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_./:+=,-|]+$")

# Values never acceptable anywhere (spec: `/dev/mem` is rejected outright).
_FORBIDDEN_VALUES = frozenset({"/dev/mem", "/dev/kmem"})

# Expect-session death markers: when the spawned jumpin/serial-session process
# dies (or a prompt times out), expect still exits 0 while printing these to
# stdout/stderr -- without detection the harness would record garbage as
# successful probe output.
_DEAD_SESSION = re.compile(
    r"spawn id exp\d+ not open|expect: timed out|child process exited abnormally"
    r"|Status Description: .*failed"
)


@dataclass(frozen=True)
class _ProbeRule:
    """Per-program read-only probe contract (deny-by-default).

    - ``flags``: exact tokens allowed on their own
    - ``value_flags``: tokens that must be followed by exactly one safe value
    - ``values``: bare positional values allowed (each still validated)
    - ``subcommands``: when non-empty, the first positional must be one of these
    - ``subcommand_values``: when given, positional values AFTER a subcommand must
      be in the subcommand's set (e.g. ``ipmitool sel`` only allows ``list``, so
      ``ipmitool sel clear`` is not expressible)
    - ``value_check``: optional extra predicate on a value (e.g. ``cat`` paths)
    """

    flags: frozenset[str] = frozenset()
    value_flags: frozenset[str] = frozenset()
    values: bool = False
    subcommands: frozenset[str] = frozenset()
    subcommand_values: dict[str, frozenset[str]] | None = None
    value_check: Callable[[str], bool] | None = None


def _cat_value(value: str) -> bool:
    # `cat` may only read sysfs/procfs; never /dev or arbitrary files, and never
    # through `..` traversal (`/sys/../../etc/shadow`).
    parts = value.split("/")
    return value.startswith(("/sys/", "/proc/")) and ".." not in parts


def _dev_value(value: str) -> bool:
    return value.startswith("/dev/")


# Read-only probe contracts, mirroring the allowlist philosophy: exact tokens,
# pinned flags. Destructive forms (`nvme format`, `smartctl -t/-s`, `dmesg -c`)
# are simply not expressible here. ``i2cset`` (the I2C WRITE tool) is absent by
# design -- only read tools (i2cdump/i2cget/i2cdetect) are allowed.
_PROBE_SPEC: dict[str, _ProbeRule] = {
    "lspci": _ProbeRule(
        flags=frozenset({"-xxx", "-x", "-vvv", "-v", "-n", "-nn", "-b", "-D", "-t", "-k"}),
        value_flags=frozenset({"-s", "-d"}),
    ),
    "dmidecode": _ProbeRule(value_flags=frozenset({"-t", "-s"})),
    "dmesg": _ProbeRule(
        flags=frozenset({"-x", "-t", "-T", "-d", "-u", "-e", "-E", "-r"}),
        value_flags=frozenset({"-l", "-p", "-s"}),
    ),
    "nvme": _ProbeRule(
        subcommands=frozenset({"list", "smart-log", "id-ctrl", "error-log", "id-ns", "ns-descs"}),
        values=True,
        value_check=_dev_value,
    ),
    "smartctl": _ProbeRule(
        value_flags=frozenset({"-a", "-x"}),
        value_check=_dev_value,
    ),
    "lsblk": _ProbeRule(
        flags=frozenset({"-f", "-d", "-n", "-l", "-p", "-t"}),
        value_flags=frozenset({"-o", "-S", "-I"}),
        values=True,
    ),
    "cat": _ProbeRule(values=True, value_check=_cat_value),
    "grep": _ProbeRule(
        flags=frozenset({"-E", "-i", "-w", "-x", "-n", "-c", "-l", "-o", "-v", "-H", "-h"}),
        value_flags=frozenset({"-e", "-m", "-A", "-B", "-C"}),
        values=True,
    ),
    # BMC console (serial session on the BMC access port): I2C register reads.
    "i2cdump": _ProbeRule(
        flags=frozenset({"-y", "-f", "-a"}),
        value_flags=frozenset({"-r"}),
        values=True,
    ),
    "i2cget": _ProbeRule(
        flags=frozenset({"-y", "-f", "-a"}),
        value_flags=frozenset({"-m"}),
        values=True,
    ),
    "i2cdetect": _ProbeRule(
        flags=frozenset({"-y", "-a", "-l", "-q", "-r"}),
        values=True,
    ),
    # ipmitool run on the BMC console (already on the BMC; no -H/-U/-E pins).
    "ipmitool": _ProbeRule(
        subcommands=frozenset({"sensor", "sel", "fru"}),
        subcommand_values={
            "sensor": frozenset({"list", "elist"}),
            "sel": frozenset({"list", "info"}),   # NOT "clear"/"delete"
            "fru": frozenset({"print"}),
        },
    ),
}


def _validate_value(value: str, rule: _ProbeRule) -> None:
    if value in _FORBIDDEN_VALUES:
        raise SerialProbeDenied(f"forbidden value {value!r}")
    if not _SAFE_VALUE.match(value):
        raise SerialProbeDenied(f"unsafe value {value!r}")
    if rule.value_check is not None and not rule.value_check(value):
        raise SerialProbeDenied(f"value not allowed for this program: {value!r}")


def _validate_segment(parts: list[str]) -> None:
    if not parts:
        raise SerialProbeDenied("empty probe segment")
    if parts[0] == "sudo" and parts[1] == "-S":
        # The BMC console needs `sudo -S` (password via stdin) for i2c/ipmitool
        # reads. Only a read-only probe program may follow -- i2cset can never
        # appear, so `sudo -S i2cset ...` is not expressible.
        inner = parts[2:]
        if not inner:
            raise SerialProbeDenied("sudo -S must wrap a probe program")
        _validate_segment(inner)
        return
    prog = parts[0].split("/")[-1]
    rule = _PROBE_SPEC.get(prog)
    if rule is None:
        raise SerialProbeDenied(f"probe program not read-only: {parts[0]!r}")

    i = 1
    sub_seen = False
    subcommand: str | None = None
    while i < len(parts):
        token = parts[i]
        if token.startswith("-"):
            if token in rule.flags:
                i += 1
                continue
            if token in rule.value_flags:
                if i + 1 >= len(parts) or parts[i + 1].startswith("-"):
                    raise SerialProbeDenied(f"flag {token!r} for {prog} needs a value")
                _validate_value(parts[i + 1], rule)
                i += 2
                continue
            raise SerialProbeDenied(f"flag not allowed for {prog}: {token!r}")
        if rule.subcommands and not sub_seen:
            if token not in rule.subcommands:
                raise SerialProbeDenied(f"subcommand not allowed for {prog}: {token!r}")
            sub_seen = True
            subcommand = token
            i += 1
            continue
        allowed = (rule.subcommand_values or {}).get(subcommand) if subcommand else None
        if sub_seen and allowed is not None:
            if token not in allowed:
                raise SerialProbeDenied(
                    f"subcommand argument not allowed for {prog} {subcommand}: {token!r}")
            i += 1
            continue
        if not rule.values:
            raise SerialProbeDenied(f"unexpected argument for {prog}: {token!r}")
        _validate_value(token, rule)
        i += 1


def validate_serial_probe(cmd: str) -> str:
    """Validate a console-probe command; raise ``SerialProbeDenied`` if unsafe.

    The command may contain a ``|`` pipeline; the command is tokenized with
    ``shlex`` (so a pipe inside quotes stays one token) and every pipe segment is
    validated independently against the read-only probe spec (deny-by-default).
    """
    if not cmd or not isinstance(cmd, str):
        raise SerialProbeDenied("empty probe")
    if _DANGEROUS.search(cmd):
        raise SerialProbeDenied(f"disallowed construct in probe: {cmd!r}")
    try:
        parts = shlex.split(cmd)
    except ValueError:
        raise SerialProbeDenied(f"unbalanced quotes in probe: {cmd!r}") from None
    if not parts:
        raise SerialProbeDenied("empty probe")

    segment: list[str] = []
    for token in parts:
        if token == "|":
            _validate_segment(segment)
            segment = []
        else:
            segment.append(token)
    _validate_segment(segment)
    return cmd.strip()


# ---- rack-manager identifier validation (injection-safe into expect string) ----
_IDENT = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_identifier(value: str, label: str) -> str:
    if not _IDENT.match(value):
        raise SerialProbeDenied(f"invalid {label}: {value!r}")
    return value


# ---- expect script builder (mirrors the production pattern) ----

@dataclass(frozen=True)
class ExpectPrompts:
    rack_manager: str = "RScmCli#"
    node: str = "~#"


def _tcl_escape(text: str) -> str:
    # Inside the Tcl double-quoted send string: escape quotes, backslashes and
    # dollar signs so the echoed line matches the validated command exactly.
    return (text.replace(chr(92), chr(92) + chr(92))
                .replace(chr(34), chr(92) + chr(34))
                .replace("$", chr(92) + "$"))


def render_expect_script(*, tool: str, rack: str, cable: str,
                         commands: list[str],
                         prompts: ExpectPrompts | None = None,
                         port: int | None = None,
                         sudo_password: str | None = None) -> str:
    """Render the ``expect -c`` script the harness pipes to the rack manager shell.

    ``rack``/``cable``/``tool`` are wrapped in single quotes in the Tcl ``spawn`` and
    ``send`` strings; values are identifier-validated to prevent injection. The rack
    id is normalized for the rack manager CLI: a leading ``q``/``Q`` is stripped and
    the rest lowercased, so ``Q61``, ``q61`` and ``61`` all render ``q61-1``. Each
    ``command`` is validated read-only. ``port`` is the BMC access port (2200);
    when set, the session is started with ``start serial session -i <cable> -p <port>``
    and the ``sudo -S`` probes in ``commands`` get the password handshake:
    the script waits for the ``[sudo] password for`` prompt, sends the password,
    then waits for the node prompt again. If the serial session fails to start
    (rack manager prints ``Status Description:``), the script exits non-zero
    BEFORE any probe or the sudo password reaches the wire.
    """
    prompts = prompts or ExpectPrompts()
    validate_identifier(tool, "tool")
    validate_identifier(rack, "rack")
    validate_identifier(cable, "cable")
    rack_token = "q" + rack.lower().lstrip("q")
    if port is not None and not 1 <= port <= 65535:
        raise SerialProbeDenied(f"port out of range: {port!r}")
    if sudo_password is not None and not _SAFE_VALUE.match(sudo_password):
        raise SerialProbeDenied("sudo password contains unsafe characters")
    commands = [validate_serial_probe(c) for c in commands]

    lines = ["expect -c '"]
    lines.append(f"    spawn {tool} {rack_token}-1 rm")
    lines.append(f'    expect "{prompts.rack_manager}"')
    port_arg = f" -p {port}" if port is not None else ""
    lines.append(f'    send "start serial session -i {cable}{port_arg}\\r"')
    # If the serial session cannot be established the rack manager prints
    # "Status Description: ... failed ..." and returns to its own prompt. The
    # expect script MUST die there (non-zero exit): otherwise the probes and the
    # sudo password would be typed into the rack manager CLI shell instead of
    # the node. The failure branch is listed first so it wins the match.
    lines.append("    expect {")
    lines.append('        "Status Description:" { exit 3 }')
    lines.append(f'        "{prompts.node}" {{}}')
    lines.append("    }")

    for cmd in commands:
        lines.append(f'    send "{_tcl_escape(cmd)}\\r"')
        if sudo_password is not None and cmd.startswith("sudo -S "):
            # Password handshake: shell prompts for it on stdin, we feed it.
            lines.append('    expect "password for"')
            lines.append(f'    send "{_tcl_escape(sudo_password)}\\r"')
        lines.append(f'    expect "{prompts.node}"')

    lines.append('    send "exit\\r"')
    lines.append("'")
    return "\n".join(lines)


@dataclass
class ConsoleResult:
    output: str
    probe_count: int
    elapsed_ms: int = 0

    def probe_lines(self, pattern: str = r"LnkSta:") -> list[str]:
        """Lines matching a register pattern (example: the LnkSta link-speed check)."""
        return [ln for ln in self.output.splitlines() if re.search(pattern, ln)]


# ---- the console session itself ----

class SerialConsole:
    """Drive a rack manager SSH hop + serial console to run read-only probes.

    The transport is opened by paramiko to the rack manager; the expect script is
    piped to its (bash) stdin, mirroring the operator's ``write-output | ssh ... /bin/bash``.
    """

    def __init__(self, console: ConsoleDomain, store: SecretStore,
                 tmp_dir: Path | None = None, timeout: float = 300.0) -> None:
        if console.trust_level not in ("lab", "qa"):
            raise SerialConsoleError(
                f"serial console allowed only at lab/qa, not {console.trust_level}")
        self.console = console
        self._store = store
        self._tmp_dir = Path(tmp_dir or tempfile.gettempdir())
        self._timeout = timeout
        self._client: paramiko.SSHClient | None = None
        self.log: list[str] = []

    def open(self) -> None:
        client = paramiko.SSHClient()
        # Pin host keys; do NOT auto-accept (never replicate StrictHostKeyChecking=no).
        client.set_missing_host_key_policy(_MissingHostReject())
        client.load_host_keys(self.console.known_hosts_path)
        key_path = load_key_material(self._store, self.console.identity_vault_path, self._tmp_dir)
        try:
            client.connect(
                hostname=self.console.address,
                username=self.console.user,
                key_filename=str(key_path),
                look_for_keys=False,
                allow_agent=False,
                timeout=min(self._timeout, 30.0),
            )
        finally:
            if key_path.exists():
                key_path.unlink()
        self._client = client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def run_probes(self, commands: list[str]) -> ConsoleResult:
        """Build the expect script from read-only probes and run it on the console."""
        if self._client is None:
            self.open()
        prompts = ExpectPrompts(rack_manager=self.console.prompts[0], node=self.console.prompts[1])
        sudo_password = None
        if self.console.sudo_vault_path is not None:
            try:
                secret = self._store.get(self.console.sudo_vault_path)
            except KeyError:
                raise SerialConsoleError(
                    f"sudo password missing from vault: {self.console.sudo_vault_path!r}") from None
            sudo_password = secret.decode(errors="strict").rstrip("\r\n")
        script = render_expect_script(
            tool=self.console.tool, rack=self.console.rack, cable=self.console.cable,
            commands=commands, prompts=prompts,
            port=self.console.port, sudo_password=sudo_password,
        )
        stdin, stdout, stderr = self._client.exec_command("/bin/bash", timeout=self._timeout)
        stdin.write(script)
        stdin.channel.shutdown_write()
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        code = stdout.channel.recv_exit_status()
        combined = out + ("\n[stderr]\n" + err if err.strip() else "")
        if code != 0:
            raise SerialConsoleError(f"console script exited {code}: {err.strip()[:300]}")
        if _DEAD_SESSION.search(out + "\n" + err):
            # The expect session died before the probes ran (e.g. the rack
            # manager rejected `jumpin`, or the node prompt never appeared):
            # expect may still exit 0 while printing "spawn id ... not open".
            # A dead session must never masquerade as successful probe output.
            raise SerialConsoleError(
                "serial session died before probes ran "
                f"({_DEAD_SESSION.search(out + chr(10) + err).group(0)!r})")
        self.log.append(script)
        return ConsoleResult(output=combined, probe_count=len(commands))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _MissingHostReject(paramiko.client.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        raise paramiko.SSHException(f"host key for {hostname} not in pinned known_hosts")


class ConsoleRunner:
    """``Runner``-shaped adapter: executes read-only commands over the serial
    console (rack manager + cable), so the whole diagnostic pipeline (model
    detection, collectors) can run through the console path selected per launch.

    Security: every argv is passed through the hard read-only security gate AND
    the console probe spec (``validate_serial_probe``) before anything reaches
    the wire. Denied probes and console failures become failed
    ``CommandResult`` records -- the collector then records ``ok=False`` and the
    pipeline continues instead of crashing.
    """

    def __init__(self, console: SerialConsole) -> None:
        self._console = console
        self.calls: list[CommandResult] = []

    is_console = True

    def execute(self, argv: list[str], timeout: float = 300.0) -> CommandResult:
        security_check(argv)  # hard no-write guarantee, always on
        cmd = " ".join(argv)
        try:
            validate_serial_probe(cmd)
        except SerialProbeDenied as exc:
            return self._record(CommandResult(
                argv=list(argv), stdout="", stderr=f"denied: {exc}",
                exit_code=2, elapsed_ms=0))
        start = time.monotonic()
        try:
            result = self._console.run_probes([cmd])
            out = CommandResult(argv=list(argv), stdout=result.output, stderr="",
                                exit_code=0, elapsed_ms=0)
        except SerialConsoleError as exc:
            out = CommandResult(argv=list(argv), stdout="", stderr=str(exc),
                                exit_code=1, elapsed_ms=0)
        out.elapsed_ms = int((time.monotonic() - start) * 1000)
        return self._record(out)

    def _record(self, result: CommandResult) -> CommandResult:
        self.calls.append(result)
        return result