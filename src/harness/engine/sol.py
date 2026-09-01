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
import socket
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
# (a regex alternation for grep), never as a shell pipe; `@` is allowed so
# ``i2ctransfer`` write/read descriptors like ``w1@0xb`` are expressible.
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_./:+=,-|@]+$")

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

# Probe-did-not-run markers: the shell prints ``prog: not found`` (or sudo's
# ``sudo: prog: command not found``) while expect still exits 0. Without
# detection a missing tool would record a fake-ok probe with no register data.
_CMD_NOT_FOUND = re.compile(r":\s*(?:command\s+)?not found|No such file or directory")


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
    - ``command_check``: optional whole-command predicate over the positional
      tokens (e.g. ``i2ctransfer`` descriptor shape, read-before-write)
    """

    flags: frozenset[str] = frozenset()
    value_flags: frozenset[str] = frozenset()
    values: bool = False
    subcommands: frozenset[str] = frozenset()
    subcommand_values: dict[str, frozenset[str]] | None = None
    value_check: Callable[[str], bool] | None = None
    command_check: Callable[[list[str]], bool] | None = None


def _cat_value(value: str) -> bool:
    # `cat` may only read sysfs/procfs; never /dev or arbitrary files, and never
    # through `..` traversal (`/sys/../../etc/shadow`).
    parts = value.split("/")
    return value.startswith(("/sys/", "/proc/")) and ".." not in parts


def _dev_value(value: str) -> bool:
    return value.startswith("/dev/")


def _system_path_value(value: str) -> bool:
    # `ls` may only read system tool/layout paths; never /dev, arbitrary files,
    # or `..` traversal. Used for BMC tool-availability discovery.
    return (value.startswith(("/bin/", "/sbin/", "/usr/", "/opt/", "/var/",
                              "/lib/", "/etc/", "/run/", "/sys/", "/proc/"))
            and ".." not in value.split("/"))


# i2ctransfer message tokens: ``r<n>`` read descriptor, ``w<n>@0x..`` write
# descriptor, or a byte payload (``0x..`` / bare 2-hex-digit). Writing the
# register pointer (w1/w2) is part of the documented read protocol, so those
# are allowed -- but only as pointer writes, and never without a read.
_I2C_TRANSFER_TOKEN_RE = re.compile(
    r"^(?P<kind>[rw])(?P<count>\d+)(?:@(?P<addr>0x[0-9a-fA-F]{1,2}))?$")
_I2C_TRANSFER_BYTE_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{1,2}$")


def _i2ctransfer_command_check(tokens: list[str]) -> bool:
    """Whole-command gate for ``i2ctransfer``: read descriptors (``r<n>``) and
    register-pointer writes (``w1@addr``/``w2@addr``) with byte payloads only,
    and every command must contain at least one read. A ``w256@addr ...`` block
    write is not expressible.
    """
    reads = 0
    for tok in tokens:
        m = _I2C_TRANSFER_TOKEN_RE.match(tok)
        if m is None:
            if _I2C_TRANSFER_BYTE_RE.match(tok):
                continue
            return False
        kind, count, addr = m.group("kind"), int(m.group("count")), m.group("addr")
        if kind == "r":
            reads += 1
            continue
        if kind == "w" and 1 <= count <= 2 and addr is not None:
            continue
        return False
    return reads >= 1


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
    # ls on system paths only: read-only tool/layout discovery on the BMC.
    "ls": _ProbeRule(
        flags=frozenset({"-l", "-a", "-1", "-h", "-i"}),
        values=True,
        value_check=_system_path_value,
    ),
    # i2ctransfer: block read/write descriptors (e.g. ``w1@0xb 0x00 r256``).
    # The command_check keeps descriptor writes to register-pointer width
    # (w1/w2, per the vendor read sequences) and demands at least one read.
    "i2ctransfer": _ProbeRule(
        flags=frozenset({"-y", "-f", "-v"}),
        values=True,
        command_check=_i2ctransfer_command_check,
    ),
    # ipmitool run on the BMC console (already on the BMC; no -H/-U/-E pins).
    "ipmitool": _ProbeRule(
        subcommands=frozenset({"sensor", "sel", "fru", "sdr"}),
        subcommand_values={
            "sensor": frozenset({"list", "elist"}),
            "sel": frozenset({"list", "info"}),   # NOT "clear"/"delete"
            "fru": frozenset({"print"}),
            "sdr": frozenset({"list", "elist"}),
        },
    ),
    # ---- LLM endpoint discovery on the node (read-only listings only) ----
    # hostname: node IPv4 addresses for the rackmgr-side tunnel HOST.
    "hostname": _ProbeRule(
        flags=frozenset({"-I", "-A", "-f", "-s"}),
    ),
    # ss: listening TCP sockets (vLLM port discovery); no positionals, so
    # filters like `ss ... state X` / `ss ... dst X` are not expressible.
    "ss": _ProbeRule(
        flags=frozenset({"-l", "-t", "-n", "-p", "-4", "-6"}),
    ),
    # ip: address/route listings only. No positional values, so `ip addr add`
    # (or any mutation) is not expressible.
    "ip": _ProbeRule(
        subcommands=frozenset({"addr", "route"}),
        flags=frozenset({"-4", "-6", "-o", "-br"}),
    ),
    # docker: `ps` listing only (container port mappings). Subcommand
    # whitelist without subcommand_values => `run`/`rm`/`exec`/`kill` are
    # rejected outright and any positional after `ps` is rejected.
    "docker": _ProbeRule(
        subcommands=frozenset({"ps"}),
        flags=frozenset({"-a"}),
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

    if rule.command_check is not None:
        positionals = [t for t in parts[1:] if not t.startswith("-")]
        if not rule.command_check(positionals):
            raise SerialProbeDenied(
                f"command shape not read-only for {prog}: {positionals!r}")


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
                         sudo_password: str | None = None,
                         node_user: str | None = None,
                         node_password: str | None = None) -> str:
    """Render the ``expect -c`` script the harness pipes to the rack manager shell.

    ``rack``/``cable``/``tool`` are wrapped in single quotes in the Tcl ``spawn`` and
    ``send`` strings; values are identifier-validated to prevent injection. The rack
    id is normalized for the rack manager CLI: a leading ``q``/``Q`` is stripped and
    the rest lowercased, so ``Q61``, ``q61`` and ``61`` all render ``q61-1``. Each
    ``command`` is validated read-only. ``port`` is the node console service port
    (e.g. 2200 = BMC access port, 22 = host SOL); when set, the session is started
    with ``start serial session -i <cable> -p <port>`` and the ``sudo -S`` probes in
    ``commands`` get the password handshake: the script waits for the
    ``[sudo] password for`` prompt, sends the password, then waits for the node
    prompt again. If the serial session fails to start (rack manager prints
    ``Status Description:``), the script exits non-zero BEFORE any probe or the
    sudo password reaches the wire.

    ``tool`` selects the start mechanism: ``jumpin`` (default) spawns the jump
    CLI (``jumpin q<rack>-1 rm``) and waits for the rack-manager CLI prompt
    (``prompts.rack_manager``) before starting the session; ``direct`` runs
    ``start serial session`` straight in the rack manager's SSH shell -- the
    script is piped into a non-interactive bash that executes stdin lines as
    they arrive, so there is no prompt to wait for.

    ``node_user``/``node_password`` enable the node LOGIN handshake: when the
    serial console reattaches at a getty ``login:`` prompt (e.g. after a node
    reboot), the script logs in before running the probes. The password is
    the node user's own (sudo and login share it); never embedded in config.
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
    if (node_password is not None and not _SAFE_VALUE.match(node_password)):
        raise SerialProbeDenied("node login password contains unsafe characters")
    if node_password is not None and not node_user:
        raise SerialProbeDenied("node login needs node_user")
    if node_user:
        validate_identifier(node_user, "node user")
    commands = [validate_serial_probe(c) for c in commands]

    lines = ["expect -c '"]
    direct = tool == "direct"
    if direct:
        # No jump CLI: the fleet manager runs `start serial session` as a
        # plain shell builtin. Expect needs a process to drive, so spawn a
        # local interactive bash on the rack manager and run the session
        # through it. There is no prompt to wait for: the tty buffers the
        # start command until bash reads stdin.
        lines.append("    spawn /bin/bash")
    else:
        lines.append(f"    spawn {tool} {rack_token}-1 rm")
        lines.append(f'    expect "{prompts.rack_manager}"')
    port_arg = f" -p {port}" if port is not None else ""
    lines.append(f'    send "start serial session -i {cable}{port_arg}\\r"')
    # If the serial session cannot be established the rack manager prints
    # "Status Description: ... failed ..." and returns to its own prompt. The
    # expect script MUST die there (non-zero exit): otherwise the probes and the
    # sudo password would be typed into the rack manager CLI shell instead of
    # the node. The failure branch is listed first so it wins the match. When
    # the console reattaches at a getty login prompt, the node login handshake
    # runs first (the session then persists for later runs).
    lines.append("    expect {")
    lines.append('        "Status Description:" { exit 3 }')
    if node_user and node_password is not None:
        lines.append('        "login:" {')
        lines.append(f'            send "{_tcl_escape(node_user)}\\r"')
        lines.append('            expect "Password:"')
        lines.append(f'            send "{_tcl_escape(node_password)}\\r"')
        lines.append(f'            expect "{prompts.node}"')
        lines.append("        }")
    lines.append(f'        "{prompts.node}" {{}}')
    lines.append("    }")

    first_sudo = True
    for cmd in commands:
        lines.append(f'    send "{_tcl_escape(cmd)}\\r"')
        # Consume THIS command's echoed line first: the echo carries the node
        # prompt text as its prefix, so a plain prompt-expect would match the
        # echoed line and race the next send while the command is still
        # running. Prompt matching must only ever see real post-output prompts.
        lines.append(f'    expect "{_tcl_escape(cmd)}\\r"')
        if sudo_password is not None and cmd.startswith("sudo -S ") and first_sudo:
            # Password handshake for the first sudo only: openBMC prints
            # "Password: ", stock sudo prints "[sudo] password for ...".
            # Do NOT type the password into a prompt-less shell (it would echo
            # into the transcript and the shell would run it as a command) --
            # when the prompt never appears (already authenticated), nothing is
            # sent and the prompt-anchor expect below still confirms the run.
            # sudo's timestamp cache covers the remaining sudo probes in this
            # session; a mid-session re-prompt degrades to an unauthenticated
            # probe rather than a misdirected password.
            lines.append("    expect {")
            lines.append('        "password for" { '
                         f'send "{_tcl_escape(sudo_password)}\\r"' + " }")
            lines.append('        "Password:" { '
                         f'send "{_tcl_escape(sudo_password)}\\r"' + " }")
            lines.append("    }")
        lines.append(f'    expect "{prompts.node}"')
        if cmd.startswith("sudo -S "):
            first_sudo = False

    # Detach the serial session and capture the LAST probe's trailing output
    # before the expect process exits and kills the session.
    lines.append('    send "exit\\r"')
    if direct:
        # Back at the spawned rack-manager bash (unknown prompt text): give
        # the detach a beat, end the bash, and run to EOF.
        lines.append("    sleep 1")
        lines.append('    send "exit\\r"')
        lines.append("    expect eof")
    else:
        lines.append(f'    expect "{prompts.rack_manager}"')
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

    When the rack managers sit on a network the workstation cannot route
    (``bastion`` is set), the SSH connection opens THROUGH the bastion instead:
    paramiko connects to the bastion (key auth), opens a ``direct-tcpip``
    channel to ``rackmgr:22``, and nests a second SSH over that channel -- key
    auth first, then the vault-sourced password (``console.password_vault_path``)
    when the rack managers have no keys.
    """

    def __init__(self, console: ConsoleDomain, store: SecretStore,
                 tmp_dir: Path | None = None, timeout: float = 300.0,
                 bastion: ConsoleDomain | None = None) -> None:
        if console.trust_level not in ("lab", "qa"):
            raise SerialConsoleError(
                f"serial console allowed only at lab/qa, not {console.trust_level}")
        self.console = console
        self.bastion = bastion
        self._store = store
        self._tmp_dir = Path(tmp_dir or tempfile.gettempdir())
        self._timeout = timeout
        self._client: paramiko.SSHClient | None = None
        self._bastion_client: paramiko.SSHClient | None = None
        self.log: list[str] = []

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
            # Unreachable / refused / auth failure must surface as a staged,
            # caught error (the discovery batch turns it into probe notes and
            # the wizard falls back to manual entry) -- never a raw traceback.
            # When auth itself failed, name the rejected methods so the fix is
            # obvious (key not installed / wrong vault password).
            methods = f"key ({key_path.name})" if key_path else "key"
            if password is not None:
                methods += " + vault password"
            hint = ""
            if "Authentication failed" in str(exc):
                hint = (f" -- both auth methods rejected for {user}@{hostname}"
                        + (f" (fix the material at {password_vault_path} or "
                           "install the harness key on the host)"
                           if password_vault_path else
                           " (install the harness key on the host)"))
            raise SerialConsoleError(
                f"ssh to {hostname} failed: {exc}{hint}") from exc

    def open(self) -> None:
        sock: socket.socket | None = None
        if self.bastion is not None:
            # Two-hop fleet: reach the rack manager THROUGH the bastion. The
            # bastion client uses the bastion domain's key auth; the nested
            # client rides a direct-tcpip channel to rackmgr:22.
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
            transport = self._bastion_client.get_transport()
            rackmgr_addr = self.console.address_for_rack()
            try:
                sock = transport.open_channel(
                    "direct-tcpip", (rackmgr_addr, 22), ("127.0.0.1", 0),
                    timeout=min(self._timeout, 30.0))
            except Exception as exc:
                raise SerialConsoleError(
                    f"bastion {self.bastion.address_for_rack()} could not open "
                    f"a channel to {rackmgr_addr}: {exc}") from exc

        client = paramiko.SSHClient()
        # Pin host keys; do NOT auto-accept (never replicate StrictHostKeyChecking=no).
        client.set_missing_host_key_policy(_MissingHostReject())
        try:
            client.load_host_keys(self.console.known_hosts_path)
        except FileNotFoundError:
            pass  # no pinned keys yet; _MissingHostReject fails closed
        key_path = load_key_material(self._store, self.console.identity_vault_path, self._tmp_dir)
        password = None
        if self.console.password_vault_path is not None:
            try:
                password = self._store.get(self.console.password_vault_path).decode(
                    errors="strict").rstrip("\r\n")
            except KeyError:
                raise SerialConsoleError(
                    "rack-manager password missing from vault: "
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

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._bastion_client is not None:
            self._bastion_client.close()
            self._bastion_client = None

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
        node_password = None
        if self.console.node_password_vault_path is not None:
            try:
                secret = self._store.get(self.console.node_password_vault_path)
            except KeyError:
                raise SerialConsoleError(
                    "node login password missing from vault: "
                    f"{self.console.node_password_vault_path!r}") from None
            node_password = secret.decode(errors="strict").rstrip("\r\n")
        script = render_expect_script(
            tool=self.console.tool, rack=self.console.rack, cable=self.console.cable,
            commands=commands, prompts=prompts,
            port=self.console.port, sudo_password=sudo_password,
            node_user=self.console.node_user, node_password=node_password,
        )
        start_cmd = (f"start serial session -i {self.console.cable}"
                     + (f" -p {self.console.port}" if self.console.port else ""))
        tried = (f"via {self.console.tool} @ {self.console.address_for_rack()}: "
                 f"{start_cmd!r}")
        stdin, stdout, stderr = self._client.exec_command("/bin/bash", timeout=self._timeout)
        stdin.write(script)
        stdin.channel.shutdown_write()
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        code = stdout.channel.recv_exit_status()
        combined = out + ("\n[stderr]\n" + err if err.strip() else "")
        if code != 0:
            raise SerialConsoleError(
                f"console script exited {code} ({tried}): {err.strip()[:300]}")
        if _DEAD_SESSION.search(out + "\n" + err):
            # The expect session died before the probes ran (e.g. the rack
            # manager rejected the session start, or the node prompt never
            # appeared): expect may still exit 0 while printing
            # "spawn id ... not open". A dead session must never masquerade
            # as successful probe output.
            raise SerialConsoleError(
                "serial session died before probes ran "
                f"({_DEAD_SESSION.search(out + chr(10) + err).group(0)!r}; "
                f"tried {tried})")
        self.log.append(script)
        return ConsoleResult(output=combined, probe_count=len(commands))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class _MissingHostReject(paramiko.client.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        raise paramiko.SSHException(f"host key for {hostname} not in pinned known_hosts")


_I2C_TOOLS_BARE = frozenset({"i2cdump", "i2ctransfer", "i2cget", "i2cdetect"})


def _absolutize_bmc_i2c_tools(cmd: str) -> str:
    """Use absolute paths for BMC i2c-tools programs.

    OpenBMC serial shells observed with PATH missing ``/usr/sbin``: the tools
    are installed (``/usr/sbin/i2cdump -> i2cdump.i2c-tools``) yet every bare
    invocation fails with ``command not found``. Rewriting bare i2c-tools
    commands to ``/usr/sbin/<tool>`` makes them run regardless of PATH.
    """
    parts = cmd.split()
    if not parts:
        return cmd
    prog_i = 2 if parts[0] == "sudo" and len(parts) > 2 and parts[1] == "-S" else 0
    if "/" in parts[prog_i]:
        return cmd
    prog = parts[prog_i].split("/")[-1]
    if prog not in _I2C_TOOLS_BARE:
        return cmd
    parts[prog_i] = f"/usr/sbin/{prog}"
    return " ".join(parts)


_NAME_BEFORE_ERROR = re.compile(r":\s*([^\s:\"]+)\"?\s*$")
_EXEC_MISSING = re.compile(r'can\'t exec "([^"]+)": No such file or directory')


def _probe_program(cmd: str) -> str:
    parts = cmd.split()
    if parts[:2] == ["sudo", "-S"]:
        parts = parts[2:]
    if not parts:
        return ""
    return parts[0].rsplit("/", 1)[-1]


def _not_found_error(output: str, cmd: str) -> str | None:
    """Return the shell error line for a probe result ONLY when the error names
    the probe's own program (``sudo: i2cdump: command not found``). Stray
    session noise (e.g. a password send landing at a bare prompt becomes
    ``-sh: 0penBmc: command not found``) never references the probe program and
    must not fake a 127 over real probe output."""
    prog = _probe_program(cmd)
    for ln in output.splitlines():
        m = _CMD_NOT_FOUND.search(ln)
        if m is None:
            continue
        failed = None
        am = _NAME_BEFORE_ERROR.search(ln[:m.start()])
        if am is not None:
            failed = am.group(1)
        else:
            em = _EXEC_MISSING.search(ln)
            if em is not None:
                failed = em.group(1)
        if failed is not None and (
                failed == prog
                or (failed == "sudo" and cmd.startswith("sudo -S "))):
            return ln
    return None


def _split_batch_output(transcript: str, wire_cmds: list[str]) -> list[str]:
    """Slice one console transcript into per-command blocks, line-aware.

    The shell echoes each sent command as a line ending in the command text
    (``<node prompt><cmd>``), while expect's ``send:`` debug lines never end
    with the command. Each block spans the lines after a command's echo up to
    the next command's echo (or the transcript end).
    """
    lines = transcript.splitlines()
    blocks: list[str] = []
    anchor = 0
    for i, cmd in enumerate(wire_cmds):
        start = None
        j = anchor
        while j < len(lines):
            if lines[j].rstrip().endswith(cmd):
                start = j
                j += 1
                break
            j += 1
        if start is None:
            blocks.append("")
            anchor = len(lines)
            continue
        end = len(lines)
        if i + 1 < len(wire_cmds):
            for k in range(j, len(lines)):
                if lines[k].rstrip().endswith(wire_cmds[i + 1]):
                    end = k
                    break
        blocks.append("\n".join(lines[start + 1:end]))
        anchor = j
    return blocks


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

    def __init__(self, console: SerialConsole,
                 on_probe: Callable[[CommandResult], None] | None = None) -> None:
        self._console = console
        self.calls: list[CommandResult] = []
        # Optional live listener: fired per recorded result (UI streaming).
        self.on_probe = on_probe
        # Results keyed by the WIRE command (absolute paths): a plan-level
        # pre-batch runs every probe once in ONE console session; later
        # per-collector executions dedupe against this cache.
        self.probe_cache: dict[str, CommandResult] = {}

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
        wire_cmd = _absolutize_bmc_i2c_tools(cmd)
        if wire_cmd != cmd:
            # The wire always carries a gate-checked command; the recorded argv
            # keeps the planned form (source strings, traces, audit stay stable).
            validate_serial_probe(wire_cmd)
        cached = self.probe_cache.get(wire_cmd)
        if cached is not None:
            return cached    # already executed in the plan-level batch session
        start = time.monotonic()
        try:
            result = self._console.run_probes([wire_cmd])
            out = CommandResult(argv=list(argv), stdout=result.output, stderr="",
                                exit_code=0, elapsed_ms=0)
        except SerialConsoleError as exc:
            out = CommandResult(argv=list(argv), stdout="", stderr=str(exc),
                                exit_code=1, elapsed_ms=0)
        if out.exit_code == 0:
            m = _not_found_error(out.stdout, " ".join(argv))
            if m is not None:
                # The probe program does not exist on the console: expect exits
                # 0 while the shell prints ``prog: not found``. Fake-ok dumps
                # with no register data must never reach the decoder/LLM.
                out.exit_code = 127
                out.stderr = f"probe did not run on console: {m}"
        out.elapsed_ms = int((time.monotonic() - start) * 1000)
        return self._record(out)

    def _record(self, result: CommandResult) -> CommandResult:
        self.calls.append(result)
        if self.on_probe is not None:
            self.on_probe(result)
        return result

    def batch_execute(self, cmds: list[str], timeout: float = 300.0) -> list[CommandResult]:
        """Run several probes in ONE console session (one rack-manager hop +
        one serial-session start), returning per-command results in order.

        ``cmds`` are probe strings (e.g. ``sudo -S ipmitool sensor list``);
        each is gate-checked individually; the recorded argv keeps the planned
        form. Sequential stop-on-success chains should still use ``execute``
        per probe -- batching runs every command regardless.
        """
        wire_list: list[str] = []
        denied: dict[str, CommandResult] = {}
        for cmd in cmds:
            argv = shlex.split(cmd)
            security_check(argv)
            try:
                validate_serial_probe(cmd)
            except SerialProbeDenied as exc:
                denied[cmd] = self._record(CommandResult(
                    argv=argv, stdout="", stderr=f"denied: {exc}",
                    exit_code=2, elapsed_ms=0))
                continue
            wire_cmd = _absolutize_bmc_i2c_tools(cmd)
            validate_serial_probe(wire_cmd)
            wire_list.append(wire_cmd)
        if not wire_list:
            return [denied[c] for c in cmds]
        start = time.monotonic()
        try:
            out = self._console.run_probes(wire_list)
            transcript = out.output
        except SerialConsoleError as exc:
            failed: list[CommandResult] = []
            for cmd in cmds:
                if cmd in denied:
                    failed.append(denied[cmd])
                else:
                    failed.append(self._record(CommandResult(
                        argv=shlex.split(cmd), stdout="", stderr=str(exc),
                        exit_code=1, elapsed_ms=0)))
            return failed
        elapsed = int((time.monotonic() - start) * 1000)
        blocks = _split_batch_output(transcript, wire_list)
        results: list[CommandResult] = []
        wire_idx = 0
        for cmd in cmds:
            argv = shlex.split(cmd)
            if cmd in denied:
                results.append(denied[cmd])
                continue
            block = blocks[wire_idx] if wire_idx < len(blocks) else ""
            result = CommandResult(argv=argv, stdout=block, stderr="",
                                   exit_code=0, elapsed_ms=elapsed)
            m = _not_found_error(result.stdout, cmd)
            if m is not None:
                # The probe program does not exist on the console: expect exits
                # 0 while the shell prints ``prog: not found``.
                result.exit_code = 127
                result.stderr = f"probe did not run on console: {m}"
            self.probe_cache[_absolutize_bmc_i2c_tools(cmd)] = result
            results.append(self._record(result))
            wire_idx += 1
        return results