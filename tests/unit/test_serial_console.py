"""Serial-over-LAN console: expect/jumpin script rendering, probe validation,
rack-manager hop security, and the LnkSta (PCIe link-speed) example register.
"""

import pytest

from harness.config.models import ConsoleDomain
from harness.engine.sol import (
    ConsoleResult,
    SerialProbeDenied,
    render_expect_script,
    validate_identifier,
    validate_serial_probe,
)

# ---- probe validation ----

def test_probe_accepts_read_only_register_check():
    probe = 'lspci -vvv -n -s 00:06:00.0 | grep -E "LnkSta:"'
    assert validate_serial_probe(probe) == probe


def test_probe_rejects_destructive_and_injection():
    for bad in ("shutdown -h now", "dd if=/dev/zero of=/dev/sda",
                "cat /etc/passwd > /tmp/x", "echo x >> /etc/fstab",
                "lspci; rm -rf /", "$(whoami)"):
        with pytest.raises(SerialProbeDenied):
            validate_serial_probe(bad)


def test_probe_rejects_destructive_subcommands_and_smuggled_flags():
    for bad in (
        "nvme format /dev/nvme0n1",          # destructive nvme subcommand
        "nvme secure-erase /dev/nvme0n1",
        "nvme fw-download /dev/nvme0n1 -f x.img",
        "cat /dev/mem",                       # spec: /dev/mem rejected outright
        "cat /dev/kmem",
        "cat /etc/shadow",                    # cat only reads sysfs/procfs
        "smartctl -a -t short /dev/sda",      # flag-smuggling past pinned -a
        "smartctl -a -s on /dev/sda",
        "smartctl -t /dev/sda",
        "dmesg -c",                           # -c clears the kernel ring buffer
        "cat /sys/foo & nvme format /dev/nvme0n1",  # backgrounding a second command
        "cat /sys/$HOME/x",                   # variable expansion in a value
        "cat /sys/../../etc/shadow",
        "nvme format /dev/mem",
    ):
        with pytest.raises(SerialProbeDenied):
            validate_serial_probe(bad)


def test_probe_accepts_legit_read_only_forms():
    for good in (
        "lspci -xxx",
        "lspci -vvv -n -s 00:06:00.0",
        "dmidecode",
        "dmidecode -s product-name",
        "dmesg",
        "dmesg -l err",
        "dmesg -l err,crit,alert,emerg",
        "nvme list",
        "nvme smart-log /dev/nvme0n1",
        "smartctl -a /dev/sda",
        "smartctl -x /dev/nvme0n1",
        "lsblk",
        "lsblk -o NAME,SIZE",
        "cat /sys/class/net/eth0/operstate",
        "cat /proc/meminfo",
        'lspci -vvv -n -s 00:06:00.0 | grep -E "LnkSta:"',
        'lspci -vvv -n -s 00:06:00.0 | grep -i -E "link|status"',
        "i2cdump -y -f 0 0x51",              # BMC shell: I2C read
        "i2cdump -y -f 0 0x51 0x0 0x80",     # -r range form
        "i2cget -y -f 0 0x51 0x40",
        "i2cdetect -l",
        "i2ctransfer -y 8 w1@0xb 0x00 r256", # i2c-tools v4+ block read
        "sudo -S i2ctransfer -y 8 w1@0xb 0x00 r256",
        "ls -l /usr/sbin/i2cdump /usr/sbin/i2ctransfer",  # BMC tool discovery
        "ls /usr/bin",
        "sudo -S i2cdump -y -f 0 0x51",      # sudo -S wraps a read-only probe
        "sudo -S ipmitool sensor list",
        "sudo -S ipmitool sel list",
        "sudo -S ipmitool fru print",
        "ipmitool sdr",
        "ipmitool sdr elist",
        "ipmitool sensor elist",
    ):
        assert validate_serial_probe(good) == good


# ---- LLM endpoint discovery probes (read-only listings, exact-flag pinned) ----

def test_probe_accepts_llm_discovery_probes():
    for good in (
        "hostname -I",
        "hostname",
        "ss -l -t -n",
        "ss -l -t -n -p",
        "ss -4 -l -t",
        "ip -4 addr",
        "ip -br addr",
        "ip -o -4 route",
        "sudo -S docker ps",
        "docker ps",
        "docker ps -a",
    ):
        assert validate_serial_probe(good) == good


def test_probe_rejects_llm_discovery_mutations_and_bundles():
    for bad in (
        "ip addr add 10.0.0.9/24 dev eth0",   # mutation via positional value
        "ip route add blackhole 10.0.0.9",    # route mutation
        "ip addr flush dev eth0",             # second positional after subcommand
        "docker run -it alpine sh",           # subcommand whitelist: ps only
        "docker rm -f vllm",
        "docker exec vllm sh",
        "docker ps --format '{{.Names}}'",    # no value flags pinned for ps
        "ss -ltnp",                           # bundled short flags: not pinned
        "ss -l -t -n state established",      # no positionals on ss
        "hostname some-name",                 # no positionals on hostname
        "sudo -S docker rm -f vllm",          # sudo cannot smuggle mutations
    ):
        with pytest.raises(SerialProbeDenied):
            validate_serial_probe(bad)


def test_probe_rejects_i2c_write_and_destructive_sudo():
    for bad in (
        "i2cset -y 0 0x51 0x40 0xff",          # i2cset is the I2C WRITE tool
        "i2cset",                              # never expressible at all
        "ipmitool sel clear",                  # destructive ipmitool subcommand
        "ipmitool sel delete 5",               # destructive ipmitool subcommand
        "ipmitool sensor thresh set 0 80",     # write form
        "ipmitool raw 0x30 0x40",              # raw write not allowed
        "sudo -S i2cset -y 0 0x51 0x40 0xff",  # sudo cannot smuggle writes in
        "sudo -S shutdown -h now",
        "sudo -S",                             # sudo must wrap a probe program
        "sudo i2cdump -y -f 0 0x51",           # only -S form (stdin password)
        "sudo -S nvme format /dev/nvme0n1",
        "i2ctransfer -y 8 w256@0xb 0x00",      # block write of any size
        "i2ctransfer -y 8 w4@0xb 0x00 0x01 0x02 0x03 r1",  # >2-byte write
        "i2ctransfer -y 8 w1@0xb 0x00",        # write-only, no read at all
        "i2ctransfer -y 8 w1 0x00 r256",       # write descriptor needs @addr
        "ls /dev/sda",                         # ls on devices, not system dirs
        "ls ../../etc/passwd",                 # no path traversal
    ):
        with pytest.raises(SerialProbeDenied):
            validate_serial_probe(bad)


def test_identifier_validation_blocks_injection():
    with pytest.raises(SerialProbeDenied):
        validate_identifier("03; ls", "rack")
    with pytest.raises(SerialProbeDenied):
        validate_identifier("12`reboot`", "cable")
    assert validate_identifier("03", "rack") == "03"


# ---- expect script rendering ----

def test_render_expect_script_matches_production_pattern():
    script = render_expect_script(tool="jumpin", rack="03", cable="12", commands=[
        'lspci -vvv -n -s 00:06:00.0 | grep -E "LnkSta:"',
    ])
    assert "spawn jumpin q03-1 rm" in script
    assert "start serial session -i 12" in script
    assert 'send "lspci -vvv -n -s 00:06:00.0 | grep -E \\"LnkSta:\\"\\r"' in script
    assert 'send "exit\\r"' in script
    # node prompt wait after each probe, plus script run via expect -c
    assert "expect \"~#\"" in script
    assert script.startswith("expect -c '")
    # session failure is fatal BEFORE any probe or password is sent
    assert '        "Status Description:" { exit 3 }' in script
    assert "expect {" in script


def test_render_normalizes_rack_letter_case():
    for rack in ("Q61", "q61", "61"):
        script = render_expect_script(tool="jumpin", rack=rack, cable="6", commands=["dmesg -l err"])
        assert "spawn jumpin q61-1 rm" in script
        assert "qQ" not in script
    script = render_expect_script(tool="jumpin", rack="Q03", cable="12", commands=["dmesg -l err"])
    assert "spawn jumpin q03-1 rm" in script


def test_render_rejects_unsafe_command():
    with pytest.raises(SerialProbeDenied):
        render_expect_script(tool="jumpin", rack="03", cable="12", commands=["shutdown -h now"])


def test_render_includes_bmc_port():
    script = render_expect_script(tool="jumpin", rack="03", cable="12", commands=["lspci"],
                                  port=2200)
    assert "start serial session -i 12 -p 2200" in script
    script = render_expect_script(tool="jumpin", rack="03", cable="12", commands=["lspci"])
    assert "start serial session -i 12 -p" not in script


def test_render_rejects_bad_port():
    for port in (0, 65536, -1):
        with pytest.raises(SerialProbeDenied):
            render_expect_script(tool="jumpin", rack="03", cable="12", commands=["lspci"],
                                 port=port)


def test_render_sudo_password_handshake():
    script = render_expect_script(
        tool="jumpin", rack="03", cable="12",
        commands=["sudo -S i2cdump -y -f 0 0x51", "ipmitool sensor list"],
        port=2200, sudo_password="s3cret",
    )
    assert 'send "sudo -S i2cdump -y -f 0 0x51\\r"' in script
    # the echoed command line is consumed before prompt matching, so prompt
    # patterns only ever match real post-output prompts
    assert 'expect "sudo -S i2cdump -y -f 0 0x51\\r"' in script
    # handshake after the first sudo probe only; the password is sent ONLY when
    # the prompt appears (openBMC prints "Password: " not "[sudo] password for
    # ..."; cached sudo prompts nothing and must not get a stray password typed
    # at the bare shell prompt); plain probes get no password prompt
    assert '"password for" {' in script
    assert '"Password:" {' in script
    # one handshake block (one send per prompt pattern), first sudo only;
    # the other brace-block is the session-start failure branch
    assert script.count('send "s3cret\\r"') == 2
    assert script.count("expect {") == 2
    # failure branch comes before the node prompt branch and before the first
    # probe/password send (the session-start send naturally comes earlier)
    fail_idx = script.index('"Status Description:" { exit 3 }')
    node_idx = script.index('"~#" {}')
    probe_send = script.index('send "sudo -S i2cdump -y -f 0 0x51')
    assert fail_idx < node_idx and fail_idx < probe_send


def test_run_probes_failed_session_exits_before_password_sent():
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole, SerialConsoleError
    # rack manager rejects the serial session: "Status Description: ... failed"
    client = _FakeClient(exit_code=3,
        out=b"RScmCli# start serial session -i 6 -p 2200\r\n"
            b"ssh: connect to host 172.17.0.21 port 2200: No route to host\r\n"
            b"    Status Description: start_serial_session failed to connect.\r\n"
            b"    Completion Code: Failure\r\nRScmCli# ")
    sc = SerialConsole(_lab_console(), MemorySecretStore({"secret/bmc/sudo": b"0penBmc\n"}))
    sc._client = client
    # exit 3 = expect aborted on the session-failure branch (before any probe
    # or the sudo password could reach the wire; ordering is covered by
    # test_render_sudo_password_handshake)
    with pytest.raises(SerialConsoleError, match="console script exited 3"):
        sc.run_probes(["sudo -S ipmitool sensor list"])


def test_render_rejects_unsafe_sudo_password():
    with pytest.raises(SerialProbeDenied):
        render_expect_script(tool="jumpin", rack="03", cable="12",
                             commands=["lspci"], sudo_password="$(whoami)")


def test_console_result_probe_lines():
    res = ConsoleResult(output="~# LnkSta: Speed 8GT/s, Width x16\n~# LnkSta: Speed 2.5GT/s, Width x1\n", probe_count=2)
    lines = res.probe_lines(r"LnkSta:")
    assert len(lines) == 2
    assert "Width x1" in lines[1]  # slow link (degraded harddrive path)


# ---- console trust gating ----

def _console(**kw):
    base = {"address": "192.168.202.51", "user": "log", "identity_vault_path": "secret/rm",
            "known_hosts_path": "config/kh", "rack": "03", "cable": "12", "trust_level": "prod"}
    base.update(kw)
    return ConsoleDomain(**base)


def test_console_blocked_at_prod_by_default():
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole, SerialConsoleError
    with pytest.raises(SerialConsoleError):
        SerialConsole(_console(), MemorySecretStore())


def test_console_allowed_at_lab():
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole
    sc = SerialConsole(_console(trust_level="lab"), MemorySecretStore())
    assert sc.open is not None  # construction ok at lab


# ---- per-rack manager addresses ----

def test_address_for_rack_prefers_per_rack_map():
    d = _console(rack="Q71", rack_addresses={"Q61": "10.0.128.74",
                                             "Q71": "10.0.128.98"})
    assert d.address_for_rack() == "10.0.128.98"
    assert d.address_for_rack("q61") == "10.0.128.74"      # numeric compare
    assert d.address_for_rack("61") == "10.0.128.74"
    assert d.address_for_rack("Q99") == "192.168.202.51"   # unmapped -> default
    assert _console(rack="Q71").address_for_rack() == "192.168.202.51"


def test_console_open_connects_to_per_rack_address(tmp_path, monkeypatch):
    import harness.engine.sol as sol_mod

    connected = {}

    class _Client:
        def set_missing_host_key_policy(self, policy):
            pass

        def load_host_keys(self, path):
            pass

        def connect(self, **kw):
            connected["hostname"] = kw["hostname"]

    monkeypatch.setattr(sol_mod.paramiko, "SSHClient", _Client)
    monkeypatch.setattr(sol_mod, "load_key_material",
                        lambda store, vault, tmp: tmp_path / "id.pem")
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole

    sc = SerialConsole(
        _lab_console(rack="Q71",
                     rack_addresses={"Q71": "10.0.128.98"},
                     known_hosts_path=str(tmp_path / "kh")),
        MemorySecretStore({"secret/bmc/sudo": b"x\n"}),
    )
    sc.open()
    assert connected["hostname"] == "10.0.128.98"


def test_console_open_failure_is_staged_not_raw(tmp_path, monkeypatch):
    """An unreachable rack manager (timeout / refused / auth) surfaces as a
    staged SerialConsoleError naming the address -- never a raw paramiko
    traceback out of the wizard."""
    import harness.engine.sol as sol_mod

    class _DeadClient:
        def set_missing_host_key_policy(self, policy):
            pass

        def load_host_keys(self, path):
            pass

        def close(self):
            pass

        def connect(self, **kw):
            raise TimeoutError(10060, "connection attempt failed")

    monkeypatch.setattr(sol_mod.paramiko, "SSHClient", _DeadClient)
    monkeypatch.setattr(sol_mod, "load_key_material",
                        lambda store, vault, tmp: tmp_path / "id.pem")
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole, SerialConsoleError

    sc = SerialConsole(
        _lab_console(rack="Q71", rack_addresses={"Q71": "10.0.128.98"},
                     known_hosts_path=str(tmp_path / "kh")),
        MemorySecretStore({"secret/bmc/sudo": b"x\n"}),
    )
    with pytest.raises(SerialConsoleError,
                       match=r"ssh to 10\.0\.128\.98"):
        sc.open()


def test_console_bastion_chain_connects_through_jump_host(
        tmp_path, monkeypatch):
    """Two-hop fleet: paramiko reaches the BASTION with the debug console's
    key, then nests SSH to the per-rack manager over a direct-tcpip channel
    (password from the vault; key tried first)."""
    import harness.engine.sol as sol_mod

    connects = []

    class _Client:
        def __init__(self):
            self._transport = _FakeTransport()

        def get_transport(self):
            return self._transport

        def set_missing_host_key_policy(self, policy):
            pass

        def load_host_keys(self, path):
            pass

        def connect(self, **kw):
            connects.append({"hostname": kw["hostname"],
                             "user": kw["username"],
                             "password": kw.get("password"),
                             "sock": kw.get("sock")})

        def close(self):
            pass

    class _FakeTransport:
        def open_channel(self, kind, target, src, timeout=None):
            assert kind == "direct-tcpip"
            connects.append({"channel": target})
            return object()  # opaque sock for the nested connect

        def close(self):
            pass

    monkeypatch.setattr(sol_mod.paramiko, "SSHClient", _Client)
    monkeypatch.setattr(sol_mod, "load_key_material",
                        lambda store, vault, tmp: tmp_path / "id.pem")
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole

    bastion = _lab_console(rack="Q71", known_hosts_path=str(tmp_path / "kh"))
    inner = _lab_console(rack="Q71", tool="direct", port=22, user="root",
                         rack_addresses={"Q71": "10.0.128.98"},
                         password_vault_path="secret/rm-pw",
                         known_hosts_path=str(tmp_path / "kh"))
    sc = SerialConsole(inner, MemorySecretStore({"secret/rm-pw": b"s3cret\n"}),
                       bastion=bastion)
    sc.open()
    assert connects[0] == {"hostname": "192.168.202.51", "user": "log",
                           "password": None, "sock": None}       # bastion: key auth
    assert connects[1] == {"channel": ("10.0.128.98", 22)}       # through bastion
    assert connects[2]["hostname"] == "10.0.128.98"              # nested rackmgr
    assert connects[2]["user"] == "root"
    assert connects[2]["password"] == "s3cret"                   # vault password
    assert connects[2]["sock"] is not None
    sc.close()


def test_console_open_tolerates_missing_known_hosts(tmp_path, monkeypatch):
    """A known_hosts path whose parent dir does not exist (e.g. setup declined
    the rack-manager install) must not crash open() with FileNotFoundError; the
    host-key policy still fails closed."""
    import harness.engine.sol as sol_mod

    calls = {"load": 0, "connect": 0}

    class _Client:
        def set_missing_host_key_policy(self, policy):
            self.policy = policy

        def load_host_keys(self, path):
            calls["load"] += 1
            raise FileNotFoundError(f"no such directory: {path}")

        def connect(self, **kw):
            calls["connect"] += 1

    monkeypatch.setattr(sol_mod.paramiko, "SSHClient", _Client)
    monkeypatch.setattr(sol_mod, "load_key_material",
                        lambda store, vault, tmp: tmp_path / "id.pem")
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole

    sc = SerialConsole(
        _lab_console(known_hosts_path=str(tmp_path / "no" / "such" / "kh")),
        MemorySecretStore({"secret/bmc/sudo": b"x\n"}),
    )
    sc.open()
    assert calls["load"] == 1
    assert calls["connect"] == 1
    assert isinstance(sc._client, _Client)


class _FakeClient:
    def __init__(self, out=b"", err=b"", exit_code=0):
        self.written = ""
        self._out, self._err, self._exit_code = out, err, exit_code

    def exec_command(self, _shell, timeout=None):
        self.stdin = _FakeStream(self)
        self.stdout = _FakeStream(None, self._out, self._exit_code)
        self.stderr = _FakeStream(None, self._err, self._exit_code)
        return self.stdin, self.stdout, self.stderr


class _FakeStream:
    def __init__(self, client, data=b"", exit_code=0):
        self._client = client
        self._data = data
        self.channel = _FakeChannel(exit_code)

    def write(self, text):
        self._client.written += text

    def read(self):
        return self._data


class _FakeChannel:
    def __init__(self, exit_code=0):
        self._exit_code = exit_code

    def shutdown_write(self):
        pass

    def recv_exit_status(self):
        return self._exit_code


def _lab_console(**kw):
    base = {"trust_level": "lab", "port": 2200, "sudo_vault_path": "secret/bmc/sudo"}
    base.update(kw)
    return _console(**base)


def test_run_probes_resolves_sudo_from_store():
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole
    client = _FakeClient()
    sc = SerialConsole(
        _lab_console(),
        MemorySecretStore({"secret/bmc/sudo": b"0penBmc\n"}),
    )
    sc._client = client
    res = sc.run_probes(["sudo -S i2cdump -y -f 0 0x51", "ipmitool sensor list"])
    assert res.probe_count == 2
    assert "start serial session -i 12 -p 2200" in client.written
    assert 'send "sudo -S i2cdump -y -f 0 0x51\\r"' in client.written
    assert 'send "0penBmc\\r"' in client.written  # trailing newline stripped


def test_run_probes_direct_mode_renders_no_jumpin():
    """``tool: direct`` fleets (plain `start serial session` builtin in the
    rackmgr shell): no jumpin spawn, no CLI-prompt wait, host SOL port, and a
    clean double-exit to EOF."""
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole
    client = _FakeClient()
    sc = SerialConsole(
        _lab_console(tool="direct", port=22),
        MemorySecretStore({"secret/bmc/sudo": b"0penBmc\n"}),
    )
    sc._client = client
    sc.run_probes(["hostname -I", "sudo -S docker ps"])
    script = client.written
    assert "spawn jumpin" not in script
    assert 'expect "RScmCli#"' not in script
    assert "spawn /bin/bash" in script
    assert 'send "start serial session -i 12 -p 22\\r"' in script
    assert '"Status Description:" { exit 3 }' in script       # fail-fast kept
    assert 'send "hostname -I\\r"' in script
    assert 'send "sudo -S docker ps\\r"' in script
    assert 'send "0penBmc\\r"' in script                      # handshake kept
    assert script.count('send "exit\\r"') == 2                # detach + bash exit
    assert "expect eof" in script


def test_console_auth_failure_names_rejected_methods(tmp_path, monkeypatch):
    """When the rackmgr rejects key AND password auth, the staged error says
    which methods were tried and where to fix the material."""
    import harness.engine.sol as sol_mod

    class _AuthFailClient:
        def set_missing_host_key_policy(self, policy):
            pass

        def load_host_keys(self, path):
            pass

        def close(self):
            pass

        def connect(self, **kw):
            raise ConnectionError("Authentication failed.")

    monkeypatch.setattr(sol_mod.paramiko, "SSHClient", _AuthFailClient)
    monkeypatch.setattr(sol_mod, "load_key_material",
                        lambda store, vault, tmp: tmp_path / "id.pem")
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole, SerialConsoleError

    sc = SerialConsole(
        _lab_console(rack="Q71", user="root",
                     rack_addresses={"Q71": "10.0.128.98"},
                     password_vault_path="secret/rm-pw",
                     known_hosts_path=str(tmp_path / "kh")),
        MemorySecretStore({"secret/rm-pw": b"pw\n"}),
    )
    with pytest.raises(SerialConsoleError, match="both auth methods rejected"):
        sc.open()
    with pytest.raises(SerialConsoleError, match="secret/rm-pw"):
        sc.open()


def test_run_probes_dead_session_error_includes_tried_command():
    """The staged error names the exact start command + tool + address so a
    tool/port mismatch (jumpin vs direct, 2200 vs 22) is visible instantly."""
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole, SerialConsoleError
    client = _FakeClient(out=b"Status Description: start_serial_session failed\n")
    sc = SerialConsole(_lab_console(), MemorySecretStore({"secret/bmc/sudo": b"x\n"}))
    sc._client = client
    with pytest.raises(SerialConsoleError, match="tried via jumpin"):
        sc.run_probes(["lspci"])
    with pytest.raises(SerialConsoleError,
                       match=r"start serial session -i 12 -p 2200"):
        client.written = ""
        sc.run_probes(["lspci"])


def test_run_probes_missing_sudo_secret_raises():
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole, SerialConsoleError
    sc = SerialConsole(_lab_console(), MemorySecretStore())
    sc._client = _FakeClient()
    with pytest.raises(SerialConsoleError, match="missing from vault"):
        sc.run_probes(["lspci"])


def test_run_probes_no_sudo_when_path_unset():
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole
    client = _FakeClient()
    sc = SerialConsole(_console(trust_level="lab"), MemorySecretStore())
    sc._client = client
    sc.run_probes(["lspci"])
    assert '"password for"' not in client.written
    assert 'expect "lspci\\r"' in client.written


def test_run_probes_dead_session_raises_not_fake_ok():
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole, SerialConsoleError
    # expect exits 0 but the jumpin process died before any probe ran
    client = _FakeClient(
        out=b"spawn jumpin q61-1 rm\r\nYou are might not on right PXE\r\n",
        err=b"send: spawn id exp3 not open\n    while executing\n\"send\"")
    sc = SerialConsole(_lab_console(), MemorySecretStore({"secret/bmc/sudo": b"x\n"}))
    sc._client = client
    with pytest.raises(SerialConsoleError, match="serial session died"):
        sc.run_probes(["dmesg -r"])


def test_run_probes_prompt_timeout_raises_not_fake_ok():
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole, SerialConsoleError
    client = _FakeClient(err=b"expect: timed out\n    while executing\n\"expect\"")
    sc = SerialConsole(_lab_console(), MemorySecretStore({"secret/bmc/sudo": b"x\n"}))
    sc._client = client
    with pytest.raises(SerialConsoleError, match="serial session died"):
        sc.run_probes(["lspci -xxx"])


def test_run_probes_healthy_output_passes():
    from harness.config.vault import MemorySecretStore
    from harness.engine.sol import SerialConsole
    client = _FakeClient(
        out=b"RScmCli# start serial session -i 6 -p 2200\r\n"
            b"admin@m1120-c4a15:~$ ipmitool sensor list\r\nCPU0 Temp | ok\r\n"
            b"admin@m1120-c4a15:~$ ")
    sc = SerialConsole(_lab_console(), MemorySecretStore({"secret/bmc/sudo": b"x\n"}))
    sc._client = client
    res = sc.run_probes(["ipmitool sensor list"])
    assert res.probe_count == 1
    assert "CPU0 Temp" in res.output


# ---- ConsoleRunner: pipeline access over the selected console ----

class _FakeConsole:
    def __init__(self):
        self.probes = []

    def run_probes(self, commands):
        self.probes.extend(commands)
        return ConsoleResult(output="LnkSta: Speed 8GT/s, Width x16\n", probe_count=1)


def test_console_runner_executes_read_only_probes():
    from harness.engine.sol import ConsoleRunner
    console = _FakeConsole()
    runner = ConsoleRunner(console)
    res = runner.execute(["/usr/bin/lspci", "-xxx"])

    assert res.ok and "LnkSta" in res.stdout
    assert console.probes == ["/usr/bin/lspci -xxx"]  # argv joined into one probe
    assert len(runner.calls) == 1
    assert res.argv == ["/usr/bin/lspci", "-xxx"]


def test_console_runner_denies_write_probes_without_reaching_wire():
    from harness.engine.sol import ConsoleRunner
    console = _FakeConsole()
    runner = ConsoleRunner(console)

    res = runner.execute(["/usr/sbin/i2cset", "-y", "0", "0x51", "0x40", "0xff"])
    assert not res.ok and "denied" in res.stderr
    assert console.probes == []  # never reached the console
    assert len(runner.calls) == 1


def test_console_runner_records_console_failures_as_failed_results():
    from harness.engine.sol import ConsoleRunner, SerialConsoleError

    class _FailingConsole:
        def run_probes(self, commands):
            raise SerialConsoleError("console script exited 1")

    res = ConsoleRunner(_FailingConsole()).execute(["/usr/bin/lspci"])
    assert not res.ok and "console script exited" in res.stderr
    assert res.exit_code == 1


def test_console_runner_marks_missing_probe_tool_as_failed():
    """expect exits 0 while the shell prints `command not found`: not a fake-ok."""
    from harness.engine.sol import ConsoleRunner

    class _MissingToolConsole:
        def run_probes(self, commands):
            return ConsoleResult(
                output="RScmCli# start serial session -i 2 -p 2200\r\n"
                       "admin@m1120-c4a15:~$ sudo -S i2cdump -y 8 0xb\r\n"
                       "Password: \r\n"
                       "sudo: i2cdump: command not found\r\n"
                       "admin@m1120-c4a15:~$ ",
                probe_count=1)

    res = ConsoleRunner(_MissingToolConsole()).execute(["sudo -S i2cdump -y 8 0xb"])
    assert res.exit_code == 127
    assert not res.ok
    assert "command not found" in res.stderr


def test_console_runner_stray_not_found_noise_does_not_fake_127():
    """A probe that RAN keeps its output even when unrelated session noise
    (e.g. a stray line) contains `command not found` -- only a block whose
    output is just the shell error is a hard 127."""
    from harness.engine.sol import ConsoleRunner, _not_found_error

    assert _not_found_error("-sh: 0penBmc: command not found\n",
                            "sudo -S ipmitool sensor list") is None
    assert _not_found_error("-sh: ipmitool: command not found\n",
                            "sudo -S ipmitool sensor list") is not None
    assert _not_found_error("sudo: i2cdump: command not found\n",
                            "sudo -S i2cdump -y 8 0xb") is not None
    assert _not_found_error(
        "Password: \nCPU0 Temp | 45.000 | degrees C | ok\n"
        "admin@m1120-c4a15:~$ 0penBmc\n-sh: 0penBmc: command not found\n",
        "sudo -S ipmitool sensor list") is None

    class _NoisyConsole:
        def run_probes(self, commands):
            return ConsoleResult(
                output="admin@m1120-c4a15:~$ sudo -S ipmitool sensor list\n"
                       "CPU0 Temp | 45.000 | degrees C | ok\n"
                       "admin@m1120-c4a15:~$ 0penBmc\n"
                       "-sh: 0penBmc: command not found\n"
                       "admin@m1120-c4a15:~$ ",
                probe_count=1)

    res = ConsoleRunner(_NoisyConsole()).execute(["sudo -S ipmitool sensor list"])
    assert res.exit_code == 0
    assert res.ok
    assert "CPU0 Temp" in res.stdout


def test_console_runner_uses_absolute_path_when_bmc_path_lacks_sbin():
    """OpenBMC shells may lack /usr/sbin on PATH even though i2c-tools live
    there: bare i2cdump/i2ctransfer must go on the wire as /usr/sbin/<tool>."""
    from harness.engine.sol import ConsoleRunner

    class _RecordingConsole:
        def __init__(self):
            self.probes = []

        def run_probes(self, commands):
            self.probes.extend(commands)
            return ConsoleResult(output="1b: 05\n", probe_count=1)

    console = _RecordingConsole()
    runner = ConsoleRunner(console)
    res = runner.execute(["sudo -S i2cdump", "-y", "8", "0xb"])
    assert res.ok
    assert console.probes == ["sudo -S /usr/sbin/i2cdump -y 8 0xb"]
    # recorded argv keeps the planned form; source strings stay stable
    assert res.argv == ["sudo -S i2cdump", "-y", "8", "0xb"]
    # bare i2ctransfer and i2cget are absolutized too
    runner.execute(["i2ctransfer", "-y", "8", "w1@0xb", "0x00", "r256"])
    runner.execute(["sudo", "-S", "i2cget", "-y", "8", "0xb", "0x1b"])
    assert console.probes[1:] == [
        "/usr/sbin/i2ctransfer -y 8 w1@0xb 0x00 r256",
        "sudo -S /usr/sbin/i2cget -y 8 0xb 0x1b",
    ]
    # non-i2c commands are untouched
    runner.execute(["/usr/bin/lspci", "-xxx"])
    assert console.probes[-1] == "/usr/bin/lspci -xxx"


def test_console_runner_runs_detect_model_and_collectors():
    from harness.engine.sol import ConsoleRunner
    console = _FakeConsole()
    runner = ConsoleRunner(console)
    # model detection + a pcie collector both ride the console path
    assert runner.execute(["/bin/dmidecode"]).ok
    assert runner.execute(["/usr/bin/lspci", "-xxx"]).ok
    assert console.probes == ["/bin/dmidecode", "/usr/bin/lspci -xxx"]


def test_console_runner_is_flagged_as_console():
    from harness.engine.sol import ConsoleRunner
    assert ConsoleRunner(_FakeConsole()).is_console is True


_FAKE_I2C_ROWS = (
    "1b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
    "a1: 05 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
)


def _fake_stdout(cmd: str) -> str:
    return _FAKE_I2C_ROWS if "i2cdump" in cmd else "ok"


def test_bmc_console_collector_issues_bmc_shell_probes():
    from harness.engine.runner import CommandResult
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector

    class _FakeRunner:
        is_console = True

        def __init__(self):
            self.calls = []

        def execute(self, argv, timeout=30.0):
            cmd = " ".join(argv)
            result = CommandResult(argv=list(argv), stdout=_fake_stdout(cmd),
                                   stderr="", exit_code=0, elapsed_ms=1)
            self.calls.append(result)
            return result

    for subsystem, expected in {
        "cpu": ["sudo -S ipmitool sensor list", "sudo -S i2cdump -y 8 0xb"],
        "kernel": ["sudo -S ipmitool sel list", "dmesg -r"],
        "ipmi": ["sudo -S ipmitool sensor list", "sudo -S ipmitool sel list",
                 "sudo -S ipmitool fru print"],
    }.items():
        fake = _FakeRunner()
        dumps = BmcConsoleCollector(fake, subsystem=subsystem).collect()
        executed = [" ".join(c.argv) for c in fake.calls]
        assert [d.source for d in dumps] == expected
        assert executed == expected
        assert all(d.ok for d in dumps)


def test_bmc_console_collectors_dedupe_probes_across_subsystems():
    from harness.engine.runner import CommandResult
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector

    class _FakeRunner:
        is_console = True

        def __init__(self):
            self.calls = []

        def execute(self, argv, timeout=30.0):
            cmd = " ".join(argv)
            result = CommandResult(argv=list(argv), stdout=_fake_stdout(cmd),
                                   stderr="", exit_code=0, elapsed_ms=1)
            self.calls.append(result)
            return result

    fake = _FakeRunner()
    BmcConsoleCollector(fake, subsystem="cpu").collect()    # sensor list + i2cdump
    BmcConsoleCollector(fake, subsystem="ipmi").collect()   # sel list + fru print
    BmcConsoleCollector(fake, subsystem="kernel").collect()  # dmesg -r
    executed = [" ".join(c.argv) for c in fake.calls]
    assert executed == [
        "sudo -S ipmitool sensor list",
        "sudo -S i2cdump -y 8 0xb",
        "sudo -S ipmitool sel list",
        "sudo -S ipmitool fru print",
        "dmesg -r",
    ]


def test_bmc_console_collector_materializes_prebatched_probes():
    """The plan-level pre-batch runs the whole plan in ONE console session;
    collect() must turn those recorded results into RegisterDumps instead of
    dropping them (otherwise evidence is empty on console runs)."""
    from harness.engine.runner import CommandResult
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector

    class _PrewarmedRunner:
        is_console = True

        def __init__(self, results):
            self.calls = list(results)
            self.executed = []

        def execute(self, argv, timeout=30.0):
            self.executed.append(argv)
            raise AssertionError(f"collector re-ran a pre-batched probe: {argv!r}")

        def batch_execute(self, cmds, timeout=300.0):
            raise AssertionError("collector opened a new batch after the pre-batch")

    sensor = CommandResult(argv=["sudo", "-S", "ipmitool", "sensor", "list"],
                           stdout="ok", stderr="", exit_code=0, elapsed_ms=1)
    i2cdump = CommandResult(argv=["sudo", "-S", "i2cdump", "-y", "8", "0xb"],
                            stdout=_FAKE_I2C_ROWS, stderr="", exit_code=0,
                            elapsed_ms=1)
    runner = _PrewarmedRunner([sensor, i2cdump])
    dumps = BmcConsoleCollector(runner, subsystem="cpu").collect()
    assert runner.executed == []
    assert [d.source for d in dumps] == ["sudo -S ipmitool sensor list",
                                         "sudo -S i2cdump -y 8 0xb"]
    assert all(d.ok for d in dumps)


def test_bmc_console_collector_reruns_probe_that_previously_failed():
    from harness.engine.runner import CommandResult
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector

    class _FakeRunner:
        is_console = True

        def __init__(self):
            self.calls = []

        def execute(self, argv, timeout=30.0):
            cmd = " ".join(argv)
            failed = len(self.calls) == 0  # only the very first probe fails
            result = CommandResult(argv=list(argv), stdout=_fake_stdout(cmd),
                                   stderr="", exit_code=1 if failed else 0, elapsed_ms=1)
            self.calls.append(result)
            return result

    fake = _FakeRunner()
    BmcConsoleCollector(fake, subsystem="cpu").collect()     # first attempt fails
    BmcConsoleCollector(fake, subsystem="ipmi").collect()    # retries sensor list
    executed = [" ".join(c.argv) for c in fake.calls]
    assert executed == ["sudo -S ipmitool sensor list",
                        "sudo -S i2cdump -y 8 0xb",
                        "sudo -S ipmitool sensor list",
                        "sudo -S ipmitool sel list",
                        "sudo -S ipmitool fru print"]


def test_bmc_console_collector_cpld_chain_falls_back_when_i2cdump_missing():
    """When i2cdump is absent, the chain must fall through to i2ctransfer."""
    from harness.engine.runner import CommandResult
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector

    class _FakeRunner:
        is_console = True

        def __init__(self):
            self.calls = []

        def execute(self, argv, timeout=30.0):
            cmd = " ".join(argv)
            if "i2cdump" in cmd:  # tool missing: shell prints not-found, exit 127
                stdout, code = "sudo: i2cdump: command not found", 127
            elif "i2ctransfer" in cmd:
                stdout = "1b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
                code = 0
            else:
                stdout, code = "ok", 0
            result = CommandResult(argv=list(argv), stdout=stdout, stderr="",
                                   exit_code=code, elapsed_ms=1)
            self.calls.append(result)
            return result

    fake = _FakeRunner()
    dumps = BmcConsoleCollector(fake, subsystem="cpu").collect()
    executed = [" ".join(c.argv) for c in fake.calls]
    # sensor list, then the chain: sudo i2cdump (missing) -> plain i2cdump
    # (missing) -> sudo i2ctransfer (data). Plain i2ctransfer never needed.
    assert executed == [
        "sudo -S ipmitool sensor list",
        "sudo -S i2cdump -y 8 0xb",
        "i2cdump -y 8 0xb",
        "sudo -S i2ctransfer -y 8 w1@0xb 0x00 r256",
    ]
    ok = [d for d in dumps if d.ok]
    assert [d.source for d in ok] == ["sudo -S ipmitool sensor list",
                                      "sudo -S i2ctransfer -y 8 w1@0xb 0x00 r256"]
    # The failed attempts are recorded (audit trail), not dropped.
    assert any("command not found" in d.raw for d in dumps)


def test_detect_model_uses_fru_on_console_runner():
    from harness.engine.runner import CommandResult
    from harness.inspect.model import detect_model

    # ipmitool fru print pads the label before the colon
    fru_out = (
        "FRU Device Description : Builtin FRU Device (ID 0)\n"
        " Board Mfg             : Microsoft\n"
        " Board Product         : C4A15\n"
        " Product Manufacturer  : Microsoft\n"
        " Product Name          : C4A15\n"
        " Product Part Number   : M1382332-001\n"
    )

    class _FruRunner:
        is_console = True

        def execute(self, argv, timeout=30.0):
            assert argv == ["sudo -S ipmitool fru print"]
            return CommandResult(argv=argv, stdout=fru_out, stderr="",
                                 exit_code=0, elapsed_ms=1)

    model = detect_model(_FruRunner())
    assert model is not None
    assert model.product_name == "C4A15"
    assert model.bios_vendor == "Microsoft"


def test_detect_model_fru_falls_back_to_board_product_and_skips_na():
    from harness.engine.runner import CommandResult
    from harness.inspect.model import detect_model

    fru_out = (
        "FRU Device Description : PDB (ID 4)\n"
        " Chassis Part Number   : N/A\n"
        " Board Mfg             : 091\n"
        " Board Product         : C4A15\n"
        " Product Name          : N/A\n"
        " Product Manufacturer  : N/A\n"
    )

    class _FruRunner:
        is_console = True

        def execute(self, argv, timeout=30.0):
            return CommandResult(argv=argv, stdout=fru_out, stderr="",
                                 exit_code=0, elapsed_ms=1)

    model = detect_model(_FruRunner())
    assert model is not None
    assert model.product_name == "C4A15"      # Product Name N/A -> Board Product
    assert model.bios_vendor == "091"         # Product Mfr N/A -> Board Mfg


def test_detect_model_fru_returns_none_when_no_product():
    from harness.engine.runner import CommandResult
    from harness.inspect.model import detect_model

    class _FruRunner:
        is_console = True

        def execute(self, argv, timeout=30.0):
            return CommandResult(argv=argv, stdout="FRU Device Description : PDB (ID 4)\n"
                                                  " Product Name          : N/A\n",
                                 stderr="", exit_code=0, elapsed_ms=1)

    assert detect_model(_FruRunner()) is None


def test_detect_model_keeps_dmidecode_on_host_runner():
    from harness.engine.runner import CommandResult, Runner
    from harness.inspect.model import detect_model

    class _HostRunner(Runner):
        is_console = False

        def __init__(self):
            super().__init__(None)  # policy unused in this fake

        def execute(self, argv, timeout=30.0):
            assert argv == ["/bin/dmidecode"]
            return CommandResult(argv=argv, stdout="Product Name: X\n", stderr="",
                                 exit_code=0, elapsed_ms=1)

    model = detect_model(_HostRunner())
    assert model is not None and model.product_name == "X"


# ---- batched console probes (one session per collector, not per probe) ----

def test_split_batch_output_isolates_each_probe_block():
    from harness.engine.sol import _split_batch_output

    transcript = (
        "root@bmc:~$ sudo -S ipmitool sensor list\r\n"
        "CPU0 Temp | 45.000 | degrees C | ok\r\n"
        "root@bmc:~$ sudo -S ipmitool sel list\r\n"
        " 5 | Memory | Uncorrectable ECC\r\n"
        "root@bmc:~$"
    )
    blocks = _split_batch_output(transcript, [
        "sudo -S ipmitool sensor list",
        "sudo -S ipmitool sel list",
    ])
    assert "45.000" in blocks[0] and "sel list" not in blocks[0]
    assert "Uncorrectable ECC" in blocks[1]


def test_split_batch_output_handles_missing_echo():
    from harness.engine.sol import _split_batch_output

    blocks = _split_batch_output("only one echo line\r\n", [
        "sudo -S i2cdump -y 8 0xb",
        "sudo -S ipmitool sel list",
    ])
    assert blocks[0] == ""          # first echo not found
    assert blocks[1] == ""          # second echo not found either


def test_split_batch_output_ignores_expect_debug_lines():
    """expect's `send:` debug lines contain the command text but never END with
    it; only the real echoed line (prompt + command) may delimit blocks."""
    from harness.engine.sol import _split_batch_output

    transcript = (
        "spawn q63-1 rm\n"
        "send: sending \"sudo -S ipmitool sensor list\\r\" to { exp0 }\n"
        "root@bmc:~$ sudo -S ipmitool sensor list\n"
        "CPU0 Temp | 45.000 | degrees C | ok\n"
        "send: sending \"sudo -S ipmitool sel list\\r\" to { exp0 }\n"
        "root@bmc:~$ sudo -S ipmitool sel list\n"
        " 5 | Memory | Uncorrectable ECC\n"
        "root@bmc:~$"
    )
    blocks = _split_batch_output(transcript, [
        "sudo -S ipmitool sensor list",
        "sudo -S ipmitool sel list",
    ])
    assert "45.000" in blocks[0]
    assert "root@bmc:~$ sudo -S ipmitool sel list" not in blocks[0]
    assert "Uncorrectable ECC" in blocks[1]
    assert "send:" not in blocks[1]


def test_bmc_console_collector_batches_non_i2c_probes():
    from harness.engine.runner import CommandResult
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector

    class _BatchRunner:
        is_console = True

        def __init__(self):
            self.calls = []
            self.batches: list[list[str]] = []

        def batch_execute(self, cmds, timeout=300.0):
            self.batches.append(list(cmds))
            results = []
            for cmd in cmds:
                stdout = _FAKE_I2C_ROWS if "i2cdump" in cmd else "ok"
                argv = cmd.split()
                result = CommandResult(argv=argv, stdout=stdout,
                                       stderr="", exit_code=0, elapsed_ms=1)
                self.calls.append(result)
                results.append(result)
            return results

        def execute(self, argv, timeout=300.0):
            cmd = " ".join(argv)
            result = CommandResult(argv=list(argv), stdout=_fake_stdout(cmd),
                                   stderr="", exit_code=0, elapsed_ms=1)
            self.calls.append(result)
            return result

    fake = _BatchRunner()
    dumps = BmcConsoleCollector(fake, subsystem="cpu").collect()
    # one batched session for sensor list; the CPLD chain still runs
    # sequentially and stops at the first real register dump.
    assert fake.batches == [["sudo -S ipmitool sensor list"]]
    executed = [" ".join(c.argv) for c in fake.calls]
    assert executed == ["sudo -S ipmitool sensor list",
                        "sudo -S i2cdump -y 8 0xb"]
    assert [d.source for d in dumps] == executed


def test_ssh_session_open_tolerates_missing_known_hosts(tmp_path, monkeypatch):
    """SSHSession.open must not crash when the known_hosts path's parent dir is
    missing; the reject policy still fails closed on unknown hosts."""
    import harness.engine.session as sess_mod
    from harness.config.models import BMCDomain, Host, SSHDomain
    from harness.config.vault import MemorySecretStore
    from harness.engine.allowlist import default_policy

    calls = {"load": 0, "connect": 0}

    class _Client:
        def set_missing_host_key_policy(self, policy):
            self.policy = policy

        def load_host_keys(self, path):
            calls["load"] += 1
            raise FileNotFoundError(f"no such directory: {path}")

        def connect(self, **kw):
            calls["connect"] += 1

    monkeypatch.setattr(sess_mod.paramiko, "SSHClient", _Client)
    monkeypatch.setattr(sess_mod, "load_key_material",
                        lambda store, vault, tmp: tmp_path / "id.pem")
    host = Host(
        name="h1", address="10.0.0.10", model="model_x",
        ssh=SSHDomain(user="diagbot",
                      identity_vault_path="secret/harness/diagbot/id_ed25519",
                      known_hosts_path=str(tmp_path / "no" / "such" / "kh")),
        bmc=BMCDomain(address="10.0.0.11", username="bmc-ro",
                      password_vault_path="secret/harness/bmc/bmc-ro"),
        collector_profile="cpu_msr",
    )
    sess = sess_mod.SSHSession(host, default_policy(), MemorySecretStore())
    sess.open()
    assert calls["load"] == 1
    assert calls["connect"] == 1