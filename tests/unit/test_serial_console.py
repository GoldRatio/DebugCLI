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
        "sudo -S i2cdump -y -f 0 0x51",      # sudo -S wraps a read-only probe
        "sudo -S ipmitool sensor list",
        "sudo -S ipmitool sel list",
        "sudo -S ipmitool fru print",
        "ipmitool sensor elist",
    ):
        assert validate_serial_probe(good) == good


def test_probe_rejects_i2c_write_and_destructive_sudo():
    for bad in (
        "i2cset -y 0 0x51 0x40 0xff",          # i2cset is the I2C WRITE tool
        "i2cset",                              # never expressible at all
        "ipmitool sel clear",                  # destructive ipmitool subcommand
        "ipmitool sel delete 5",
        "ipmitool sensor thresh set 0 80",     # write form
        "ipmitool raw 0x30 0x40",              # raw write not allowed
        "sudo -S i2cset -y 0 0x51 0x40 0xff",  # sudo cannot smuggle writes in
        "sudo -S shutdown -h now",
        "sudo -S",                             # sudo must wrap a probe program
        "sudo i2cdump -y -f 0 0x51",           # only -S form (stdin password)
        "sudo -S nvme format /dev/nvme0n1",
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
    # handshake only after sudo probes; plain probes get no password prompt
    assert script.count('expect "password for"') == 1
    assert 'send "s3cret\\r"' in script
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
    assert 'expect "password for"' in client.written
    assert 'send "0penBmc\\r"' in client.written  # trailing newline stripped


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
    assert 'expect "password for"' not in client.written


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


def test_bmc_console_collector_issues_bmc_shell_probes():
    from harness.engine.runner import CommandResult
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector

    class _FakeRunner:
        is_console = True

        def __init__(self):
            self.calls = []

        def execute(self, argv, timeout=30.0):
            result = CommandResult(argv=list(argv), stdout="ok", stderr="",
                                   exit_code=0, elapsed_ms=1)
            self.calls.append(result)
            return result

    for subsystem, expected in {
        "cpu": ["sudo -S ipmitool sensor list"],
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
            result = CommandResult(argv=list(argv), stdout="ok", stderr="",
                                   exit_code=0, elapsed_ms=1)
            self.calls.append(result)
            return result

    fake = _FakeRunner()
    BmcConsoleCollector(fake, subsystem="cpu").collect()    # sensor list
    BmcConsoleCollector(fake, subsystem="ipmi").collect()   # sel list + fru print
    BmcConsoleCollector(fake, subsystem="kernel").collect()  # dmesg -r
    executed = [" ".join(c.argv) for c in fake.calls]
    assert executed == [
        "sudo -S ipmitool sensor list",
        "sudo -S ipmitool sel list",
        "sudo -S ipmitool fru print",
        "dmesg -r",
    ]


def test_bmc_console_collector_reruns_probe_that_previously_failed():
    from harness.engine.runner import CommandResult
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector

    class _FakeRunner:
        is_console = True

        def __init__(self):
            self.calls = []

        def execute(self, argv, timeout=30.0):
            failed = not any(c.ok for c in self.calls)
            result = CommandResult(argv=list(argv), stdout="ok", stderr="",
                                   exit_code=1 if failed else 0, elapsed_ms=1)
            self.calls.append(result)
            return result

    fake = _FakeRunner()
    BmcConsoleCollector(fake, subsystem="cpu").collect()     # first attempt fails
    BmcConsoleCollector(fake, subsystem="ipmi").collect()    # retries sensor list
    executed = [" ".join(c.argv) for c in fake.calls]
    assert executed == ["sudo -S ipmitool sensor list",
                        "sudo -S ipmitool sensor list",
                        "sudo -S ipmitool sel list",
                        "sudo -S ipmitool fru print"]


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