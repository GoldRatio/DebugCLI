"""Multi-service console routing (engine.services.MultiConsoleRunner)."""

from __future__ import annotations

from harness.engine.runner import CommandResult
from harness.engine.services import MultiConsoleRunner


class FakeServiceRunner:
    """Console-runner stand-in: records calls, stamps its service label."""

    is_console = True
    force_read_only = True

    def __init__(self, name: str):
        self.label = name
        self.calls: list[CommandResult] = []
        self.on_probe = None
        self.probe_cache: dict[str, CommandResult] = {}

    def execute(self, argv, timeout=300.0):
        res = CommandResult(argv=list(argv), stdout=f"{self.label}:{' '.join(argv)}",
                            stderr="", exit_code=0, elapsed_ms=1)
        res.service = self.label
        self.calls.append(res)
        if self.on_probe is not None:
            self.on_probe(res)
        return res

    def batch_execute(self, cmds, timeout=300.0):
        return [self.execute(cmd.split()) for cmd in cmds]


def _multi():
    bmc, host = FakeServiceRunner("bmc"), FakeServiceRunner("host")
    runner = MultiConsoleRunner(
        {"bmc": bmc, "host": host},
        {"bmc": ("ipmitool", "i2cdump", "ls", "dmesg"),
         "host": ("lspci", "dmidecode", "smartctl")})
    return runner, bmc, host


def test_route_by_program_table():
    runner, _, _ = _multi()
    assert runner.route(["sudo", "-S", "ipmitool", "sensor", "list"]) == "bmc"
    assert runner.route(["sudo", "-S", "i2cdump", "-y", "8", "0xb"]) == "bmc"
    assert runner.route(["/bin/dmesg", "-r"]) == "bmc"
    assert runner.route(["/usr/bin/lspci", "-xxx"]) == "host"
    assert runner.route(["/bin/smartctl", "-a", "/dev/sda"]) == "host"
    assert runner.route(["dmidecode"]) == "host"


def test_route_fallback_first_service():
    runner, _, _ = _multi()
    assert runner.route(["whatever"]) == "bmc"


def test_execute_delegates_and_labels_service():
    runner, _, _ = _multi()
    res = runner.execute(["/usr/bin/lspci", "-xxx"])
    assert res.service == "host" and res.stdout.startswith("host:")


def test_batch_execute_preserves_original_order():
    runner, bmc, host = _multi()
    cmds = ["sudo -S ipmitool sensor list", "/usr/bin/lspci -xxx",
            "sudo -S i2cdump -y 8 0xb", "dmidecode"]
    results = runner.batch_execute(cmds)
    assert [" ".join(r.argv) for r in results] == cmds
    # routed to their own service runners
    assert [" ".join(c.argv) for c in bmc.calls] == [
        "sudo -S ipmitool sensor list", "sudo -S i2cdump -y 8 0xb"]
    assert [" ".join(c.argv) for c in host.calls] == [
        "/usr/bin/lspci -xxx", "dmidecode"]


def test_on_probe_propagates_to_services():
    runner, bmc, host = _multi()
    seen: list[str] = []
    runner.on_probe = lambda r: seen.append(r.service)
    runner.execute(["sudo", "-S", "ipmitool", "fru", "print"])
    runner.execute(["/usr/bin/lspci", "-xxx"])
    assert seen == ["bmc", "host"]


def test_calls_is_ordered_union_across_services():
    runner, _, _ = _multi()
    runner.execute(["sudo", "-S", "ipmitool", "fru", "print"])
    runner.execute(["/usr/bin/lspci", "-xxx"])
    assert [" ".join(c.argv) for c in runner.calls] == [
        "sudo -S ipmitool fru print", "/usr/bin/lspci -xxx"]
