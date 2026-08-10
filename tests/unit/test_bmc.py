"""BMC channel tests: allowlist templates, read-only gate, and password secrecy."""

import pytest

from harness.engine.bmc import BmcRunner, bmc_policy
from harness.engine.runner import CommandResult
from harness.engine.security_gate import ReadOnlyViolation

VALID = ["/usr/bin/ipmitool", "-I", "lanplus", "-H", "10.0.0.11", "-U", "bmc-ro",
         "-E", "sensor"]


def test_bmc_policy_allows_lan_read_forms():
    policy = bmc_policy()
    assert policy.allows(VALID)
    assert policy.allows(VALID[:-1] + ["sel", "list"])
    assert policy.allows(VALID[:-1] + ["fru", "print"])


def test_bmc_policy_denies_other_shapes():
    policy = bmc_policy()
    # no -H/-U/-E pins: OS-domain form must not run on the BMC channel
    assert not policy.allows(["/usr/bin/ipmitool", "sensor"])
    assert not policy.allows(["/usr/bin/ipmitool", "sdr", "list"])
    assert not policy.allows(VALID[:-1] + ["raw", "0x30"])
    # write-ish invocation: `ipmitool ... raw 0x30 0x41` style is out of shape
    assert not policy.allows(["/usr/bin/ipmitool", "-I", "lanplus", "-H", "h",
                              "-U", "u", "-E", "raw"])


def test_bmc_runner_enforces_gate_and_policy():
    runner = BmcRunner("10.0.0.11", "bmc-ro", "s3cret")
    with pytest.raises(ReadOnlyViolation):
        runner.execute(["/usr/bin/ipmitool", "sensor"])  # not in BMC allowlist
    with pytest.raises(ReadOnlyViolation):
        runner.execute(["/bin/dd", "if=/dev/sda"])  # hard gate


def test_bmc_runner_passes_env_password_not_argv():
    captured = {}

    class Spy(BmcRunner):
        def _exec(self, argv, timeout=30.0):
            captured["argv"] = list(argv)
            captured["env"] = dict(self._password and {})  # placeholder
            return CommandResult(argv=argv, stdout="ok", stderr="", exit_code=0, elapsed_ms=1)

    spy = Spy("10.0.0.11", "bmc-ro", "s3cret")
    result = spy.execute(VALID)
    assert result.ok
    assert "s3cret" not in captured["argv"]
    assert "s3cret" in spy._password  # password lives on the runner, not the trace
    assert spy.calls  # recorded for audit


def test_bmc_runner_exec_uses_ipmi_password_env(monkeypatch):
    import subprocess


    env_seen = {}

    def fake_run(argv, **kwargs):
        env_seen.update(kwargs.get("env", {}))
        return type("P", (), {"stdout": "x", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = BmcRunner("10.0.0.11", "bmc-ro", "s3cret")
    runner.execute(VALID)
    assert env_seen.get("IPMI_PASSWORD") == "s3cret"
