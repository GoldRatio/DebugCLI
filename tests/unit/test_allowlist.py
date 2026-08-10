"""Allowlist enforcement tests: the read-only funnel is the security backbone."""

import pytest

from harness.engine.allowlist import AllowRule, default_policy
from harness.engine.security_gate import check, ReadOnlyViolation
from harness.engine.runner import Runner
from harness.engine.sudoers_gen import build, render_sudoers


def test_allowrule_exact_flags():
    rule = AllowRule("/bin/smartctl", ("-a", "*"))
    assert rule.matches(["/bin/smartctl", "-a", "/dev/sda"])
    # wrong flag variant not allowed
    assert not rule.matches(["/bin/smartctl", "-s", "/dev/sda"])
    # program path must be exact
    assert not rule.matches(["/bin/bash", "-a", "/dev/sda"])
    # arg-count mismatch (device required by the "*" position)
    assert not rule.matches(["/bin/smartctl", "-a"])


def test_policy_default_allows_read_probe():
    policy = default_policy()
    assert policy.allows(["/bin/dmidecode"])
    assert policy.allows(["/usr/sbin/ipmitool", "sensor"])
    assert policy.allows(["/usr/sbin/ipmitool", "sel", "list"])
    # a write variant is not in the allowlist
    assert not policy.allows(["/usr/sbin/ipmitool", "sensor", "-w"])


def test_security_gate_blocks_destructive():
    for destructive in (["dd", "if=/dev/sda", "of=/tmp/x"], ["/sbin/shutdown"], ["tee"]):
        with pytest.raises(ReadOnlyViolation):
            check(destructive)


def test_security_gate_blocks_write_flags():
    with pytest.raises(ReadOnlyViolation):
        check(["/something/setpci"])
    with pytest.raises(ReadOnlyViolation):
        check(["/bin/lspci", "0:1.0", "-w"])


def test_runner_enforces_read_only_even_for_deny_argv():
    runner = Runner(default_policy(), force_read_only=True)
    with pytest.raises(ReadOnlyViolation):
        runner.execute(["/bin/sh", "-c", "echo pwn"])
    # banned flag token is rejected regardless of allowlist
    with pytest.raises(ReadOnlyViolation):
        runner.execute(["/usr/bin/rdmsr", "-a", "-w"])


def test_runner_captures_result(monkeypatch):
    runner = Runner(default_policy(), force_read_only=True)

    class FakeResult:
        stdout = "0x8000000000000001"
        stderr = ""
        returncode = 0

    # Avoid depending on a real Linux binary existing on this dev box.
    import harness.engine.runner as mod
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: FakeResult())
    res = runner.execute(["/usr/bin/rdmsr", "-a"], timeout=5)
    assert res.exit_code == 0
    assert res.elapsed_ms >= 0
    assert "0x8000000000000001" in res.stdout


def test_sudoers_render_single_line_per_rule():
    policy = default_policy()
    text = render_sudoers("diagbot", policy)
    assert "NOPASSWD: /bin/dmidecode" in text
    assert "diagbot" in text
    assert "--write" not in text


def test_sudoers_build_sha():
    artifact = build("ignore", "diagbot")
    assert artifact.sha256