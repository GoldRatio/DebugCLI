"""Summarizer: evidence-kind tagging, SEL timestamps, current-state health notes."""

from harness.diagnosis.summarize import summarize
from harness.inspect.base import RegisterDump

SEL_TEXT = (
    "   1 | 07/17/26 | 06:05:05 UTC | Event Logging Disabled #0x09 | "
    "Log area reset/cleared | Asserted\n"
    "  12 | 07/17/26 | 15:50:36 UTC | Power Supply #0x75 | Failure detected | Asserted\n"
    "  13 | 07/17/26 | 15:50:36 UTC | Power Supply #0x7d | Failure detected | Asserted\n"
    "  14 | 07/17/26 | 15:50:36 UTC | Power Supply #0x7f | Failure detected | "
    "Deasserted\n"
    "  15 | 08/10/26 | 17:12:16 UTC | Power Supply #0x75 | Failure detected | Asserted\n"
    "  1a | 08/10/26 | 17:12:17 UTC | Button #0xf1 | Power Button pressed | Asserted\n")

SENSOR_TEXT = (
    "P12V_SCM_VOLT    | 11.985     | Volts      | ok    | na        | na\n"
    "Power_Status     | 0x0        | discrete   | 0x0180| na        | na\n"
    "CPU0_TEMP        | 95.000     | degrees C  | ucr   | na        | na\n")

KERNEL_TEXT = "MCE: memory error on DIMM_A2\nnothing else\n"


def _dump(raw, kind, source=None):
    return RegisterDump(
        subsystem="bmc",
        source=source or f"ipmitool {kind} list",
        raw=raw,
        cmd_argv=["x"],
        ok=True,
        meta={"exit": 0, "elapsed_ms": 1, "kind": kind},
    )


def test_sel_lines_tagged_historical_with_timestamp():
    s = summarize([_dump(SEL_TEXT, "sel")])
    tagged = [l for l in s.interesting if l.startswith("[sel-historical 07/17/26 15:50:36 UTC]")]
    assert len(tagged) == 3
    assert any("Failure detected" in l for l in tagged)
    recent = [l for l in s.interesting if l.startswith("[sel-historical 08/10/26 17:12:16 UTC]")]
    assert any("Power Supply" in l for l in recent)


def test_sel_notes_explain_historical_nature_and_unpaired_asserts():
    s = summarize([_dump(SEL_TEXT, "sel")])
    notes = "\n".join(s.notes)
    assert "HISTORICAL event log" in notes
    assert "NOT proof of a current fault" in notes
    assert "asserted without a matching deassert" in notes


def test_current_state_sensor_health_and_anomaly_tagging():
    s = summarize([_dump(SENSOR_TEXT, "sensor")])
    assert s.current_health == "anomaly"
    assert any("CPU0_TEMP" in l and l.startswith("[current]") for l in s.interesting)
    notes = "\n".join(s.notes)
    assert "Current live sensors: anomaly" in notes
    assert "CPU0_TEMP=ucr" in notes


def test_discrete_state_codes_not_flagged_as_anomalous():
    s = summarize([_dump(SENSOR_TEXT.replace("CPU0_TEMP", "TMP_OK"), "sensor")])
    s2 = summarize([_dump("Power_Status | 0x0 | discrete | 0x0180 | na | na\n", "sensor")])
    assert s2.current_health == "ok"
    assert s.current_health == "anomaly"


def test_all_ok_sensors_report_healthy():
    s = summarize([_dump("FAN_0A | 10360 | RPM | ok | na | na\n", "sensor")])
    assert s.current_health == "ok"
    assert "all ok/na" in "\n".join(s.notes)


def test_kind_fallback_by_source_string():
    s = summarize([
        RegisterDump(subsystem="bmc", source="sudo -S ipmitool sel list",
                     raw=SEL_TEXT, cmd_argv=["x"], ok=True, meta={}),
        RegisterDump(subsystem="bmc", source="sudo -S ipmitool sensor list",
                     raw="FAN_0A | 10360 | RPM | ok | na | na\n", cmd_argv=["x"],
                     ok=True, meta={}),
    ])
    assert s.current_health == "ok"
    assert any("sel-historical" in l for l in s.interesting)


def test_console_banner_noise_does_not_break_sensor_parsing():
    noisy = (
        "\x1b[1;33m*****************************\x1b[0m\n"
        "RScmCli# start serial session -i 3 -p 2200\n"
        "admin@m1120:~$ sudo -S ipmitool sensor list\n"
        "Password: \n"
        "FAN_0A | 10360.000 | RPM | ok | na | na\n"
        "sensor  | reading  | units  | status\n")
    s = summarize([_dump(noisy, "sensor")])
    assert s.current_health == "ok"


def test_summary_respects_max_items():
    sel = "".join(
        f"{i:2x} | 08/10/26 | 17:12:16 UTC | Power Supply #0x75 | Failure detected | Asserted\n"
        for i in range(1, 60))
    s = summarize([_dump(sel, "sel")], max_items=10)
    assert s.anomaly_count == 10
    assert len(s.interesting) == 10
