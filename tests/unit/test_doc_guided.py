"""Doc-guided planning: mine read-only probe commands from doc snippets."""

from harness.plan.doc_guided import mine_probe_commands
from harness.plan.profile import plan_collection

GB_HANGUP_P1 = (
    "[GB_HangUp_troubleshooting_v1.2.pdf p.1] For EVERY Amber Light issue, "
    'please dump "i2cdump -y 8 0xb" in the BMC to get the boot state and '
    "related register value."
)


def test_mines_i2cdump_from_gb_hangup_snippet():
    assert mine_probe_commands([GB_HANGUP_P1]) == ["i2cdump -y 8 0xb"]


def test_mines_ipmitool_read_forms():
    snippets = [
        "Run `ipmitool fru print` to check for missing FRUs.",
        "Check with ipmitool sensor list for disabled sensors.",
        "ipmitool sdr elist shows all sensor readings.",
        "Use ipmitool sel info for the last event id.",
    ]
    assert mine_probe_commands(snippets) == [
        "ipmitool fru print",
        "ipmitool sensor list",
        "ipmitool sdr elist",
        "ipmitool sel info",
    ]


def test_rejects_writes_and_foreign_tools():
    snippets = [
        "Clear the log with ipmitool sel clear.",
        "ipmitool -H 10.0.0.5 lanplus sel elist",
        "curl -X GET https://bmc/redfish | jq '.Members'",
        "echo 1 > /sys/class/hwmon/...",
    ]
    assert mine_probe_commands(snippets) == []


def test_mines_i2cget_and_i2cdetect():
    snippets = [
        "Read the register with i2cget -y 8 0xb 0x1b.",
        "Probe the bus with i2cdetect -y 8.",
    ]
    assert mine_probe_commands(snippets) == ["i2cget -y 8 0xb 0x1b", "i2cdetect -y 8"]


def test_mines_dmesg_only_with_flags():
    assert mine_probe_commands(["check dmesg for errors",
                                "run dmesg -r on the BMC"]) == ["dmesg -r"]


def test_dedupes_repeated_commands():
    assert mine_probe_commands([GB_HANGUP_P1, GB_HANGUP_P1]) == ["i2cdump -y 8 0xb"]


def test_plain_prose_mines_nothing():
    assert mine_probe_commands(
        ["The server failed to boot. Register 0x1b shows the CPU is in a no boot state."]
    ) == []


def test_plan_collection_carries_doc_probes():
    plan = plan_collection("amber light, server no boot", [GB_HANGUP_P1])
    assert plan.doc_probes == ["i2cdump -y 8 0xb"]
    # Without snippets the plan carries no doc probes.
    assert plan_collection("amber light, server no boot").doc_probes == []


def test_plan_collection_ignores_unsafe_doc_commands():
    plan = plan_collection("amber light",
                           ["[doc p.2] Run `ipmitool sel clear` to reset the log."])
    assert plan.doc_probes == []
