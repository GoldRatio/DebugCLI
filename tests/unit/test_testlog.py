"""Test-log parsing: metadata, failure codes, dedupe, redaction, fallback,
retrieval queries, and the learning-loop terms."""

from harness.testlog import FileLogSource, LogSourceError, parse_test_log

FAT_LOG = """\
DEBUG:2026-08-11 22:01:26 || **************************************************************************
INFO :2026-08-11 22:01:26 || Test Program Version: 2.0.3
INFO :2026-08-11 22:01:34 || |===========================================================|
INFO :2026-08-11 22:01:34 || Model     : T6T
INFO :2026-08-11 22:01:34 || Station   : FAT test start
INFO :2026-08-11 22:01:34 || Chassis SN: P23326287013301E
INFO :2026-08-11 22:01:34 || Test Stage: M3, L10
INFO :2026-08-11 22:01:34 || Build Site: QMF
ERROR:2026-08-11 22:01:34 || [FAIL] Can not find /mnt/monitor/t6t/rm/rm_mapping.ini
DEBUG:2026-08-11 22:02:00 || [START] pcie_cmp_chk
DEBUG:2026-08-11 22:02:00 || Test Action:PCIe device compare check...
DEBUG:2026-08-11 22:02:00 || Executing: lspci | grep '05:01:00.0'
DEBUG:2026-08-11 22:02:00 || 0005:01:00.0 Non-Volatile memory controller: SK hynix Device 2a59
ERROR:2026-08-11 22:02:01 || [FAIL][P02002001@PCIe Test Fail]
ERROR:2026-08-11 22:02:01 || PCIe compare check test Failed!
ERROR:2026-08-11 22:02:01 || Golden Info:
ERROR:2026-08-11 22:02:01 || 0005:01:00.0 SK hynix PE9010 Series
ERROR:2026-08-11 22:02:01 || UUT Info:
ERROR:2026-08-11 22:02:01 || 0005:01:00.0 SK hynix Device 2a59
ERROR:2026-08-11 22:02:01 || PCIe test: FAIL
DEBUG:2026-08-11 22:02:01 || [pcie_cmp_chk] took 1.005s
DEBUG:2026-08-11 22:02:01 || [END] pcie_cmp_chk
DEBUG:2026-08-11 22:02:01 || FAIL: pcie_cmp_chk
DEBUG:2026-08-11 22:02:01 || AssertionError: pcie_cmp_chk failed with code 1
DEBUG:2026-08-11 22:04:47 || [START] pcie_cmp_chk
DEBUG:2026-08-11 22:04:47 || Test Action:PCIe device compare check...
ERROR:2026-08-11 22:04:48 || [FAIL][P02002001@PCIe Test Fail]
ERROR:2026-08-11 22:04:48 || PCIe compare check test Failed!
DEBUG:2026-08-11 22:04:48 || [END] pcie_cmp_chk
DEBUG:2026-08-11 22:04:48 || FAIL: pcie_cmp_chk
DEBUG:2026-08-11 22:04:48 || AssertionError: pcie_cmp_chk failed with code 1
"""


def test_parses_header_metadata():
    report = parse_test_log(FAT_LOG, source="fat.log")
    assert report.source == "fat.log"
    assert report.model == "T6T"
    assert report.serial == "P23326287013301E"
    assert report.station == "FAT test start"
    assert report.test_stage == "M3, L10"
    assert report.build_site == "QMF"
    assert report.program_version == "2.0.3"


def test_extracts_coded_failure_with_test_name_and_dedupes():
    report = parse_test_log(FAT_LOG)
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure.code == "P02002001"
    assert failure.description == "PCIe Test Fail"
    assert failure.signature == "P02002001@PCIe Test Fail"
    assert failure.test_name == "pcie_cmp_chk"
    assert failure.occurrences == 2  # repeated across two test runs
    assert failure.first_seen == "2026-08-11 22:02:01"
    # context holds the surrounding ERROR lines (golden vs UUT mismatch)
    assert any("Golden Info" in c for c in failure.context)
    assert any("Device 2a59" in c for c in failure.context)
    # generic uncoded [FAIL] is not captured as a coded failure
    assert not any("rm_mapping.ini" in c for c in failure.context)


def test_rag_queries_and_case_terms():
    report = parse_test_log(FAT_LOG)
    assert report.rag_queries() == ["PCIe Test Fail P02002001",
                                    "pcie_cmp_chk harness test failure"]
    assert report.case_terms() == ["P02002001@PCIe Test Fail", "pcie_cmp_chk"]


def test_redacts_cleartext_passwords():
    log = (
        "ERROR:2026-08-11 22:02:01 || [FAIL][P00000001@Test] failed\n"
        "ERROR:2026-08-11 22:02:01 || sshpass -p '$pl3nd1D' ssh root@NA cmd\n"
        "ERROR:2026-08-11 22:02:01 || password=secret123\n"
    )
    report = parse_test_log(log)
    assert len(report.failures) == 1
    joined = "\n".join(report.failures[0].context)
    assert "pl3nd1D" not in joined
    assert "secret123" not in joined
    assert "[REDACTED]" in joined


def test_unknown_format_falls_back_to_raw_excerpt():
    text = "random prose\nno structured failure markers anywhere\n" * 10
    report = parse_test_log(text, source="weird.txt")
    assert report.failures == []
    assert report.raw_excerpt and "random prose" in report.raw_excerpt
    assert report.model is None


def test_generic_uncoded_failure_when_no_code_bracket():
    log = "ERROR:2026-08-11 22:01:34 || [FAIL] Can not find rm_mapping.ini\n"
    report = parse_test_log(log)
    assert len(report.failures) == 1
    assert report.failures[0].code == ""
    assert report.failures[0].description == "Can not find rm_mapping.ini"
    assert report.raw_excerpt is None


def test_file_log_source_loads_and_rejects_unreadable(tmp_path):
    path = tmp_path / "fat.log"
    path.write_text(FAT_LOG, encoding="utf-8")
    report = FileLogSource(path).load()
    assert report.source == str(path)
    assert report.failures[0].signature == "P02002001@PCIe Test Fail"

    try:
        FileLogSource(tmp_path / "missing.log").load()
    except LogSourceError:
        pass
    else:
        raise AssertionError("missing file must raise LogSourceError")


def test_summary_lines_render_failure_identity():
    report = parse_test_log(FAT_LOG)
    lines = report.summary_lines()
    assert any(l.startswith("model=T6T") for l in lines)
    assert any("P02002001@PCIe Test Fail" in l for l in lines)
    assert any("test=pcie_cmp_chk" in l for l in lines)
    assert any("occurrences=2" in l for l in lines)