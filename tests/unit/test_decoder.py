"""Decoder, catalog, and the no-guess-on-unknown invariant."""

import pytest

from harness.inspect.catalog.catalog_loader import CatalogUnavailable, load_catalog
from harness.inspect.decoder import Decoder


def test_decode_known_register_from_catalog():
    dec = Decoder()
    result = dec.decode("IA32_MC0_STATUS", "0x8000000000000001")
    assert result.unknown is False
    assert result.page_ref == "p. 78"
    # bit 63 valid set
    valid = next(f for f in result.decoded_fields if f.name == "valid")
    assert valid.raw_value == "1"
    assert valid.meaning == "error logged"


def test_decode_unknown_register_is_flagged_not_guessed():
    dec = Decoder()
    result = dec.decode("IA32_SOMETHING_UNKNOWN", "0x1")
    assert result.unknown is True
    assert result.decoded_fields == []


def test_decode_many_from_rdmsr_style_output():
    dec = Decoder()
    lines = "IA32_MC0_STATUS = 0x8000000000000009\nIA32_MCG_STATUS = 0x7\n"
    results = dec.decode_many(lines)
    mnemonics = {r.mnemonic for r in results}
    assert "IA32_MC0_STATUS" in mnemonics
    assert "IA32_MCG_STATUS" in mnemonics


def test_catalog_rejects_unreviewed(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1.0.0\nsignoff:\n  reviewed: false\nregisters: {}\n", encoding="utf-8")
    with pytest.raises(CatalogUnavailable):
        load_catalog(bad)


def test_frozen_lookup_case_insensitive():
    dec = Decoder()
    assert dec.decode("ia32_mc0_status", "0x0").mnemonic == "IA32_MC0_STATUS"


def test_lookup_addr_finds_cpld_register_case_insensitive():
    dec = Decoder()
    assert dec.catalog.lookup_addr("0x1b").mnemonic == "CPLD_1B_CRITICAL"
    assert dec.catalog.lookup_addr("0xa1").mnemonic == "CPLD_A1_BOOT_STATE"
    assert dec.catalog.lookup_addr("0x2a") is None


def test_decode_i2c_dump_surfaces_boot_state_registers():
    dec = Decoder()
    raw = (
        "     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n"
        "0b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
        "1b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
        "a1: 05 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
    )
    results = dec.decode_i2c_dump(raw)
    by_name = {r.mnemonic: r for r in results}
    # 0x1b (CPU_BOOT_DONE, RUN_POWER_PG ...) and 0xa1 (power-seq FSM) decoded.
    assert "CPLD_1B_CRITICAL" in by_name
    assert "CPLD_A1_BOOT_STATE" in by_name
    boot = by_name["CPLD_1B_CRITICAL"]
    assert boot.raw_hex == "0x00"
    done = next(f for f in boot.decoded_fields if f.name == "cpu_boot_done")
    assert done.meaning == "CPU in POST / no boot state"
    fsm = by_name["CPLD_A1_BOOT_STATE"]
    assert fsm.raw_hex == "0x05"
    state = next(f for f in fsm.decoded_fields if f.name == "power_seq_fsm")
    assert state.meaning == "s5_pwron_wait"
    assert state.raw_value == "5"


def test_decode_i2c_dump_flags_abnormal_bits_with_values():
    dec = Decoder()
    raw = "1b: 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
    results = dec.decode_i2c_dump(raw)
    boot = next(r for r in results if r.mnemonic == "CPLD_1B_CRITICAL")
    done = next(f for f in boot.decoded_fields if f.name == "cpu_boot_done")
    assert done.raw_value == "1"
    assert done.meaning == "CPU POST completed"


def test_decode_i2c_dump_ignores_uncataloged_offsets():
    dec = Decoder()
    raw = "0b: 05 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
    assert dec.decode_i2c_dump(raw) == []  # 0x0b..0x1a have no catalog entry


def test_decode_i2c_transfer_surfaces_boot_state_registers():
    dec = Decoder()
    # 256-byte block read; bytes at 0x1b (boot flags) and 0xa1 (FSM) are set.
    data = ["00"] * 256
    data[0x1B] = "00"
    data[0xA1] = "05"
    raw = (
        "spawn jumpin q63-1 rm\n"
        "*****************************\n"
        "RScmCli# start serial session -i 2 -p 2200\n"
        "admin@m1120-c4a15:~$ sudo -S i2ctransfer -y 8 w1@0xb 0x00 r256\n"
        "Password: \n"
        + " ".join(data) + "\n"
        "admin@m1120-c4a15:~$ \n"
    )
    results = dec.decode_i2c_transfer(raw, "sudo -S i2ctransfer -y 8 w1@0xb 0x00 r256")
    by_name = {r.mnemonic: r for r in results}
    assert "CPLD_1B_CRITICAL" in by_name
    assert by_name["CPLD_1B_CRITICAL"].raw_hex == "0x00"
    assert "CPLD_A1_BOOT_STATE" in by_name
    assert by_name["CPLD_A1_BOOT_STATE"].raw_hex == "0x05"


def test_decode_i2c_transfer_ignores_banner_and_prompt_hex():
    dec = Decoder()
    data = ["00"] * 256
    data[0xA1] = "05"
    raw = (
        "spawn jumpin q63-1 rm\n"
        "RScmCli# start serial session -i 2 -p 2200\n"
        "admin@m1120-c4a15:~$ sudo -S i2ctransfer -y 8 w1@0xb 0x00 r256\n"
        "Password: \n"
        + " ".join(data) + "\n"
        "admin@m1120-c4a15:~$ \n"
    )
    results = dec.decode_i2c_transfer(raw, "sudo -S i2ctransfer -y 8 w1@0xb 0x00 r256")
    by_name = {r.mnemonic: r for r in results}
    # Prompt hex (e.g. "ad 11 20 c4 a1 15" from the banner) must not bleed into
    # the stream and shift offsets: CPLD_A1_BOOT_STATE keeps the 0x05 we placed.
    assert by_name["CPLD_A1_BOOT_STATE"].raw_hex == "0x05"
    assert by_name["CPLD_1B_CRITICAL"].raw_hex == "0x00"
    # Only cataloged offsets within the 256-byte block surface -- nothing more.
    expected = {d.mnemonic for i in range(256)
                if (d := dec.catalog.lookup_addr(f"0x{i:02X}")) is not None}
    assert {r.mnemonic for r in results} == expected


def test_decode_i2c_get_single_register():
    dec = Decoder()
    raw = (
        "admin@m1120-c4a15:~$ sudo -S i2cget -y 8 0xb 0x1b\n"
        "Password: \n"
        "0x08\n"
        "admin@m1120-c4a15:~$ \n"
    )
    results = dec.decode_i2c_get(raw, "sudo -S i2cget -y 8 0xb 0x1b")
    assert len(results) == 1
    boot = results[0]
    assert boot.mnemonic == "CPLD_1B_CRITICAL"
    assert boot.raw_hex == "0x08"
    done = next(f for f in boot.decoded_fields if f.name == "cpu_boot_done")
    assert done.meaning == "CPU POST completed"


def test_decode_i2c_get_without_source_or_uncataloged_is_empty():
    dec = Decoder()
    assert dec.decode_i2c_get("0x05", "") == []
    assert dec.decode_i2c_get("0x05", "sudo -S i2cget -y 8 0xb 0x2a") == []  # no catalog entry