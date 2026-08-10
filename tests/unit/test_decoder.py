"""Decoder, catalog, and the no-guess-on-unknown invariant."""

import pytest

from harness.inspect.decoder import Decoder
from harness.inspect.catalog.catalog_loader import load_catalog, CatalogUnavailable


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