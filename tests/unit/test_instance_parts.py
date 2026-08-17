"""Instance-parts store tests: per-target operator-answered parts persistence."""

from harness.docs.parts.instance_store import InstancePartsStore, merge_store_into_parts


def test_instance_store_merge_load_and_path(tmp_path):
    store = InstancePartsStore(tmp_path)
    assert store.load("q75-cable2") == {}

    store.merge("q75-cable2", [
        {"slot": "right_bianca", "fru": "Bianca Board",
         "pn": "PN-1234", "sn": "SN-5678", "rail": "pdb 12v"},
        {"slot": "left_bianca", "fru": "Bianca Board", "rail": "pdb 12v"},
    ])
    parts = store.load("q75-cable2")
    assert parts["right_bianca"]["pn"] == "PN-1234"
    assert parts["right_bianca"]["sn"] == "SN-5678"
    assert parts["right_bianca"]["rail"] == "pdb 12v"
    assert "slot" not in parts["right_bianca"]  # slot is the key, not a field
    assert parts["left_bianca"]["fru"] == "Bianca Board"

    # Upsert overwrites a slot; other slots survive.
    store.merge("q75-cable2", [{"slot": "right_bianca", "pn": "PN-9999"}])
    parts = store.load("q75-cable2")
    assert parts["right_bianca"]["pn"] == "PN-9999"
    assert parts["left_bianca"]["fru"] == "Bianca Board"

    # Blank slots are ignored.
    store.merge("q75-cable2", [{"slot": "  "}])
    assert store.load("q75-cable2")["right_bianca"]["pn"] == "PN-9999"


def test_instance_store_corrupt_or_missing_degrade_to_empty(tmp_path):
    store = InstancePartsStore(tmp_path)
    p = store.path_for("broken")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert store.load("broken") == {}


def test_instance_store_target_key_is_filesystem_safe(tmp_path):
    store = InstancePartsStore(tmp_path)
    store.merge("Q61 Cable 8", [{"slot": "a", "fru": "f"}])
    assert store.path_for("Q61 Cable 8").exists()


def test_merge_store_into_parts_csv_wins():
    stored = {"a": {"fru": "stored", "rail": "pdb 12v"},
              "b": {"fru": "stored"}}
    csv = {"a": {"fru": "csv"}}
    merged = merge_store_into_parts(csv, stored)
    assert merged["a"]["fru"] == "csv"      # explicit CSV is authoritative
    assert merged["a"]["rail"] == "pdb 12v"  # CSV slot keeps stored rail info
    assert merged["b"]["fru"] == "stored"
    assert merge_store_into_parts(None, stored) == stored
    assert merge_store_into_parts(csv, None) == csv