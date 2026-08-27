"""LLM model catalog: profile ident/display, resolution precedence, persistence,
and adapter construction."""

import json

import pytest

from harness.config.model_catalog import ModelCatalog, ModelProfile, picker_rows
from harness.config.models import Inventory, LLMConfig
from harness.config.vault import MemorySecretStore
from harness.diagnosis.llm import GeminiLLM, LocalLLM, OpenAICompatLLM, StubLLM

# ---- ModelProfile ----

def test_profile_ident_and_display():
    assert ModelProfile(provider="gemini", model="gemini-2.5-flash").ident == \
        "gemini/gemini-2.5-flash"
    assert ModelProfile(provider="openai", model="gpt-4o").ident == "openai/gpt-4o"
    assert ModelProfile(provider="stub", model="stub").ident == "stub"
    assert ModelProfile().ident == "openai/harness-diag"
    assert ModelProfile(provider="stub", model="stub",
                        label="no reasoning").display == "no reasoning"
    assert ModelProfile(provider="openai", model="m").display == "openai/m"


def test_profile_round_trip_dict():
    p = ModelProfile(provider="gemini", model="gemini-2.5-pro", url="http://gw/v1",
                     api_key_vault_path="secret/llm/key", timeout=60.0,
                     label="stronger")
    assert ModelProfile.from_dict(p.to_dict()) == p
    assert ModelProfile.from_dict({"provider": "stub"}).ident == "stub"


def test_profile_from_dict_rejects_bad_entries():
    for bad in ({"provider": "anthropic", "model": "x"},
                {"provider": "openai"}):  # missing model id
        try:
            ModelProfile.from_dict(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")
    with pytest.raises(TypeError):  # not an object at all
        ModelProfile.from_dict("not a dict")


# ---- catalog assembly ----

def test_catalog_defaults_with_no_file(tmp_path):
    catalog = ModelCatalog.load(tmp_path / "nope.yaml")
    assert [p.ident for p in catalog.available] == [
        "openai/harness-diag", "gemini/gemini-2.5-flash",
        "local/harness-diag", "stub"]
    assert catalog.current.ident == "openai/harness-diag"
    assert catalog.resolve().ident == "openai/harness-diag"


def test_catalog_from_inventory_block(tmp_path):
    inv = Inventory(trust_level="lab",
                    llm=LLMConfig(provider="gemini", model="gemini-2.5-flash"))
    catalog = ModelCatalog.load(tmp_path / "nope.yaml", inv=inv)
    assert catalog.current.ident == "gemini/gemini-2.5-flash"
    assert catalog.resolve().ident == "gemini/gemini-2.5-flash"
    assert any(p.ident == "openai/harness-diag" for p in catalog.available)


def test_catalog_inventory_matches_default_dedup(tmp_path):
    inv = Inventory(trust_level="lab",
                    llm=LLMConfig(provider="openai", model="harness-diag"))
    catalog = ModelCatalog.load(tmp_path / "nope.yaml", inv=inv)
    idents = [p.ident for p in catalog.available]
    assert idents.count("openai/harness-diag") == 1


def test_catalog_persisted_current_wins_over_inventory(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(json.dumps({
        "current": {"provider": "gemini", "model": "gemini-2.5-pro"}}))
    inv = Inventory(trust_level="lab",
                    llm=LLMConfig(provider="openai", model="gpt-4o"))
    catalog = ModelCatalog.load(path, inv=inv)
    assert catalog.current.ident == "gemini/gemini-2.5-pro"
    assert catalog.resolve().ident == "gemini/gemini-2.5-pro"


def test_catalog_corrupt_file_degrades_to_defaults(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("not json {{", encoding="utf-8")
    catalog = ModelCatalog.load(path)
    assert catalog.current.ident == "openai/harness-diag"


def test_catalog_user_models_loaded(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(json.dumps({
        "models": [{"provider": "openai", "model": "gpt-4o",
                    "url": "http://gw:8000/v1"}]}))
    catalog = ModelCatalog.load(path)
    assert any(p.ident == "openai/gpt-4o" for p in catalog.available)
    assert any(p.ident == "openai/gpt-4o" for p in catalog.user_models)


# ---- resolution precedence ----

def test_flag_provider_beats_persisted(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(json.dumps({
        "current": {"provider": "gemini", "model": "gemini-2.5-pro"}}))
    catalog = ModelCatalog.load(path)
    assert catalog.resolve(provider="openai").ident == "openai/harness-diag"
    assert catalog.resolve(provider="stub").ident == "stub"


def test_model_id_beats_provider_flag(tmp_path):
    catalog = ModelCatalog.load(tmp_path / "nope.yaml")
    p = catalog.resolve(provider="openai", model_id="gemini/gemini-2.5-flash")
    assert p.ident == "gemini/gemini-2.5-flash"
    # a bare model id inherits the provider flag ...
    p2 = catalog.resolve(provider="gemini", model_id="gpt-4o")
    assert p2.ident == "gemini/gpt-4o"
    # ... and otherwise the remembered provider.
    p3 = catalog.resolve(model_id="gpt-4o")
    assert p3.ident == "openai/gpt-4o"


def test_get_matches_ident_and_bare_model(tmp_path):
    catalog = ModelCatalog.load(tmp_path / "nope.yaml")
    assert catalog.get("gemini/gemini-2.5-flash").ident == "gemini/gemini-2.5-flash"
    assert catalog.get("gemini-2.5-flash").ident == "gemini/gemini-2.5-flash"
    assert catalog.get("stub").ident == "stub"
    assert catalog.get("does-not-exist") is None


# ---- persistence ----

def test_add_and_save_round_trip(tmp_path):
    path = tmp_path / "nested" / "models.yaml"
    catalog = ModelCatalog.load(path)
    catalog.add(ModelProfile(provider="openai", model="gpt-4o",
                             url="http://gw:8000/v1"))
    assert catalog.current.ident == "openai/gpt-4o"
    catalog.save(path)
    assert path.exists()

    reloaded = ModelCatalog.load(path)
    assert reloaded.current.ident == "openai/gpt-4o"
    assert any(p.ident == "openai/gpt-4o" for p in reloaded.user_models)
    assert any(p.ident == "openai/gpt-4o" for p in reloaded.available)


def test_save_persists_only_user_models(tmp_path):
    path = tmp_path / "models.yaml"
    catalog = ModelCatalog.load(path)
    catalog.add(ModelProfile(provider="gemini", model="gemini-2.5-pro"))
    catalog.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["current"] == {"provider": "gemini", "model": "gemini-2.5-pro"}
    assert data["models"] == [{"provider": "gemini", "model": "gemini-2.5-pro"}]


# ---- adapter construction ----

def test_build_llm_instances(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("HARNESS_LLM_API_KEY", raising=False)
    store = MemorySecretStore({"secret/llm/key": b"sk-123\n"})
    assert isinstance(ModelProfile(provider="stub").build(store), StubLLM)
    g = ModelProfile(provider="gemini", model="gemini-2.5-flash",
                     api_key_vault_path="secret/llm/key").build(store)
    assert isinstance(g, GeminiLLM) and g.api_key == "sk-123"
    o = ModelProfile(provider="openai", model="harness-diag").build(store)
    assert isinstance(o, OpenAICompatLLM) and o.model == "harness-diag"
    assert ModelProfile(provider="openai", model="m").build(store).api_key is None


def test_build_missing_vault_path_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k-env")
    store = MemorySecretStore()
    g = ModelProfile(provider="gemini", model="gemini-2.5-flash",
                     api_key_vault_path="secret/absent").build(store)
    assert g.api_key == "k-env"


# ---- picker rows ----

def test_picker_rows_marks_current_and_add_row(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(json.dumps({
        "current": {"provider": "gemini", "model": "gemini-2.5-flash"}}))
    catalog = ModelCatalog.load(path)
    labels, profiles, add_idx = picker_rows(catalog)
    assert len(profiles) == add_idx
    assert "[current]" in labels[1]          # gemini-2.5-flash marked
    assert "[current]" not in labels[0]      # openai default not marked
    assert labels[add_idx].startswith("+ add a custom model")


# ---- local provider ----

def test_local_provider_profile_and_ident():
    p = ModelProfile.from_dict({"provider": "local", "model": "Qwen2.5-7B-Instruct",
                                "url": "http://10.0.0.42:8000/v1"})
    assert p.ident == "local/Qwen2.5-7B-Instruct"
    assert ModelProfile(provider="local").ident == "local/harness-diag"


def test_resolve_url_override_wins_over_catalog_entry(tmp_path):
    """``--llm-url`` rewrites the endpoint of whichever profile resolves --
    a catalog hit, an ad-hoc provider/model pair, or the remembered current."""
    catalog = ModelCatalog.load(tmp_path / "nope.yaml")
    pinned = "http://10.0.0.42:8000/v1"
    hit = catalog.resolve(model_id="gemini/gemini-2.5-flash", url=pinned)
    assert hit.url == pinned and hit.provider == "gemini"
    adhoc = catalog.resolve(provider="openai", model_id="gpt-4o", url=pinned)
    assert adhoc.url == pinned and adhoc.model == "gpt-4o"
    current = catalog.resolve(url=pinned)
    assert current.url == pinned
    # absent url -> profile untouched (catalog/env defaults apply)
    assert catalog.resolve().url is None


def test_build_local_llm_instance():
    store = MemorySecretStore()
    llm = ModelProfile(provider="local", model="Qwen2.5-7B-Instruct",
                       url="http://127.0.0.1:9/v1").build(store)
    assert isinstance(llm, LocalLLM)
    assert llm.model == "Qwen2.5-7B-Instruct"
    assert llm.json_mode is True              # vLLM supports response_format