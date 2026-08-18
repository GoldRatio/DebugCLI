"""LLM model catalog: the selectable models behind ``harness session`` /
``harness menu`` and the ``--llm-model`` flag.

"Model" here is the LLM reasoning backend (provider + model id + endpoint), NOT
the server model being diagnosed (that is ``Host.model`` / ``model_hint``). The
catalog turns the operator's pick into a concrete ``ModelProfile`` that
``operator.cli._resolve_llm`` builds into a live adapter, and it remembers the
last pick in a machine-local ``config/models.yaml`` (already git-ignored via
``config/*.yaml``) so the next ``harness`` run keeps the same model -- the way
opencode/pi keep the selected model for the session and the project.

Sources of *available* models (deduplicated by ident):
- the persisted ``models:`` list (operator-added custom endpoints/models),
- the inventory ``llm`` block,
- the well-known defaults (``openai/harness-diag``, ``gemini/gemini-2.5-flash``),
- ``stub`` (pipeline-only, no reasoning) -- always available last.

Resolution precedence for the *current* model:
``--llm-model <ident>`` > ``--llm <provider>`` > persisted ``current`` >
inventory ``llm`` block > default (``openai/harness-diag``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_PROVIDERS = ("openai", "gemini", "stub")

_DEFAULT_MODELS = {"openai": "harness-diag", "gemini": "gemini-2.5-flash"}


@dataclass(frozen=True)
class ModelProfile:
    """One selectable LLM backend. ``url``/``api_key_vault_path`` are optional
    and fall back to each adapter's env defaults when omitted."""

    provider: str = "openai"
    model: str = "harness-diag"
    url: str | None = None
    api_key_vault_path: str | None = None
    timeout: float = 120.0
    label: str | None = None

    @property
    def ident(self) -> str:
        """Stable calibration key, e.g. ``gemini/gemini-2.5-flash``; ``stub``
        for the pipeline-only backend."""
        return "stub" if self.provider == "stub" else f"{self.provider}/{self.model}"

    @property
    def display(self) -> str:
        return self.label or self.ident

    def build(self, store: object) -> object:
        """Build the live adapter. The API key, when vault-path configured, is
        resolved through the secret store; otherwise the adapter falls back to
        env (``GEMINI_API_KEY`` / ``HARNESS_LLM_API_KEY``)."""
        from ..diagnosis.llm import GeminiLLM, OpenAICompatLLM, StubLLM

        api_key = None
        if self.api_key_vault_path:
            try:
                api_key = store.get(self.api_key_vault_path).decode().strip()
            except (KeyError, AttributeError):
                pass
        if self.provider == "stub":
            return StubLLM()
        if self.provider == "gemini":
            return GeminiLLM(url=self.url, api_key=api_key, model=self.model,
                             timeout=self.timeout)
        return OpenAICompatLLM(url=self.url, api_key=api_key, model=self.model,
                               timeout=self.timeout)

    def to_dict(self) -> dict:
        d = {"provider": self.provider, "model": self.model}
        if self.url:
            d["url"] = self.url
        if self.api_key_vault_path:
            d["api_key_vault_path"] = self.api_key_vault_path
        if self.timeout != 120.0:
            d["timeout"] = self.timeout
        if self.label:
            d["label"] = self.label
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ModelProfile:
        if not isinstance(data, dict):
            raise TypeError("model entry must be an object")
        provider = str(data.get("provider", "")).strip().lower()
        if provider not in _PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        model = str(data.get("model", "")).strip()
        if provider == "stub":
            model = "stub"
        if not model:
            raise ValueError("model id is required")
        return cls(
            provider=provider,
            model=model,
            url=str(data["url"]).strip() if data.get("url") else None,
            api_key_vault_path=(
                str(data["api_key_vault_path"]).strip()
                if data.get("api_key_vault_path") else None),
            timeout=float(data.get("timeout", 120.0)),
            label=str(data["label"]).strip() if data.get("label") else None,
        )


class ModelCatalog:
    """The selectable models plus the current pick (and how to persist it)."""

    def __init__(self, available: list[ModelProfile], current: ModelProfile,
                 user_models: list[ModelProfile]) -> None:
        self.available = available
        self.current = current
        self.user_models = user_models

    @classmethod
    def load(cls, path: str | Path = "config/models.yaml",
             inv: object | None = None) -> ModelCatalog:
        """Build the catalog from the persisted file and the inventory's
        ``llm`` block. A missing or corrupt file degrades to the inventory /
        defaults -- the catalog never blocks a run."""
        data = _read_json(path)
        user_models: list[ModelProfile] = []
        for entry in data.get("models") or []:
            try:
                user_models.append(ModelProfile.from_dict(entry))
            except ValueError:
                continue

        inv_profile = None
        inv_llm = getattr(inv, "llm", None)
        if inv is not None and inv_llm is not None:
            model = (inv_llm.model or os.environ.get("HARNESS_LLM_MODEL")
                     or _DEFAULT_MODELS.get(inv_llm.provider, "harness-diag"))
            inv_profile = ModelProfile(
                provider=inv_llm.provider,
                model=model,
                url=inv_llm.url,
                api_key_vault_path=inv_llm.api_key_vault_path,
                timeout=inv_llm.timeout,
            )

        available = _dedup([*user_models,
                            *([inv_profile] if inv_profile else []),
                            *_DEFAULTS])

        current = None
        raw_current = data.get("current")
        if isinstance(raw_current, dict):
            try:
                current = ModelProfile.from_dict(raw_current)
            except ValueError:
                current = None
        if current is None:
            current = inv_profile or DEFAULT_CURRENT
        return cls(available=available, current=current, user_models=user_models)

    def resolve(self, provider: str | None = None,
                model_id: str | None = None) -> ModelProfile:
        """The effective profile under the resolution precedence (flag-provider
        and ``model_id`` come from the CLI; otherwise the remembered current)."""
        if model_id:
            found = self.get(model_id)
            if found is not None:
                return found
            if "/" in model_id:
                prov, name = model_id.split("/", 1)
                if prov in _PROVIDERS and name:
                    return ModelProfile(provider=prov, model=name)
            if provider in _PROVIDERS:
                fallback = provider
            else:
                fallback = (self.current.provider
                            if self.current.provider != "stub" else "openai")
            return ModelProfile(provider=fallback, model=model_id)
        if provider:
            return self.for_provider(provider)
        return self.current

    def get(self, ident_or_model: str) -> ModelProfile | None:
        """Match a profile by full ident (``gemini/gemini-2.5-flash``) or bare
        model id (``gemini-2.5-flash``)."""
        needle = ident_or_model.strip()
        for p in self.available:
            if p.ident == needle or (p.provider != "stub" and p.model == needle):
                return p
        return None

    def for_provider(self, provider: str) -> ModelProfile:
        if provider == "stub":
            return _stub_profile()
        for p in self.available:
            if p.provider == provider:
                return p
        # provider known but not present in the catalog -> a sensible default
        if provider == "gemini":
            return ModelProfile(provider="gemini", model="gemini-2.5-flash")
        return ModelProfile(provider="openai", model="harness-diag")

    def choose(self, profile: ModelProfile) -> None:
        """Make ``profile`` the current (remembered) model."""
        self.current = profile

    def add(self, profile: ModelProfile) -> None:
        """Register a custom model (persisted in the user ``models:`` list) and
        select it."""
        if not any(p.ident == profile.ident for p in self.user_models):
            self.user_models.append(profile)
        if not any(p.ident == profile.ident for p in self.available):
            self.available = _dedup([profile, *self.available])
        self.current = profile

    def save(self, path: str | Path = "config/models.yaml") -> None:
        """Persist the current pick + the user-added models list."""
        payload = {
            "current": self.current.to_dict() if self.current else None,
            "models": [p.to_dict() for p in self.user_models],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


DEFAULT_CURRENT = ModelProfile(provider="openai", model="harness-diag")

_STUB = ModelProfile(provider="stub", model="stub",
                     label="stub (pipeline only, no reasoning)")

_DEFAULTS = [
    DEFAULT_CURRENT,
    ModelProfile(provider="gemini", model="gemini-2.5-flash",
                 label="gemini-2.5-flash (Google Gemini, OpenAI-compatible)"),
    _STUB,
]


def _stub_profile() -> ModelProfile:
    return _STUB


def _read_json(path: str | Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _dedup(profiles: list[ModelProfile]) -> list[ModelProfile]:
    seen: set[str] = set()
    out: list[ModelProfile] = []
    for p in profiles:
        if p.ident in seen:
            continue
        seen.add(p.ident)
        out.append(p)
    return out


def picker_rows(catalog: ModelCatalog) -> tuple[list[str], list[ModelProfile], int]:
    """Render the interactive picker: option labels (current marked), the
    profiles in the same order, and the index of the "+ add" row."""
    labels: list[str] = []
    profiles: list[ModelProfile] = []
    current_ident = catalog.current.ident if catalog.current else None
    for p in catalog.available:
        mark = "  [current]" if p.ident == current_ident else ""
        labels.append(f"{p.display}{mark}")
        profiles.append(p)
    add_idx = len(profiles)
    labels.append("+ add a custom model (provider, model id, optional url / key vault path)")
    return labels, profiles, add_idx