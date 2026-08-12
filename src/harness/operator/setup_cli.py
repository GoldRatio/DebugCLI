"""One-time setup wizard (``harness setup``) — non-agent credential CLI.

Walks a NEW machine through what used to be a manual chore: point at an
inventory (or create one), register the LLM API key (never echoed), make sure
an SSH identity exists (generating an ed25519 keypair when there is none) and
lands it at the vault path the inventory references, optionally register the
BMC/sudo passwords, then verify every vault path the inventory mentions.

Everything here is interactive/non-agent (like ``operator/secrets_cli``):
prompts never see key material. Answers are injectable via ``overrides`` so
tests drive the wizard deterministically without a tty.
"""

from __future__ import annotations

import getpass
import shutil
import subprocess
import sys
from pathlib import Path

from ..config.inventory_lint import load_inventory
from ..config.vault import DirSecretStore

DEFAULT_SECRET_DIR = "secrets"
LLM_VAULT = "secret/harness/llm"
DIAGBOT_SSH_VAULT = "secret/harness/diagbot/id_ed25519"
RACKMGR_SSH_VAULT = "secret/harness/rackmgr/id_ed25519"
BMC_PASSWORD_VAULT = "secret/harness/bmc/bmc-ro"
BMC_SUDO_VAULT = "secret/harness/bmc/sudo"

_KEY_HEADER_HINTS = (b"-----BEGIN", b"ssh-")


class SetupError(RuntimeError):
    """User-facing setup failure (missing tool, bad key file, ...)."""


def _store_for(args) -> DirSecretStore:
    secret_dir = Path(getattr(args, "secret_dir", None) or DEFAULT_SECRET_DIR)
    secret_dir.mkdir(parents=True, exist_ok=True)
    return DirSecretStore(secret_dir)


def _ask(args, overrides: dict, prompt: str, default: str = "") -> str:
    ask = overrides.get("ask")
    if ask is None:
        from .menu import ask_text
        ask = ask_text
    return (ask(prompt, default=default) or "").strip() or default


def _confirm(args, overrides: dict, prompt: str, default: bool = False) -> bool:
    confirm = overrides.get("confirm")
    if confirm is None:
        from .menu import confirm as _confirm
        confirm = _confirm
    return confirm(prompt, default=default)


def _ask_secret(args, overrides: dict, prompt: str) -> str:
    secret = overrides.get("secret")
    if secret is None:
        secret = getpass.getpass
    return secret(prompt) or ""


# ---- inventory ----

_MINIMAL_INVENTORY = (
    "trust_level: lab\n"
    "llm:\n"
    "  provider: {provider}\n"
    "  model: {model}\n"
    "  api_key_vault_path: {vault}\n"
    "hosts: []\n"
)


def _llm_vault_path(provider: str, inventory_text: str) -> str:
    """The inventory's configured LLM key path, or a per-provider default."""
    for line in inventory_text.splitlines():
        line = line.strip()
        if line.startswith("api_key_vault_path:"):
            path = line.split(":", 1)[1].strip()
            if path:
                return path
    return f"{LLM_VAULT}/{provider}-key"


def _patch_llm_block(text: str, *, provider: str, model: str,
                     vault_path: str) -> str:
    """Replace the top-level ``llm:`` provider/model lines and set the key path.

    Line-based and strict: it only rewrites lines that live one level under a
    top-level ``llm:`` key; anything else in the file is untouched. An absent
    ``llm:`` block is inserted before the first top-level key.
    """
    lines = text.splitlines()
    out: list[str] = []
    in_llm = False
    llm_done = False
    llm_end = 0  # index in ``out`` of the last emitted llm line (header or child)
    for line in lines:
        if line == "llm:" or line.startswith("llm: "):
            in_llm = True
            llm_done = False
            out.append("llm:")
            llm_end = len(out) - 1
            continue
        if in_llm:
            if line and not line[0].isspace():
                in_llm = False
            else:
                if not llm_done:
                    if line.startswith("  provider:"):
                        out.append(f"  provider: {provider}")
                        llm_end = len(out) - 1
                        continue
                    if line.startswith("  model:"):
                        out.append(f"  model: {model}" if model
                                   else "  # model: (set this)")
                        out.append(f"  api_key_vault_path: {vault_path}")
                        llm_end = len(out) - 1
                        llm_done = True
                        continue
                    if line.startswith("  api_key_vault_path:"):
                        out.append(f"  api_key_vault_path: {vault_path}")
                        llm_end = len(out) - 1
                        llm_done = True
                        continue
                out.append(line)
                llm_end = len(out) - 1
                continue
        out.append(line)
    if not any(l == "llm:" for l in out):
        block = [(f"llm:\n  provider: {provider}\n  model: {model}\n"
          f"  api_key_vault_path: {vault_path}\n")]
        return "".join(block) + (text if text.endswith("\n") else text + "\n")
    if not llm_done:
        # llm block exists but has no model/vault lines: splice them in after
        # its last emitted line (the header when the block is empty).
        add = ([f"  model: {model}"] if model
               else ["  # model: (set this)"])
        add.append(f"  api_key_vault_path: {vault_path}")
        out[llm_end + 1:llm_end + 1] = add
    return "\n".join(out) + "\n"


def _ensure_inventory(args, overrides: dict) -> Path:
    explicit = getattr(args, "inventory", None)
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SetupError(f"inventory not found: {path}")
        return path
    existing = Path("inventory.yaml")
    if existing.exists():
        keep = _confirm(
            args, overrides,
            f"Use existing inventory {existing}? (n creates a minimal one)",
            default=True)
        if keep:
            return existing
        overwrite = _confirm(
            args, overrides,
            f"Overwrite {existing} with a minimal inventory? (host edits are lost)",
            default=False)
        if not overwrite:
            return existing
    provider = _ask(args, overrides, "LLM provider (openai|gemini|stub)",
                    default="gemini").lower()
    model = _ask(args, overrides, "Model (e.g. gemini-2.5-flash)")
    vault = f"{LLM_VAULT}/{provider}-key"
    existing.write_text(
        _MINIMAL_INVENTORY.format(provider=provider, model=model, vault=vault),
        encoding="utf-8")
    print(f"wrote {existing} (hosts: [] -- add named hosts or use "
          "--address/--rack/--cable targeting)")
    return existing


# ---- LLM key ----

def _setup_llm(args, overrides: dict, inv_path: Path, store: DirSecretStore,
               inventory_text: str) -> None:

    inv = load_inventory(inv_path)
    provider = getattr(inv.llm, "provider", None) or "gemini"
    if provider == "stub":
        print("llm: provider is stub -- no API key needed (pipeline-only)")
        return
    model = _ask(args, overrides, f"Model for {provider} "
                 "(blank keeps the inventory value)",
                 default=getattr(inv.llm, "model", None) or "")
    vault = _llm_vault_path(provider, inventory_text)
    try:
        store.get(vault)
        print(f"llm: API key already registered at {vault}")
        have_key = True
    except KeyError:
        have_key = False
    if not have_key:
        key = _ask_secret(args, overrides, f"API key for {provider}: ")
        if not key:
            print("llm: skipped (no key) -- diagnoses will fall back to "
                  "HARNESS_LLM_API_KEY or fail at first LLM call")
            return
        store.put(vault, key.encode("utf-8"))
        print(f"llm: API key stored at {vault} (never echoed)")
    if model:
        new_text = _patch_llm_block(
            inventory_text, provider=provider, model=model, vault_path=vault)
        inv_path.write_text(new_text, encoding="utf-8")
        print(f"llm: inventory {inv_path} updated "
              f"(provider={provider} model={model})")


# ---- SSH identity ----

def _generate_ed25519(private_path: Path) -> None:
    ssh_keygen = shutil.which("ssh-keygen")
    if not ssh_keygen:
        raise SetupError(
            "ssh-keygen not found: install OpenSSH client, or provide an "
            "existing key file")
    private_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [ssh_keygen, "-t", "ed25519", "-N", "", "-C", "harness diagbot",
         "-f", str(private_path)],
        capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SetupError(f"ssh-keygen failed: {result.stderr.strip()}")


def _public_key_for(private_path: Path) -> str:
    pub = private_path.with_suffix(".pub")
    return pub.read_text(encoding="utf-8").strip() if pub.exists() else ""


def _setup_ssh(args, overrides: dict, store: DirSecretStore) -> str:
    """Ensure an SSH identity at the inventory's diagbot vault path.

    Returns the public key line (empty when nothing was ensured).
    """
    key_dir = Path(getattr(args, "secret_dir", None) or DEFAULT_SECRET_DIR)
    try:
        store.get(DIAGBOT_SSH_VAULT)
        print(f"ssh: identity already registered at {DIAGBOT_SSH_VAULT}")
        return ""
    except KeyError:
        pass
    if not _confirm(args, overrides, "No SSH key registered -- set one up now?",
                    default=True):
        print("ssh: skipped -- diagnoses over SSH will fail to connect")
        return ""
    choice = _ask(
        args, overrides,
        "Generate a new ed25519 key or use an existing file? (generate|file)",
        default="generate").lower()
    if choice.startswith("file"):
        key_file = Path(_ask(args, overrides, "Private key FILE path"))
        if not key_file.is_file():
            raise SetupError(f"key file not found: {key_file}")
        material = key_file.read_bytes()
        if not material.strip():
            raise SetupError(f"key file {key_file} is empty")
        if not material.startswith(_KEY_HEADER_HINTS):
            print(f"  warning: {key_file} does not look like a private key "
                  "(expects PEM or OpenSSH format)", file=sys.stderr)
        store.put(DIAGBOT_SSH_VAULT, material)
        print(f"ssh: registered {key_file} -> {DIAGBOT_SSH_VAULT}")
        return ""
    private = key_dir / "id_ed25519"
    _generate_ed25519(private)
    store.put(DIAGBOT_SSH_VAULT, private.read_bytes())
    print(f"ssh: generated {private} -> {DIAGBOT_SSH_VAULT}")
    return _public_key_for(private)


def _setup_bmc(args, overrides: dict, store: DirSecretStore, ssh_material:
               bytes | None) -> None:
    if not _confirm(args, overrides, "Register BMC credentials now?",
                    default=False):
        return
    for label, vault, kind in (
            ("BMC read-only password", BMC_PASSWORD_VAULT, "password"),
            ("BMC sudo password", BMC_SUDO_VAULT, "password"),):
        try:
            store.get(vault)
            print(f"bmc: {vault} already registered")
            continue
        except KeyError:
            pass
        if not _confirm(args, overrides, f"Set {label}? (vault {vault})",
                        default=True):
            continue
        if kind == "password":
            value = _ask_secret(args, overrides, f"{label}: ")
            if value:
                store.put(vault, value.encode("utf-8"))
                print(f"bmc: {vault} stored")
        else:
            store.put(vault, ssh_material)


# ---- verification ----

def _check_inventory_secrets(inv_path: Path, store: DirSecretStore) -> list[str]:

    missing: list[str] = []
    try:
        inv = load_inventory(inv_path)
    except Exception as exc:  # noqa: BLE001 - report inventory shape problems
        return [f"inventory: {exc}"]
    refs = []
    if inv.llm is not None and inv.llm.api_key_vault_path:
        refs.append((inv.llm.api_key_vault_path, "llm api key"))
    if inv.console_defaults is not None:
        if inv.console_defaults.identity_vault_path:
            refs.append((inv.console_defaults.identity_vault_path, "console identity"))
        if inv.console_defaults.sudo_vault_path:
            refs.append((inv.console_defaults.sudo_vault_path, "console sudo"))
    for host in inv.hosts or []:
        if host.ssh and host.ssh.identity_vault_path:
            refs.append((host.ssh.identity_vault_path, f"{host.name} ssh"))
        if host.bmc and host.bmc.password_vault_path:
            refs.append((host.bmc.password_vault_path, f"{host.name} bmc"))
    for vault, label in refs:
        try:
            store.get(vault)
        except KeyError:
            missing.append(f"{label}: {vault}")
    return sorted(set(missing))


# ---- entry point ----

def run_setup(args, overrides: dict | None = None) -> int:
    """Run the interactive wizard; returns 0 when setup completed cleanly."""
    overrides = overrides or {}
    store = _store_for(args)
    print("harness setup | secret store:", store._root)
    inv_path = _ensure_inventory(args, overrides)
    inventory_text = inv_path.read_text(encoding="utf-8")
    _setup_llm(args, overrides, inv_path, store, inventory_text)
    public = _setup_ssh(args, overrides, store)
    try:
        ssh_material = store.get(DIAGBOT_SSH_VAULT)
    except KeyError:
        ssh_material = None
    _setup_bmc(args, overrides, store, ssh_material)
    if public:
        print("\ngrant access from the remote machine "
              "(append once, per machine):")
        print(f"  echo \"{public}\" >> ~/.ssh/authorized_keys")
    missing = _check_inventory_secrets(inv_path, store)
    if missing:
        print(f"\n{len(missing)} vault path(s) still unregistered:")
        for line in missing:
            print(f"  - {line}")
        print("re-run `harness setup` when ready, or register via "
              "`harness secrets`")
        return 1
    print("\nsetup complete: every vault path in the inventory resolves.\n"
          "next: harness lint --inventory inventory.yaml, then "
          "harness menu")
    return 0