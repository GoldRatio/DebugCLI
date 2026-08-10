"""Secrets management interface.

Harness never stores credentials. ``load`` resolves a vault path to bytes/material
from whatever backend is configured. A no-op memory backend is provided for tests;
a real deployment plugs in HashiCorp Vault / AWS Secrets Manager here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class SecretStore(ABC):
    @abstractmethod
    def get(self, vault_path: str) -> bytes: ...
    @abstractmethod
    def put(self, vault_path: str, value: bytes) -> None: ...
    @abstractmethod
    def keys(self) -> list[str]: ...
    @abstractmethod
    def delete(self, vault_path: str) -> None: ...


class MemorySecretStore(SecretStore):
    """In-memory store (tests only). Never persists."""

    def __init__(self, seed: dict[str, bytes] | None = None) -> None:
        self._store: dict[str, bytes] = dict(seed or {})

    def get(self, vault_path: str) -> bytes:
        try:
            return self._store[vault_path]
        except KeyError:
            raise KeyError(f"secret not found: {vault_path!r}")

    def put(self, vault_path: str, value: bytes) -> None:
        self._store[vault_path] = value

    def keys(self) -> list[str]:
        return list(self._store)

    def delete(self, vault_path: str) -> None:
        try:
            del self._store[vault_path]
        except KeyError:
            raise KeyError(f"secret not found: {vault_path!r}") from None


class DirSecretStore(SecretStore):
    """File-backed store for lab use: each vault path maps to a file under ``root``.

    Path traversal is rejected (a vault path may never escape the store root).
    The inventory still holds paths only -- the files themselves are the secrets.
    Files are written with mode 0o600.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _resolve(self, vault_path: str) -> Path:
        candidate = (self._root / vault_path).resolve()
        if not str(candidate).startswith(str(self._root.resolve())):
            raise KeyError(f"secret path escapes store root: {vault_path!r}")
        return candidate

    def get(self, vault_path: str) -> bytes:
        path = self._resolve(vault_path)
        if not path.is_file():
            raise KeyError(f"secret not found: {vault_path!r}")
        return path.read_bytes()

    def put(self, vault_path: str, value: bytes) -> None:
        path = self._resolve(vault_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        try:
            path.chmod(0o600)
        except OSError:  # pragma: no cover - Windows best-effort
            pass

    def keys(self) -> list[str]:
        root = self._root.resolve()
        out = []
        for p in sorted(root.rglob("*")):
            if p.is_file():
                out.append(str(p.relative_to(root)).replace("\\", "/"))
        return out

    def delete(self, vault_path: str) -> None:
        path = self._resolve(vault_path)
        if not path.is_file():
            raise KeyError(f"secret not found: {vault_path!r}")
        path.unlink()


def load_key_material(store: SecretStore, vault_path: str, tmp_dir: Path) -> Path:
    """Materialize a private key from the store onto disk read-only, for paramiko.

    The key is written with mode 0o600 and a unique filename to avoid colliding
    with other sessions. Callers should delete the file after the session closes.
    """
    material = store.get(vault_path)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    key_path = tmp_dir / f"diagbot_key_{vault_path.replace('/', '_')}.pem"
    key_path.write_bytes(material)
    key_path.chmod(0o600)
    return key_path