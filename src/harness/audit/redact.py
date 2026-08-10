"""Secret redaction.

All traces are scrubbed before storage so credentials, host key material, and key
fingerprints never land on disk/SIEM. Redaction is applied to every event string at
write time in ``AuditLog``.
"""

from __future__ import annotations

import re


class Redactor:
    def __init__(self, secrets: list[str] | None = None) -> None:
        self._patterns = [re.compile(re.escape(s), re.IGNORECASE) for s in (secrets or [])]

    def add_secret(self, secret: str) -> None:
        self._patterns.append(re.compile(re.escape(secret), re.IGNORECASE))

    def redact(self, text: str) -> str:
        out = text
        for pat in self._patterns:
            out = pat.sub("[REDACTED]", out)
        return out

    @staticmethod
    def scrub_ssh_keys(text: str) -> str:
        """Replace private key blocks so they cannot be written to a trace."""
        block = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL)
        return block.sub("[REDACTED_KEY]", text)