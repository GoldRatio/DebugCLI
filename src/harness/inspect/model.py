"""Model detection via dmidecode, mapping to a collector profile.

The harness determines the server model DETERMINISTICALLY (dmidecode on the host
OS, FRU over a BMC serial console) and records how it learned it
(``DetectedModel.source``). The LLM never derives the model: it is presented to
the prompt as a fact with provenance. When detection fails, the engine may fall
back to a target-alias hint or a single optional operator question
(``from_alias`` / ``from_operator``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..engine.runner import Runner

# subsystem -> [Collector class names] run for diagnostics per subsystem.
PROFILE_COLLECTORS = {
    "memory": ["cpu_msr", "kernel"],
    "cpu": ["cpu_msr", "kernel"],
    "pcie": ["pcie", "kernel"],
    "bmc": ["ipmi", "kernel"],
    "storage": ["storage", "kernel"],
    "generic": ["cpu_msr", "pcie", "ipmi", "kernel", "storage"],
}

# canonical model key -> known product-name variants (spellings, short names).
# Keep it small and reviewed; this map decides case-library and doc-tag
# matching, so entries are a manual, deliberate act.
MODEL_ALIASES: dict[str, list[str]] = {
    "poweredge_r650": [
        "poweredge r650", "r650", "r650xe", "r650 xeon",
        "dell poweredge r650", "poweredge r650xe",
    ],
    "poweredge_r750": [
        "poweredge r750", "r750", "poweredge r750xa", "r750xa",
    ],
    "proliant_dl380g10": [
        "proliant dl380 gen10", "proliant dl380g10", "dl380 gen10", "dl380g10",
    ],
}

# Vendor/registration noise that never belongs in a model key.
_NON_MODEL_TOKENS = ("system", "server", "platform", "product name", "special")

_MODEL_NORMALIZE_RE = re.compile(r"\s+")

# ipmitool fru print pads the label before the colon: "Product Name          : C4A15".
_FRU_LINE = re.compile(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$")


def normalize_product(raw: str) -> str | None:
    """Normalize a raw product string to a canonical model key or a clean slug.

    Strips vendor/noise tokens and parentheses, resolves known variants through
    ``MODEL_ALIASES``, and returns None only for empty input. Unknown models
    pass through as a clean lowercase slug -- never None, so detection results
    stay comparable.
    """
    text = raw.strip().lower()
    text = re.sub(r"\(.*?\)", " ", text)
    for token in _NON_MODEL_TOKENS:
        text = text.replace(token, " ")
    text = _MODEL_NORMALIZE_RE.sub(" ", text).strip()
    if not text:
        return None
    preferred = _MODEL_NORMALIZE_RE.sub("_", text)
    for canonical, variants in MODEL_ALIASES.items():
        if text == canonical or text in variants or preferred == canonical:
            return canonical
    return preferred


@dataclass(frozen=True)
class DetectedModel:
    product_name: str
    bios_vendor: str
    bios_version: str | None
    raw: str
    # "dmidecode" | "fru" | "operator" | "alias" | "unknown"
    source: str = "unknown"

    @property
    def model_key(self) -> str:
        key = normalize_product(self.product_name)
        return key or self.product_name.replace(" ", "_").lower()


def from_operator(product_name: str) -> DetectedModel:
    """Model provided by the operator at run time (unverified, lower trust)."""
    return DetectedModel(
        product_name=product_name,
        bios_vendor="operator",
        bios_version=None,
        raw=f"operator provided: {product_name}",
        source="operator",
    )


def from_alias(key: str) -> DetectedModel:
    """Model cached on a target alias (previously detected, driftatable)."""
    return DetectedModel(
        product_name=key,
        bios_vendor="alias",
        bios_version=None,
        raw=f"target alias model: {key}",
        source="alias",
    )


def detect_model(runner: Runner) -> DetectedModel | None:
    """Detect the chassis/product from dmidecode (host OS) or FRU (BMC console).

    On a BMC serial console there is no ``dmidecode``; ``ipmitool fru print``
    carries the same product/vendor fields read-only.
    """
    if getattr(runner, "is_console", False):
        result = runner.execute(["sudo -S ipmitool fru print"])
        if not result.ok:
            return None
        product = _fru_field(result.stdout, "Product Name") \
            or _fru_field(result.stdout, "Board Product")
        if not product:
            return None
        vendor = (_fru_field(result.stdout, "Product Manufacturer")
                  or _fru_field(result.stdout, "Board Mfg"))
        return DetectedModel(
            product_name=product, bios_vendor=vendor or "unknown",
            bios_version=None, raw=result.stdout, source="fru",
        )
    result = runner.execute(["/bin/dmidecode"])
    if not result.ok:
        return None
    product = _field(result.stdout, "Product Name")
    vendor = _field(result.stdout, "BIOS Vendor") or "unknown"
    bios = _field(result.stdout, "BIOS Version")
    return DetectedModel(
        product_name=product, bios_vendor=vendor, bios_version=bios,
        raw=result.stdout, source="dmidecode",
    )


def _field(text: str, label: str) -> str | None:
    for line in text.splitlines():
        if line.strip().startswith(label + ":"):
            return line.split(":", 1)[1].strip()
    return None


def _fru_field(text: str, label: str) -> str | None:
    """First FRU area value for ``label``, skipping the "N/A" placeholders."""
    for line in text.splitlines():
        m = _FRU_LINE.match(line)
        if m is None or m.group(1) != label:
            continue
        value = m.group(2).strip()
        if value and value.upper() != "N/A":
            return value
    return None