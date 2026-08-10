"""Model detection via dmidecode, mapping to a collector profile."""

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


@dataclass(frozen=True)
class DetectedModel:
    product_name: str
    bios_vendor: str
    bios_version: str | None
    raw: str

    @property
    def model_key(self) -> str:
        return self.product_name.replace(" ", "_").lower()


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
            bios_version=None, raw=result.stdout,
        )
    result = runner.execute(["/bin/dmidecode"])
    if not result.ok:
        return None
    product = _field(result.stdout, "Product Name")
    vendor = _field(result.stdout, "BIOS Vendor") or "unknown"
    bios = _field(result.stdout, "BIOS Version")
    return DetectedModel(product_name=product, bios_vendor=vendor, bios_version=bios, raw=result.stdout)


def _field(text: str, label: str) -> str | None:
    for line in text.splitlines():
        if line.strip().startswith(label + ":"):
            return line.split(":", 1)[1].strip()
    return None


# ipmitool fru print pads the label before the colon: "Product Name          : C4A15".
_FRU_LINE = re.compile(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$")


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