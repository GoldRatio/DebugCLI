"""Collector registry: name -> Collector class. Used by the DiagnosticEngine factory."""

from __future__ import annotations

from .collectors import Collector
from .collectors.cpu_msr import CpuMsrCollector
from .collectors.ipmi import IpmiCollector
from .collectors.kernel import KernelCollector
from .collectors.pcie import PcieCollector
from .collectors.storage import StorageCollector

_COLLECTORS: dict[str, type[Collector]] = {
    "cpu_msr": CpuMsrCollector,
    "pcie": PcieCollector,
    "ipmi": IpmiCollector,
    "kernel": KernelCollector,
    "storage": StorageCollector,
}


def make_collector(name: str, runner) -> Collector:
    cls = _COLLECTORS.get(name)
    if cls is None:
        raise KeyError(f"unknown collector: {name!r}")
    return cls(runner)


def collector_names() -> list[str]:
    return list(_COLLECTORS)