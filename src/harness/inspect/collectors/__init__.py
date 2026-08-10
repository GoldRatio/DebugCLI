"""Collector interface.

A collector probes one subsystem (CPU MSRs, PCIe, IPMI sensors, storage, kernel
log) and returns a set of raw ``RegisterDump`` objects. Collectors may hold a
reference to the read-only Runner; they never bypass it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...engine.runner import Runner
from ..base import RegisterDump


class Collector(ABC):
    subsystem: str = "generic"

    def __init__(self, runner: Runner) -> None:
        self.runner = runner

    @abstractmethod
    def collect(self, **kwargs) -> list[RegisterDump]: ...