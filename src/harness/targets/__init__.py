"""Dynamic target resolution: map a runtime spec (rack/cable, IP, alias, or
named host) to a connection plan WITHOUT per-server YAML entries."""

from .resolver import Target, TargetError, TargetSpec, resolve_target

__all__ = ["Target", "TargetError", "TargetSpec", "resolve_target"]
