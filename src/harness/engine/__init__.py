"""engine: secure SSH/exec with strict read-only enforcement.

``runner`` is the SINGLE choke point through which all inspection commands flow.
Every argv is validated by the allowlist and must be read-only. There is no other
path that can execute a command against the target.
"""