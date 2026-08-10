# Agent Task: Zero-YAML Server Targeting + Secure Credential CLI

## Role
You are implementing a feature in the `harness` repo (Python 3.11+, paramiko SSH
debugging companion) at `C:\Users\Phillip.Peng\Documents\Default Project`.
You will WRITE CODE. Before designing anything, read `README.md`,
`REVISED_SPEC.md`, and the modules under `src/harness/config/`,
`src/harness/operator/`, `src/harness/engine/`.

## Goal
Operators should no longer hand-edit a YAML inventory entry for every server.
Two addressing modes must work with zero per-server YAML:

1. Natural language location: "Diagnose the server in Q61 Cable 8"
   -> resolve via rack-manager console + serial cable (console/SOL path).
2. Direct IP: "Diagnose 10.0.0.10" -> resolve via SSH (OS path).

Because SSH keys and passwords MUST NEVER be passed through the agent prompt
(they would leak into LLM context, transcripts, and audit logs), you will also
add non-agent CLI tooling to securely register credentials into the secrets
backend (the vault store), so prompts only ever carry identifiers and vault paths.

## Current state (verify by reading, don't trust this summary)

- `src/harness/config/inventory.yaml` is the only inventory: one `hosts:` list
  where each host bundles ssh domain, bmc domain, and a `console:` block
  (rack manager address/user/identity/known_hosts + `rack`, `cable`, `port`,
  `sudo_vault_path`). `src/harness/config/models.py` defines `Host`,
  `SSHDomain`, `BMCDomain`, `ConsoleDomain`, `Inventory` (frozen dataclasses).
- `src/harness/config/inventory_lint.py` enforces: inventory holds VAULT PATHS
  only; any inline credential is rejected. This invariant must keep holding.
- `src/harness/config/vault.py`: `SecretStore` ABC with `get`/`put`;
  `MemorySecretStore` (tests), `DirSecretStore` (file-backed lab store),
  `load_key_material()` materializes keys to temp files, mode 0o600.
- `src/harness/operator/cli.py` subcommands: `lint`, `docs`, `diagnose`,
  `console`, `verify`, `session`. Per-launch console overrides already exist:
  `--console-address`, `--rack`, `--cable`, `--port`, `--sudo-vault-path`
  (see `_console_overrides`, `validate_identifier` in `engine/sol.py`).
- `src/harness/operator/router.py`: LLM intent router returning `SessionCommand`
  (intents: diagnose/probe/docs/verify/status/reply; `host` field validated
  against known inventory host names; unknown host silently falls back to the
  active host).
- `src/harness/operator/repl.py`: interactive `harness session` REPL; slash
  commands `/help`, `/hosts`, `/use <host>`, etc.
- Security invariants (REVISED_SPEC.md): read-only only, exact argv allowlist
  in `engine.runner`/`engine.allowlist`, console path gated to trust
  lab/qa (never prod), append-only audit log with secret redaction.
- Dev env is Windows/PowerShell; the CLI must be cross-platform (no `chmod`
  reliance beyond what `os.chmod` supports, getpass for password input).
- Checks: `pytest` (tests under `tests/`), `ruff` (line-length 100).

## Requirements

### 1. Dynamic target resolution (no per-server YAML)
Design a `Target`/resolution layer that maps a runtime spec to a connection
plan WITHOUT requiring a per-server inventory entry:
- Extend the inventory with a fleet-level `console_defaults:` block (rack
  manager address, user, identity vault path, known_hosts, prompts, port,
  sudo vault path, trust level) so the rack manager is configured ONCE and
  per-target rack/cable are runtime parameters.
- A target spec is one of: `(rack, cable)` -> console session on that rack
  manager; `(ip)` -> SSH session to that address (identity + known_hosts
  resolved from the store); optionally both (IP gives the OS path when the
  console can't reach it).
- Named hosts already in inventory keep working unchanged (backward compat).
- Implement `resolve_target(spec, inv, store) -> Target` where Target exposes
  enough for the existing engine: session open, collector run, close.

### 2. Two addressing modes end to end
- `harness diagnose --rack Q61 --cable 8 "symptom"` (console path) and
  `harness diagnose --address 10.0.0.10 "symptom"` (SSH path) must run the
  full read-only pipeline (plan -> collect -> decode -> RAG -> LLM -> score
  -> approval -> audit) with the SAME output artifacts as today.
- In `harness session`, the first message (or any message) may name a target:
  "Diagnose the server in Q61 Cable 8" or "Diagnose 10.0.0.10". Extend
  `SessionCommand`/router with target fields (rack, cable, ip) and regex
  fallback for `(rack id like Q\d+|cable \d+)` and IPv4; the router must set
  the active target before the diagnose run. `/use` should accept rack/cable
  or IP too.

### 3. Credentials: never via the agent prompt (decision: NOT SAFE)
Answer embedded for you: no, credentials must not go through the prompt.
Prompt text is sent to an external LLM and is persisted in
`transcript.json`/`audit.jsonl`. Design the flow so the prompt contains only
identifiers + vault paths; the harness resolves secrets locally via
`SecretStore`.

### 4. New non-agent CLI: `harness secrets` (and optional `harness targets`)
- `harness secrets add-ssh <name> --key-file <path> [--vault-path ...]`
  stores a private key into the configured store (DirSecretStore behind
  `--secret-dir`, or real Vault backend if present), enforces 0o600, and
  NEVER accepts key material as a literal argument (only a file path or
  interactive input, to avoid shell history/process-list leaks).
- `harness secrets set-password <name> [--vault-path ...]` reads the secret
  interactively (getpass), never echoes, never logs it.
- `harness secrets list`, `harness secrets rm <name>`.
- `harness secrets check <name>` verifies a vault path resolves.
- Registration events append to the audit log (redacted).
- Optional: `harness targets add <alias> --rack Q61 --cable 8 [--address ...]`
  so future prompts can use a short alias; stored as a plain (path-only) file
  that passes `inventory_lint` semantics.

## Hard constraints (non-negotiable)
- ZERO raw hardware writes, ever. Reuse `engine.runner` allowlist; don't add
  new shelling-out paths.
- Inventory/lint invariant: no inline secrets anywhere; secrets CLI stores
  only into the vault store, never into YAML.
- Console path stays trust-gated (lab/qa only, never prod).
- Existing tests must pass; add tests for the new resolver, router fallback
  regexes, and secrets CLI (unit + one integration test using a tmp
  `DirSecretStore`).

## Deliverables
1. New/changed modules (resolution layer, router/repl changes, secrets CLI,
   inventory model additions) with tests in `tests/`.
2. `README.md` usage examples for the two prompt styles and the secrets CLI.
3. Everything passing: `pip install -e ".[test]"` then `pytest` and `ruff`.

## Acceptance criteria
- `harness diagnose --rack Q61 --cable 8 "<symptom>"` connects over the rack
  manager console and produces a diagnosis using only the existing
  inventory.yaml (no per-host YAML added).
- `harness diagnose --address <ip> "<symptom>"` works via SSH when the vault
  holds the key and known_hosts.
- In a session, typing exactly "Diagnose the server in Q61 Cable 8" routes to
  a diagnose run against that target (verified with the stub LLM).
- No secret string ever appears in transcript.json, audit.jsonl, or the LLM
  prompt (test asserts redaction).
- Named-host behavior from the current inventory.yaml is unchanged.

## Report back
Summarize: files changed, the Target resolution design, how the router
handles the two addressing modes, the exact `harness secrets` command set,
and how you verified security invariants + tests.
