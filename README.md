# DebugCLI — AI-Assisted Server Debugging

DebugCLI (called "the harness") is a command-line companion that helps you
figure out what is wrong with a server. You point it at a machine, it connects
**read-only**, gathers evidence (sensors, event logs, kernel messages, register
dumps), cross-references your vendor documentation, and an AI model writes a
diagnosis with **suggested** repair steps. A human always performs — and
approves — any actual repair.

> **Absolute rule: the harness performs zero raw hardware writes.**
> No MSR write, no `setpci -w`, no `/dev/mem` write, no firmware flash.
> It only *recommends* — a human (or a separate, independently-audited layer)
> is the only writer.

---

## What it does 

- **Connects without changing anything.** Every command it runs on your server
  is checked against a strict read-only allowlist first — anything that could
  write, flash, or reset is denied by design.
- **Talks to your server the way you would.** Over SSH (by IP address), or
  through the rack manager's serial console by rack and cable number — no
  per-server config files needed. It can also pull read-only evidence straight
  from the rack manager's Redfish API (event log, service conditions) without
  opening a serial session at all.
- **Understands your documentation.** Add your vendor's architecture PDFs and
  the AI cites them (title + page) instead of guessing.
- **Keeps secrets secret.** Passwords and keys are asked for once, stored
  locally, and never shown to the AI model or written into any log.
- **Leaves a paper trail.** Every run is archived in `harness_runs/<id>/` with
  a tamper-evident, hash-chained audit log — any diagnosis can be replayed and
  checked later.
- **Learns from confirmed repairs.** When you record what actually fixed a
  server, future runs that match the same failure pattern surface the verified
  fix before the AI even starts guessing.

## Requirements

- **Python 3.11 or newer** — check with `python --version` (developed on 3.14;
  the examples below are Windows PowerShell, the CLI itself is cross-platform).
- **A terminal** — PowerShell on Windows, or bash on Linux/macOS.
- **A way into the machine you want to debug** — the rack manager's console
  (lab/QA racks) or SSH to the server, plus the relevant password/key. You
  will be prompted when it is needed; nothing has to be set up in advance.
- **Optional extras**: vendor architecture PDFs (better, cited diagnoses), a
  factory/test log, and an AI model — Google Gemini via an API key, or your
  own locally hosted model.

## Install

```powershell
python -m pip install -e ".[test,docs]"
```

This installs the `harness` command plus the optional extras used for document
indexing (RAG) and testing. Prefer `python -m pip` over a bare `pip`: with
multiple Python installations, bare `pip` can be wired to a different
interpreter than the `python` you run, which puts the command where you will
not find it.

### Windows: `harness` is not found after install

The command ships as `harness.exe` inside the Python environment's
`Scripts\` directory. It is only discoverable while that directory is on
`PATH` — and editing `PATH` never affects terminals that are already open,
so open a new one after any change.

Find where pip put it:

```powershell
python -m pip show -f harness   # "Location:" plus a ..\..\Scripts\harness.exe entry
python -c "import sys, sysconfig; print(sysconfig.get_path('scripts'))"
where.exe harness               # no output means it is not on PATH
```

A pip warning during install ("The script harness.exe is installed in
'...' which is not on PATH") confirms the case. Common causes and fixes:

- **User-site fallback / Microsoft Store Python** — scripts land under
  `%APPDATA%\Python\Python3XX\Scripts\` or
  `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation...\LocalCache\local-packages\...\Scripts\`,
  neither of which is on `PATH`. Add the reported directory to your user
  `PATH`, then open a new terminal:

  ```powershell
  $scripts = "<the Scripts directory reported above>"
  $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
  [Environment]::SetEnvironmentVariable('Path', "$userPath;$scripts", 'User')
  ```

- **Installed into a venv** — activate it before running the command:
  `.\.venv\Scripts\Activate.ps1`.

- **Multiple Python installations** — reinstall with the interpreter you
  actually invoke: `python -m pip install -e ".[test,docs]"`.

Sanity check: `where.exe harness` prints one path, and you start `harness`
from the repository root so it auto-discovers `inventory.yaml`, `config/`,
and `secrets/`.

## Your first diagnosis (no configuration files required)

1. Start the menu:

   ```powershell
   harness
   ```

   First run with no inventory yet? It auto-launches the **setup wizard**,
   which creates one and walks you through the essentials. Re-run it any time
   with `harness setup`.

2. Choose **Debug a target** → **Rack + cable** (a lab rack behind a rack
   manager) and enter the rack id and cable number, e.g. `Q67` and `8`. Or
   pick **SSH address** for a machine you reach over the network.

3. Describe the symptom in plain English when the debug REPL opens, e.g.
   `uncorrectable ECC errors` — the agent runs the read-only diagnosis in the
   background and you can keep chatting: ask follow-up questions, request
   re-checks, or queue a factory/test log with `/testlog <path>` while it
   works. Debugging is iterative, not one-shot.

4. If a password or key is needed, the harness asks at that exact moment and
   stores it locally (`secrets/`). It is redacted from every log and never
   sent to the AI model.

5. Wait a few minutes while it connects, collects read-only evidence (sensors,
   SEL, kernel log, register dumps, docs lookups), and reasons. You get a
   plain verdict — **healthy / degraded / fault** — the evidence behind it,
   and safe, prioritized **suggested actions**. Nothing is executed on the
   server.

6. Every run is archived under `harness_runs/<id>/`; the menu's **Inspect past
   runs** re-opens any of them (verdicts, commands, evidence).

## Setting the model

The AI model does the reasoning; the harness does everything else. Pick one in
the menu's **model** picker (arrow keys, type to filter) or with `/model`
inside a chat session — the choice is remembered in `config/models.yaml`
(machine-local, git-ignored) so later runs need no flags.

| Provider | Good for | Notes |
|---|---|---|
| `gemini` | Quick start, no hardware needed | Needs `GEMINI_API_KEY` (or a vault-stored key); the wizard asks once |
| `openai` | Any OpenAI-compatible gateway | Point it at the gateway URL |
| `local` | Fully on-prem (vLLM, llama.cpp, Ollama `/v1`) | Nothing leaves your network; see below |
| `stub` | Testing the pipeline without an AI | Runs end-to-end, no reasoning |

### Connecting a locally hosted model (e.g. vLLM on the lab server)

Selecting `+ add / configure a model` in the picker starts a short wizard:

1. **Provider** → `local`.
2. **Endpoint URL** (Enter = `http://127.0.0.1:8000/v1`), then **direct** or
   **tunnel** transport. Tunnel = the model runs on the golden server (the
   rack/cable debug target) behind the rack-manager hop: the wizard probes the
   node with read-only commands (`hostname -I`, `ss`, `docker ps`) and you
   pick the `HOST:PORT` from the discovered candidates.
3. It then lists the **served model ids** live from the server — pick one
   (vLLM rejects any other name). The profile is saved; every later run and
   session uses it with zero flags.

Standalone equivalents, if you prefer commands:

```powershell
# Find the model endpoint on the target node (read-only console probes)
harness llm discover --inventory <file> --rack Q61 --cable 8

# Verify the whole path (stages: ssh -> forward -> /models -> chat)
harness llm check --tunnel 10.0.0.42:8000 --inventory <file>

# One-shot diagnosis against it
harness diagnose --inventory <file> --rack Q61 --cable 8 `
  --llm local --llm-model local/Qwen2.5-7B-Instruct --llm-tunnel 10.0.0.42:8000
```

If the rack manager refuses forwarding, both the wizard and `llm check` print
the reverse-tunnel relay recipe and can save the relay URL instead.

**Timeouts:** a self-hosted model often needs minutes for one diagnosis (long
prompt, cold start). Local endpoints therefore get a **10-minute** budget by
default, cloud endpoints 2 minutes. Raise it with `HARNESS_LLM_TIMEOUT`
(seconds) or a `timeout:` value on the model profile in `config/models.yaml`;
a timeout error names both knobs. A first attempt that times out has usually
warmed the server — simply re-run.

## Connecting to servers

The inventory file holds shared defaults (rack manager address, accounts as
vault **paths**, trust level); individual machines need no YAML:

```powershell
# Console by rack/cable (lab/QA; needs a console_defaults: block)
harness diagnose --inventory inventory.yaml --rack Q61 --cable 8 --symptom "ECC errors"

# SSH by IP
harness diagnose --inventory inventory.yaml --address 10.0.0.50 --symptom "ECC errors"

# Interactive debug session against the same targets
harness debug --inventory inventory.yaml --rack Q61 --cable 8

# Target-less chat: manuals, past runs, or a referenced log file
harness chat
```

Resolution order: `--host` > `--target <alias>` > `--rack/--cable` > `--address`.

### Optional: rack-level Redfish evidence

With a Redfish password configured, each console diagnosis **also** fetches
two read-only evidence sets straight from the rack manager's HTTPS API — the
node event log and its service conditions — even when the serial console is
the broken part:

```yaml
console_defaults:
  # ... existing console fields ...
  redfish_user: root
  redfish_password_vault_path: secret/harness/rackmgr/redfish
```

```powershell
harness secrets set-password secret/harness/rackmgr/redfish
```

GET-only by construction (the client cannot express a write), with automatic
transport fallback (direct HTTPS, else the rack-manager SSH hop). Without the
vault path, nothing changes. Full details in the
[User Guide](docs/USER_GUIDE.md).

## Feed it a factory/test log (optional)

Hand the pipeline a FAT/harness log and its failures are parsed into evidence
for every prompt:

```powershell
harness diagnose --inventory inventory.yaml --host h1 --test-log quanta_qmf_fat_*.log
#  (--symptom is optional here; it derives from the log's first failure)
```

Works in debug sessions too (`/testlog <path>`). In `harness chat` you can
simply paste a log file's path — the agent reads it and matches its failures
against prior verified fixes. The raw log is archived with
the run (passwords in it are redacted before anything reaches a prompt), and
its failure codes feed the learning loop below.

## After a repair: teach it (optional)

Record what actually fixed the server and the harness turns it into ground
truth — future runs whose evidence matches surface the verified fix as a
**Prior Verified Case**, pre-probe:

```powershell
harness label --run <run-id>          # interactive: fix + outcome
# or non-interactive:
harness report --run <run-id> --outcome fixed --taken "Reseat DIMM A1"
harness priors update                  # rebuild subsystem priors from outcomes
harness calibrate                      # rebuild per-model fix-rate calibration
```

## Safety and privacy, in short

- **Read-only by construction** — every command passes a single runner choke
  point validated against an exact argv-template allowlist (`force_read_only`);
  write tools are denied even if asked for by name.
- **Credentials never reach the AI** — registered via `harness secrets` or
  prompted on demand, stored locally, vault paths only; `harness lint` rejects
  inline secrets in inventories.
- **Tamper-evident audit trail** — append-only, hash-chained `audit.jsonl`
  per run with secret redaction.
- **Trust gate** — console and Redfish paths refuse to run at `trust_level:
  prod` (lab/QA only).

## Command reference

| Command | Purpose |
|---|---|
| `harness` / `harness menu` | Interactive menu: debug a target, inspect past runs, pick the model |
| `harness setup` | Guided first-run wizard: inventory + credentials + LLM access |
| `harness debug` (alias `session`) | Interactive debug REPL on a target; background runs, slash commands (`/model`, `/testlog`, `/quit`) |
| `harness chat` | Target-less chat REPL: manuals, past runs, referenced files — no machine is contacted |
| `harness diagnose` | One-shot read-only diagnosis of a target (CLI/automation) |
| `harness console` | Run read-only probes over the serial console (lab/QA only) |
| `harness verify --baseline <dumps.json>` | Re-run collectors, compare error counters |
| `harness llm check` | Staged AI-endpoint test: ssh → forward → /models → chat |
| `harness llm discover` | Find the model endpoint on a rack/cable node (read-only probes) |
| `harness llm pin-host` | Pin a per-rack manager's host key through the bastion |
| `harness docs add \| ls \| rm \| reindex \| retag` | Manage the RAG document library |
| `harness lint --inventory <file>` | Validate inventory (rejects inline secrets) |
| `harness secrets add-ssh \| set-password \| list \| rm \| check` | Register credentials (never via the AI) |
| `harness targets add \| ls \| rm` | Short aliases for repeated targets |
| `harness label --run <id>` / `harness report --run <id>` | Record a run's verified outcome (learning loop) |
| `harness calibrate` / `harness priors update` | Rebuild calibration / subsystem priors |
| `harness eval` | Offline regression replay against the case baseline |

Common `diagnose`/`debug`/`chat` flags: `--inventory` (diagnose/debug),
`--secret-dir`, `--docs-lib`, `--docs-dir`, `--parts-csv`, `--out-dir`,
`--llm {openai,gemini,local,stub}`, `--llm-model <ident>`,
`--llm-url`, `--llm-tunnel HOST:PORT`, `--approval` / `--approve-all`
(diagnose), `--max-tools` (chat/debug: tool calls per message), `--test-log`
(diagnose/debug).

## Troubleshooting

| What you see | What it means / what to do |
|---|---|
| `LLM response timed out after Ns ...` | The model is still generating (or cold). Re-run — a warm server is faster — or raise the budget: `HARNESS_LLM_TIMEOUT` (seconds) or `timeout:` on the profile in `config/models.yaml`. `harness llm check` confirms the path is alive. |
| `LLM HTTP 400: 'No user query found in messages.'` | The server refuses system-only requests. Current harness versions guarantee a user turn on every call — update and re-run. |
| `LLM unreachable` / forward refused | Run `harness llm check --tunnel HOST:PORT --inventory <file>` to see which leg fails (ssh / forward / http / chat). If the rack manager refuses forwarding, the check prints the reverse-tunnel relay recipe. |
| `warning: requested model ... is not in the served list` | vLLM only accepts the exact served name — pick the id the wizard/`llm check` lists. |
| `SerialConsoleError: allowed only at lab/qa` | Console targeting is blocked at `trust_level: prod` — set `lab`/`qa`. |
| `Host key not in pinned known_hosts` | Pin the rack manager's key: `harness llm pin-host --rack R --cable N` (LLM hop) or re-run `harness setup`. |
| `sudo password missing from vault` | Register it (`harness secrets set-password <vault-path>`) or answer the on-demand prompt (needs an interactive terminal). |
| `SerialProbeDenied: ... i2cset ...` | That is a write tool — only reads are allowed (by design). |

## Documentation

| Document | Contents |
|---|---|
| [README.md](README.md) | This page |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | End-to-end walkthrough: setup wizard, menu, chat, console, Redfish, local models |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module layout, data flow, pipeline stages |
| [docs/SECURITY.md](docs/SECURITY.md) | Security invariants, threat model, deployment notes |
| [docs/MODEL_SWAP.md](docs/MODEL_SWAP.md) | Runbook for swapping the AI model safely |
| [TESTING.md](TESTING.md) | Unit tests + live console test against rack Q71 cable 8 |
| [REVISED_SPEC.md](REVISED_SPEC.md) | Full design spec and risk mitigations (rev 2.0) |

## Development

```powershell
python -m pip install -e ".[test,docs]"
pytest            # 750+ unit tests, no hardware needed
ruff check .
```

On Windows, if the `harness` command is not found after installing, see
[Windows: `harness` is not found after install](#windows-harness-is-not-found-after-install).

## Repository hygiene

- `keys/`, `secrets/`, `harness_runs/`, vendor PDFs, and `known_hosts` files
  are **git-ignored** — never commit credentials or run artifacts.
- `config/*.yaml` (including the remembered model pick, `config/models.yaml`)
  is machine-local and git-ignored.
- The sample inventory in `src/harness/config/inventory.yaml` contains vault
  **paths only**; `harness lint` enforces this invariant.
- Add `docs/*.pdf` locally (they are architecture reference material); the RAG
  library (`harness docs add`) builds its index on your machine.

## License

Proprietary. All rights reserved.
