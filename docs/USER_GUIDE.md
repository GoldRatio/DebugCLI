# User Guide

End-to-end walkthrough for operating the harness: install, credentials,
targeting, diagnosing, verifying, and reading the outputs.

## 1. Install and prepare

```powershell
pip install -e ".[test,docs]"
```

The very first `harness` run finds no inventory and auto-launches
`harness setup` (see §2b) to create one; or create your inventory by hand
(vault paths only). The packaged sample is `src/harness/config/inventory.yaml`;
copy it and adjust:

```powershell
Copy-Item src\harness\config\inventory.yaml .\inventory.yaml
harness lint --inventory inventory.yaml     # must print OK, no inline secrets
```

Required inventory blocks:

- `trust_level:` — `lab | qa | prod`; gates the console path.
- `llm:` — provider (`openai`/`gemini`), model, `api_key_vault_path` (optional;
  use `--llm stub` to run the pipeline without an LLM).
- `hosts:` — named hosts with `ssh` (OS) and `bmc` (IPMI) domains, separate
  rotated credentials.
- `console_defaults:` — fleet rack-manager settings so any rack/cable can be
  targeted without per-host YAML: `address`, `user`, `identity_vault_path`,
  `known_hosts_path`, `tool`, `trust_level`, `prompts`, `port`,
  `sudo_vault_path`.

## 2. Register credentials (once)

```powershell
# SSH private key — from a FILE path, never a literal
harness secrets add-ssh 10.0.0.50 --key-file .\secrets\id_ed25519 --secret-dir .\secrets

# Password — read interactively, never echoed
harness secrets set-password bmc-ro --secret-dir .\secrets

harness secrets list --secret-dir .\secrets
harness secrets check 10.0.0.50 --secret-dir .\secrets
```

SSH-by-IP identity lookup order: `--identity-vault-path` >
`secret/harness/ssh/<ip>` > `secret/harness/diagbot/id_ed25519`.

## 2b. One-time setup wizard (easier onboarding)

On a fresh machine (e.g. a new laptop), instead of hand-writing
`inventory.yaml` and registering vaults manually:

```powershell
harness setup            # interactive; defaults to .\inventory.yaml and .\secrets
harness setup --inventory .\config\inventory.yaml --secret-dir .\secrets
```

What it does, in order:

1. **Inventory** — reuses an existing `inventory.yaml` (patches only the `llm:`
   block, host/BMC sections are left untouched) or creates a minimal one with a
   `trust_level` prompt.
2. **LLM API key** — prompts with `getpass` (never echoed), writes the key to
   `secret/harness/llm/gemini-key`, and adds `provider` / `model` /
   `api_key_vault_path` to the inventory `llm:` block. (Choose `stub` when no
   credentials are available yet.)
3. **SSH identity** — generates a fresh ed25519 pair with `ssh-keygen` if no
   key exists, otherwise lets you register an existing one; it then prints the
   `echo "<pub>" >> ~/.ssh/authorized_keys` command to run once from your
   DIAG remote account. The private key lives under `--secret-dir` (0o600,
   git-ignored) at `secret/harness/diagbot/id_ed25519` — it is shared by all
   hosts; per-host keys still win (see lookup order above).
4. **Rack manager console** (optional) — prompts for the rack-manager console
   IP; answering writes a `console_defaults:` block into the inventory so
   `--rack/--cable` targeting works immediately after setup. The console
   identity reuses the diagbot key (same lab keypair), so no extra vault is
   needed. It then **offers to install that public key onto the rack manager
   now**: one-time password auth through paramiko, the host-key fingerprint is
   shown for verification, and the verified host key is pinned into
   `config/rackmgr_known_hosts` so every later console session connects without
   prompting. Decline (or a failed install) prints the manual
   `echo "<pub>" >> ~/.ssh/authorized_keys` one-liner instead. Leave the IP
   blank to skip and add `console_defaults:` later.
5. **BMC credentials** (optional) — registers the BMC sudo password
   (console-shell escalation for `--rack/--cable`) and the BMC read-only
   password (IPMI over LAN for named hosts), but **only the vaults the
   inventory references** — a fresh `hosts: []` inventory with `console_defaults`
   asks for sudo only, a host inventory with `bmc:` blocks asks for bmc-ro only.
   Sudo is prompted first; when both apply, bmc-ro offers to reuse the same
   password.
6. **Verification** — resolves every vault path referenced by the inventory and
   exits 1 if any is missing, printing which ones to register.

Re-running `harness setup` never overwrites an existing SSH key and skips
vaults that are already registered.

## 2c. Zero-setup start (credentials on demand)

You don't have to register anything before the first run. Launch the interactive
menu / session with an empty (or missing) `--secret-dir` store; the harness asks
you for each credential **the moment it is actually needed**, pauses, stores it
(0o600, git-ignored) and continues:

- **LLM API key** — prompted at the first LLM call (`getpass`, never echoed),
  written to the inventory's `api_key_vault_path`.
- **SSH identity** — prompted at session open. Choose *generate* (fresh ed25519
  via `ssh-keygen`) or *file* (an existing private key). After the key is made,
  it offers to **install the public key onto the target**: one-time password
  auth through paramiko, the host-key fingerprint is shown for you to verify,
  and the pinned `known_hosts` entry is recorded only with your confirmation.
  The install password is used once and never stored.
- **BMC / sudo passwords** — prompted at console/IPMI time (`getpass` with a
  confirmation pass).

Prompts run in the terminal, **never through the agent**: the LLM only ever sees
vault paths and identifiers, so credentials never appear in prompts,
transcripts, or the redacted audit log — during the run or afterwards. Set
`HARNESS_NO_PROMPT=1` (or run with non-tty stdin, e.g. automation/CI) to disable
prompting and keep the plain `KeyError` / `TargetError` messages instead.

## 3. Add architecture PDFs to the RAG library

```powershell
harness docs add .\docs\*.pdf        # chunk + index
harness docs ls
harness docs reindex                 # rebuild index, retries failures
```

Vendor PDFs are git-ignored; they live on your machine only.

## 4. Diagnose

### Interactive menu (no flags)

```powershell
harness            # or: harness menu
```

Auto-discovers the inventory, then menus. The top level keeps the daily
flows only: **Chat session**, **Debug a target** (one-shot diagnosis),
**Inspect past runs**, and **Advanced...** -- which holds verify, serial
console, model, docs, targets, secrets, setup and lint.
Esc cancels; typing filters; non-tty stdin falls back to numbered prompts.

### Chat session

```powershell
harness session --inventory inventory.yaml --host h1
harness> the DIMM error is back on h1
```

- Non-slash text routes (LLM, keyword fallback) to diagnose / probe / docs /
  verify / status / reply. A `probe` request runs the read-only collectors for
  the named subsystem (or doc-named probes mined from the manual) against the
  active target and reports the decoded registers back in the chat -- it never
  asks the agent to invent commands.
- Slash commands: `/help`, `/hosts`, `/use <host|rack cable n|ip|alias>`,
  `/context <note>`, `/status`, `/stop`, `/runs`, `/history`, `/resume <dir>`,
  `/lint`, `/targets ...`, `/docs ...`, `/quit`.
- Type-ahead: messages typed while a run is in progress seed the next run.
- Sessions persist under `--session-dir`; each run keeps `diagnosis.json` /
  `trace.json` / `dumps.json` under `--out-dir`.

### One-shot flags

```powershell
harness diagnose --inventory inventory.yaml --host h1 --symptom "ECC errors"
harness diagnose --inventory inventory.yaml --rack Q61 --cable 8 --symptom "ECC errors"
harness diagnose --inventory inventory.yaml --address 10.0.0.50 --symptom "ECC errors"
harness diagnose --inventory inventory.yaml --target d1 --symptom "ECC errors"
```

Add `--approval` (y/N per action) or `--approve-all` (record all approved),
`--context "note"` / `--context-file`, and `--max-turns` (default 6).

### Test-log evidence (any target)

Before debugging, hand the pipeline the factory/harness run log (e.g. a Quanta
FAT `.log`). Its failures — error codes and test names like
`[FAIL][P02002001@PCIe Test Fail]` / `FAIL: pcie_cmp_chk` — are parsed and
shown to the agent as evidence in every prompt, and seed doc retrieval:

```powershell
harness diagnose --inventory inventory.yaml --host h1 `
    --test-log quanta_qmf_fat_c4a15_p23326287013301e_20260811220125.log
#   --symptom is optional here: it derives from the log's first failure
```

- Repeatable (`--test-log a.log --test-log b.log`), works in single-shot AND
  session mode, and from the menu (the Diagnose step asks for a log path) or
  the chat REPL (`/testlog <path>` queues one for the next run).
- The raw log is copied into `harness_runs/<id>/test_logs/` for
  reproducibility; the audit `test_log_loaded` event records only metadata and
  failure codes. Cleartext passwords in logs (e.g. `sshpass -p '...'`) are
  redacted before anything reaches a prompt, transcript, or audit.
- Learning loop: the run's `pending_case.json` carries the failure signatures.
  After the repair, confirm with `harness report --run <id> --outcome fixed
  [--taken "..."]`. A future run whose log shows the same failure code surfaces
  that verified fix as a Prior Verified Case — even before any probing.

### FAT single tests (GB targets, session mode)

The GB models expose a vendor single-test menu on the host:
`/mnt/smbfs/single_rtp_l10.sh <ServerNumber>`. In session mode the agent can
drive it: it lists the FAT tests, then runs the ones it decides will
discriminate between suspects, and treats the results as evidence.

```powershell
harness diagnose --inventory inventory.yaml --host h1 --symptom "ECC errors" `
    --interactive --server-number 3
harness session --inventory inventory.yaml --host h1 --server-number 3
```

- Requires `--server-number` (positive integer, supplied by you; the lab doc
  notes it can sometimes be derived from `ipmitool fru print 2`), an SSH target
  (not `--console`), and session mode (`--interactive` / `--context*`).
- The agent is instructed to `{"kind":"test","action":"list"}` first, then
  `{"kind":"test","action":"run","test":"<exact label>"}` for tests on the
  listed menu. Menu selections are always bare digits; no free text is ever
  sent to the menu.
- Results come back as `[test result]` messages in the transcript; a `FAIL` is
  treated as strong current-state evidence. The full transcript is saved under
  the run directory as `single_test_transcript.txt`.
- Platform gating: only GB platform families (samoa/nvl72, incl. gb200) are
  allowed; the live-detected model wins.

## 5. Console path (serial/SOL, lab/QA only)

```powershell
harness console --inventory inventory.yaml --rack Q71 --cable 8 --probe "lspci -vvv -n -s 00:06.0"
```

- Two start mechanisms via `console_defaults.tool`:
  - `jumpin` (default): spawns `jumpin q71-1 rm` and waits for the rack-manager
    CLI prompt. Rack id normalizes: `Q71`, `q71`, `71` → `jumpin q71-1 rm`.
  - `direct`: fleet managers that run `start serial session` as a plain shell
    builtin — no spawn, no prompt wait; the start command is sent straight
    into the SSH shell (the exact pattern of a manual `sol()` helper).
- `port` selects the node console service: `2200` = BMC access port (BMC
  BusyBox shell; the collector plan substitutes host-OS probes with BMC
  equivalents (`sudo -S ipmitool sensor list`, `sudo -S ipmitool sel list`,
  `dmesg -r`) and skips pcie/storage probes that need host-OS tools); `22` =
  host SOL (the host Linux root shell where `docker`/`ss`/`hostname` run —
  what LLM endpoint discovery uses).
- Each rack may run its own manager: `console_defaults.rack_addresses`
  (`Q71: 10.0.128.98`, ...) is compared numerically per rack, falling back to
  `address`. Pin each manager's host key in `known_hosts_path`
  (`ssh-keyscan <ip> >> config/rackmgr_known_hosts`) — connections fail
  closed otherwise.

### Rack-level Redfish evidence (no jumpin needed)

The rack manager also serves a per-node Redfish API on its own HTTPS port
(`https://<rack_ip>/<cable>/amc1/redfish/v1`). Configure a vault password and
the debug step collects two read-only evidence sets straight from the rack —
**no serial session, no server jumpin** (and they arrive even when the console
hop is the broken thing):

```yaml
console_defaults:
  # ... console fields above ...
  redfish_user: root
  redfish_password_vault_path: secret/harness/rackmgr/redfish
```

```powershell
harness secrets set-password secret/harness/rackmgr/redfish
```

- Fetched per target: the node **event log** (`EventLog/Entries`, sorted by
  creation time) and its **service conditions** — the same two reads the
  rack-side `get_event_logs` / `get_service_conditions` helpers do.
- GET-only by construction: the client surface cannot express a write, so
  firmware-update/power-control endpoints are unreachable (unlike raw
  `curl`, which happily POSTs).
- Transport auto-falls-back: direct HTTPS to the rack IP first; if it is not
  routable from the workstation, the request rides the rack-manager SSH hop
  (same pinned-host-key primitive as the LLM tunnel). Each dump records which
  transport served it.
- Same trust gate as the console: lab/qa only. The password is vaulted, joins
  the redaction set, and never reaches prompts or the audit log.
- Without the vault path, behavior is unchanged — Redfish collection simply
  does not run.

### LLM-only console (`llm_console:`)

The golden server hosting the vLLM model is often reached differently from
the debug console (different tool, port, or account). The optional
`llm_console:` block — same shape as `console_defaults:` — is used ONLY for
LLM endpoint discovery and the LLM tunnel hop; the debug path above keeps
using `console_defaults:` untouched:

```yaml
console_defaults:               # debug path (unchanged)
  address: 192.168.202.51
  user: log
  identity_vault_path: secret/harness/rackmgr/id_ed25519
  known_hosts_path: config/rackmgr_known_hosts
  tool: jumpin
  trust_level: lab
  port: 2200
  sudo_vault_path: secret/harness/bmc/sudo
llm_console:                    # LLM-only access to the golden server
  address: 192.168.202.51       # fallback
  user: root
  identity_vault_path: secret/harness/rackmgr/id_ed25519
  known_hosts_path: config/rackmgr_known_hosts
  tool: direct                  # start serial session straight in the SSH shell
  trust_level: lab
  port: 22                      # host SOL (host root shell: docker/ss/hostname)
  sudo_vault_path: secret/harness/bmc/sudo
  rack_addresses: {Q71: 10.0.128.98}
```

Absent `llm_console:`, the LLM paths fall back to `console_defaults:`.
- sudo probes trigger an automatic `password for` → send-password handshake
  from `sudo_vault_path`.

## 6. Verify a repair

```powershell
harness verify --inventory inventory.yaml --host h1 --symptom "ECC errors" `
  --baseline .\harness_runs\<run-id>\dumps.json --metric ecc
```

Re-runs collectors and reports whether the error counter moved relative to the
baseline run — your before/after check for a completed repair.

## 7. Target aliases

```powershell
harness targets --targets-file config\targets.yaml add d1 --rack Q61 --cable 8
harness targets --targets-file config\targets.yaml ls
harness diagnose --inventory inventory.yaml --target d1 --symptom "ECC errors"
```

Aliases are a path-only file; never credentials.

## 8. Reading the outputs

Under `harness_runs/<run-id>/`:

- `diagnosis.json` — prioritized recommendations with confidence and part
  references (pending human approval).
- `trace.json` — what was planned, collected, decoded, and concluded.
- `audit.jsonl` — append-only audit chain (redacted).
- `dumps/*.txt` — raw collector output for manual cross-checking.

## 9. Locally hosted model (temporary debug agent)

You can serve the reasoning model on a lab box (vLLM is natively
OpenAI-compatible) and point the harness at it — including on the golden
server itself, the same rack/cable target you are debugging. The simple path
is the guided setup — `harness menu → model → local/harness-diag` (or
`/model` in a session):

1. Provider and endpoint URL (Enter = `http://127.0.0.1:8000/v1`).
2. Transport: **direct** if the workstation can reach the endpoint, or
   **tunnel — model runs on the golden server** if it lives on the rack/cable
   debug target. The wizard asks the rack/cable (prefilled from the active
   session target), probes the node over the jumpin console, and you pick the
   tunnel `HOST:PORT` from the discovered candidates.
3. The wizard probes `GET /models` through the hop and lists the served
   model ids — pick one (vLLM rejects any other name).

The choice is remembered in `config/models.yaml`, so every later run targets
the server by `--rack/--cable` as usual and the LLM hop is automatic.

To see the candidates standalone (same read-only probes: `hostname -I`,
`ss`, `sudo -S docker ps` when the sudo password is configured):

```powershell
harness llm discover --inventory inventory.yaml --rack Q61 --cable 8
#   addresses : 10.0.0.42, ...
#   container : vllm-qwen -> 8000
#   suggested: harness llm check --tunnel 10.0.0.42:8000 --inventory ...
```

If the rack manager refuses the forward, the wizard prints the
reverse-tunnel relay recipe and can save the relay URL as the endpoint:

```text
# from the node console (jumpin session):
ssh -fN -R 127.0.0.1:18000:127.0.0.1:8000 <relay-reachable-from-node>
# the harness then calls http://127.0.0.1:18000/v1 like any local endpoint
```

Manual fallback (no discovery tooling on the node): open the console and read
the listening sockets directly — port `8000` shows as hex `1F40` with state
`0A` (LISTEN):

```powershell
harness console --inventory inventory.yaml --rack Q61 --cable 8 `
  --probe "cat /proc/net/tcp" --probe "cat /etc/hosts"
```

The equivalent one-shot flags (also useful to override the remembered model
for a single run):

```powershell
# 1. Preflight: reachability + served model list (staged report)
harness llm check --url http://10.0.0.42:8000/v1 --model Qwen2.5-7B-Instruct

# 2a. Directly reachable endpoint:
harness diagnose --inventory inventory.yaml --rack Q61 --cable 8 `
  --symptom "uncorrectable ECC" `
  --llm local --llm-model local/Qwen2.5-7B-Instruct `
  --llm-url http://10.0.0.42:8000/v1

# 2b. Endpoint on a node behind the rack-manager console hop -- the harness
#     forwards HTTP through the rackmgr SSH connection for this run only:
harness llm check --tunnel 10.0.0.42:8000 --inventory inventory.yaml
harness diagnose --inventory inventory.yaml --rack Q61 --cable 8 ... `
  --llm-tunnel 10.0.0.42:8000
```

Notes:

- After every pick the endpoint is probed non-blockingly; if the chosen model
  id is not in the served list, the picker offers the served ids (vLLM
  rejects any other name — a silent `harness-diag` id would 404 every call).
- `--llm-tunnel` implies provider `local`; it reuses `console_defaults`
  credentials + pinned host keys, opens at run start, closes at run end.
- If the rack manager refuses forwarding (`[FAIL] forward`), the output prints
  the reverse-tunnel recipe to run from the node console; then pass
  `--llm-url http://127.0.0.1:18000/v1` instead.
- Calibration is keyed per ident (`local/Qwen2.5-7B-Instruct`); a temporary
  model starts conservative until it accumulates verified outcomes, and
  rolling back is just dropping the flags.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `SerialConsoleError: allowed only at lab/qa` | `trust_level` is `prod`; set `lab`/`qa` |
| `sudo password missing from vault` | no secret at `sudo_vault_path` (or `HARNESS_NO_PROMPT=1` / non-tty input) |
| `SerialProbeDenied: ... i2cset ...` | write tool — only reads allowed (`i2cdump`/`i2cget`) |
| `did not see node prompt "~#"` | wrong/blank cable, or session timeout |
| `spawn q71-1 rm` fails | rack-manager user lacks `jumpin`, or wrong rack id |
| `Host key not in pinned known_hosts` | rack manager key not imported |
| `SerialProbeDenied` on `lspci \| grep` | no `sh`/`>`/`$( )` in pipelines |

See [TESTING.md](../TESTING.md) for the full live-console test walkthrough
(rack Q71, cable 8) and acceptance checklist.
