# Testing the Harness — Q71, Cable 8 (rack-manager console path)

This guide is for exercising the harness against **Q71, cable 8** (a specific server
reachable over that rack cable). Because these rack units are reached over a serial
console (SSH → rack manager → `jumpin` → `start serial session`), the harness's
console path (`engine/sol.py`) is the one to test — not a direct SSH to the server.

> Naming: in the code, the `expect` pattern is `jumpin q<rack>-1 rm` and
> `start serial session -i <cable>`. For **Q71 cable 8** that means
> `rack = Q71`, `cable = 8`.

---

## 0. Prerequisites

| Item | Requirement |
|---|---|
| Python | 3.11+ (dev machine also has 3.14) |
| Package | `pip install -e ".[test]"` (already done in this repo) |
| Rack manager SSH key | A private key the rack manager will accept for the `jumpin` session user |
| Rack manager host key | The rack manager's public host key pinned in a `known_hosts` file |

Two checks before anything else:
1. You can reach the rack manager over SSH with the intended user and key.
2. `jumpin`, `start serial session`, and expect exist on the rack manager.

---

## 1. Configure the harness inventory for Q71 cable 8

Edit `src/harness/config/inventory.yaml` (or better, a separate
`config/q71.yaml` you point the loader at). Set the `console` block:

```yaml
trust_level: lab          # console path is blocked at prod

hosts:
  - name: q71-cable8
    address: <server-mgmt-ip>         # optional; console is the real path here
    model: <server-fru-model>
    ssh:
      user: diagbot
      identity_vault_path: secret/harness/diagbot/id_ed25519
      known_hosts_path: config/known_hosts
    bmc:
      address: <server-bmc-ip>
      username: bmc-ro
      password_vault_path: secret/harness/bmc/bmc-ro
    collector_profile: default
    console:
      address: 192.168.202.51       # rack manager
      user: log                     # rack-manager login user
      identity_vault_path: secret/harness/rackmgr/id_ed25519
      known_hosts_path: config/rackmgr_known_hosts
      rack: Q71
      cable: 8                      # cable 8 on rack Q71
      tool: jumpin
      trust_level: lab
      prompts: ["RScmCli#", ":~$"]
      port: 2200                    # BMC access port for the serial session
      sudo_vault_path: secret/harness/bmc/sudo   # BMC sudo password (never inline)
```

**BMC access (important):** the serial session on the **BMC access port (2200)** drops
you into the BMC's Linux (BusyBox) shell, not the host console. Host-OS tools
(`rdmsr`, `smartctl`, `lspci`, `dmidecode`, util-linux `dmesg -l`) do not exist
there. I2C register reads (`i2cdump`/`i2cget`) and `ipmitool` need **sudo**; the
harness feeds the password from `sudo_vault_path` (the sudo password is **never
written in the repo** — put it in the secret store, e.g. a file in `--secret-dir`
for lab use). The node prompt on the BMC shell is `admin@<host>:~$` — the second
entry in `prompts` must match it (e.g. `:~$`). The rack id in the `jumpin` command
is normalized: `Q71`, `q71` and `71` all render `jumpin q71-1 rm`.

**Diagnose over the console picks BMC probes automatically:** when the target is a
console, the collector plan substitutes host-OS probes with BMC-shell equivalents
(`sudo -S ipmitool sensor list`, `sudo -S ipmitool sel list`,
`sudo -S ipmitool fru print`, and `dmesg -r` for the BusyBox ring buffer); the
`pcie`/`storage` collectors are skipped and model detection reads `ipmitool fru
print` instead of `dmidecode`. That is why a `diagnose --rack/--cable` run never
issues `rdmsr`/`smartctl` over the console.

Then run the inventory lint to make sure it parses and contains no inline secrets:

```powershell
python -c "from harness.config.inventory_lint import load_inventory; i=load_inventory('src/harness/config/inventory.yaml'); print('OK:', i.host_names)"
```

---

## 2. Unit tests (no hardware needed)

The 250 unit tests cover the console script builder, probe validation, trust gating,
decoder, RAG, audit chain, and the full orchestration with a fake runner:

```powershell
python -m pytest -q
# -> 250 passed
```

Relevant to the console path:

```powershell
python -m pytest -q tests/unit/test_serial_console.py -v
```

---

## 3. Smoke test against the rack manager (no node yet)

`render_expect_script` should produce a script that drops you into the node console.
Verify the rack-manager hop independently of the rack first. Use the real secret
store described next, or seed a memory store for a quick check:

```powershell
python -c @'
import tempfile, pathlib
from harness.config.vault import MemorySecretStore
from harness.engine.sol import render_expect_script
s = render_expect_script(
    tool="jumpin", rack="Q71", cable="8",
    commands=['lspci -vvv -n -s 00:06:00.0 | grep -E "LnkSta:"'],
    port=2200,
)
print(s)
'@
```

Confirm the printed script has:
```
spawn jumpin q71-1 rm
expect "RScmCli#"
send "start serial session -i 8 -p 2200\r"
expect "~#"
send "lspci ... | grep -E \"LnkSta:\"\r"
expect "~#"
send "exit\r"
```

For a sudo probe on the BMC shell, pass `sudo_password` and expect the handshake:

```powershell
python -c @'
from harness.engine.sol import render_expect_script
s = render_expect_script(
    tool="jumpin", rack="Q71", cable="8",
    commands=['sudo -S i2cdump -y -f 0 0x51'],
    port=2200, sudo_password="<secret-from-store>",
)
print(s)
'@
```
```
send "sudo -S i2cdump -y -f 0 0x51\r"
expect "password for"
send "<secret-from-store>\r"
expect "~#"
```

---

## 4. Live console test against Q71 cable 8

Use `ConsoleDomain(trust_level="lab")` with the rack-manager key material plus a
host-key-pinned client. This example uses a small local `SecretStore` seeded with
the rack-manager private key so you don't touch a vault first; swap in your real
`SecretStore` (Vault / AWS SM) for production.

```powershell
python -c @'
from pathlib import Path
from harness.config.models import ConsoleDomain
from harness.config.vault import MemorySecretStore
from harness.engine.sol import SerialConsole

key = Path("<PATH>/rackmgr_id_ed25519").read_bytes()
sudo = Path("<PATH>/bmc_sudo_password").read_bytes()
store = MemorySecretStore({
    "secret/harness/rackmgr/id_ed25519": key,
    "secret/harness/bmc/sudo": sudo,   # BMC sudo password, never inline
})

console = ConsoleDomain(
    address="192.168.202.51",
    user="log",
    identity_vault_path="secret/harness/rackmgr/id_ed25519",
    known_hosts_path="config/rackmgr_known_hosts",
    rack="Q71",
    cable="8",
    tool="jumpin",
    trust_level="lab",
    prompts=("RScmCli#", "~#"),
    port=2200,                        # BMC access port
    sudo_vault_path="secret/harness/bmc/sudo",
)

sc = SerialConsole(console, store)
try:
    result = sc.run_probes([
        'sudo -S i2cdump -y -f 0 0x51',          # BMC shell: I2C read via sudo
        'sudo -S ipmitool sensor list',          # BMC shell: sensors via sudo
    ])
    print("PROBE COUNT:", result.probe_count)
    print(result.output[:2000])
finally:
    sc.close()
'@
```

Expected output:
- `PROBE COUNT: 2`
- the i2c register dump and the sensor table, with the sudo password handshake
  handled automatically by the expect script.

Without `port` (host console, not BMC), the LnkSta link check still works:

```powershell
python -c @'
# ... same store/console setup, but port=None, and:
result = sc.run_probes([
    'lspci -vvv -n -s 00:06:00.0 | grep -E "LnkSta:"',
])
for ln in result.probe_lines("LnkSta:"):
    print("LNK:", ln)
'@
```

If `LnkSta` shows a slow link (e.g. `2.5GT/s, Width x1` when the slot should be
`x16`), that is the "slower harddrive" pattern you use as a register check — feed
that into the decode pipeline as described below.

> **Safety:** `SerialConsole` refuses to construct unless `trust_level` is
> `lab`/`qa`, and `validate_serial_probe` will reject any destructive or
> injection-y command you try to send. Host keys are pinned — if the rack manager's
> key ever changes, the session raises and stops rather than silently trusting it.

---

## 5. Route the console output into a register check

`ConsoleResult.probe_lines("LnkSta:")` extracts the link-status lines. To treat
that as decoded evidence, wrap it as a `RegisterDump` and run it through the
decoder (this is where your architecture doc lookup hooks in):

```python
from harness.inspect.base import RegisterDump
from harness.diagnosis.summarize import summarize

dumps = [RegisterDump(
    subsystem="pcie",
    source="console:lspci LnkSta",
    raw="\n".join(result.probe_lines("LnkSta:")),
    cmd_argv=["serial-console"], ok=True,
)]
print(summarize(dumps).interesting)   # anomalous lines surfaced to the LLM
```

---

## 6. What to check for acceptance

- [ ] Console refuses to run at `trust_level=prod`.
- [ ] Destructive probes are rejected (`shutdown`, `dd`, `>`/`;`/`$(...)`).
- [ ] `i2cset` and `ipmitool sel clear` are rejected; `i2cdump`/`i2cget` accepted.
- [ ] Rack/cable injection is rejected (e.g. `Q71;reboot`).
- [ ] Script renders with `expect -c`, correct node prompt wait, and `exit`.
- [ ] Script includes `-p 2200` when `port=2200` (BMC access port).
- [ ] sudo probe renders the `password for` -> send-password handshake.
- [ ] Live run reaches Q71 cable 8 console and returns `LnkSta:` lines.
- [ ] Host-key change on the rack manager causes a hard stop (not silent trust).

---

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `SerialConsoleError: allowed only at lab/qa` | `trust_level` is `prod`; set `lab`/`qa` |
| `sudo password missing from vault` | no secret at `sudo_vault_path`; put the BMC sudo password in the store (e.g. a file under `--secret-dir`) |
| `SerialProbeDenied: ... i2cset ...` | only read tools are allowed (`i2cdump`/`i2cget`/`i2cdetect`); `i2cset` is a write tool and is never accepted |
| `did not see node prompt "~#"` | wrong/blank `cable`, or the session timed out; confirm cable 8 maps to the target server |→node map |
| `spawn q71-1 rm` fails | rack-manager user lacks `jumpin`; or wrong rack id |
| `Host key not in pinned known_hosts` | rack manager key not imported into `config/rackmgr_known_hosts` |
| `SerialProbeDenied` on your `lspci | grep` | confirm it isn't using `sh`/`>`/`$( )` in the pipeline |