---
name: HoreKa cluster access
description: How to run jobs on HoreKa from kislurm — SSH proxy, OTP auth, file sync, job submission
type: reference
---

## Access pattern

SSH goes through a proxy: `ferreira@aadlogin.informatik.uni-freiburg.de` → `fr_ff1042@horeka.scc.kit.edu`
Auth requires OTP (TOTP) + password — no key-based auth.

## Tools (on kislurm)

- **run_remote.py**: `/work/dlclarge2/ferreira-oellm/open-instruct/oellm/utils/horeka/run_remote.py`
- **transfer_data.py**: `/work/dlclarge2/ferreira-oellm/open-instruct/oellm/utils/horeka/transfer_data.py`
- **Credentials**: `/work/dlclarge2/ferreira-oellm/open-instruct/.env` (HOREKA_TOTP_SECRET, HOREKA_PW)

## Steps

1. **Establish proxy tunnel** (once per session):
   ```
   mkdir -p /tmp/ssh-mux
   ssh -fNM -S /tmp/ssh-mux/aadlogin ferreira@aadlogin.informatik.uni-freiburg.de
   ```

2. **Sync code**: `python3 /work/dlclarge2/ferreira-oellm/open-instruct/oellm/utils/horeka/transfer_data.py --code`
   - Rsyncs to `/hkfs/work/workspace/scratch/fr_ff1042-oellm/open-instruct`
   - Excludes .venv, checkpoints, data, models, .git

3. **Sync data**: `python3 .../transfer_data.py --data <local_path> <remote_path>`
   - Each rsync needs fresh OTP — script handles via pyotp

4. **Run remote commands**: `python3 .../run_remote.py "command"`
   - Output capture is flaky — run_remote.py sometimes returns empty
   - Direct pexpect approach works more reliably (see script source)

5. **Check jobs**: `python3 .../run_remote.py "squeue -u fr_ff1042"`

## HoreKa details

- **Workspace (oellm)**: `/hkfs/work/workspace/scratch/fr_ff1042-oellm` (250TB, 60-day lifetime)
- **Workspace (autoresearch)**: `/hkfs/work/workspace/scratch/fr_ff1042-autoresearch` (expires May 15, 2026)
- **SLURM account**: `hk-project-p0024002` (also `hk-project-p0021863`)
- **GPU partitions**:
  - `accelerated` — A100 nodes (162 nodes, 648 GPUs, max 2 days)
  - `accelerated-h100` — H100 nodes (20 nodes, 80 GPUs, max 2 days)
  - `accelerated-h200` — H200 nodes (exists, details not yet retrieved)
  - `accelerated-h200-8` — H200 8-GPU nodes
  - `dev_accelerated` / `dev_accelerated-h100` — dev queues (1h max)
- OTP is single-use, 30s window — run_remote.py waits for fresh window automatically
- `sinfo` returns "Access/permission denied" but `scontrol show partition` and `squeue` work fine

## Direct pexpect pattern (when run_remote.py output capture fails)

```python
import pexpect, time, pyotp, re

with open('/work/dlclarge2/ferreira-oellm/open-instruct/.env') as f:
    env = {}
    for line in f:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            env[k] = v

totp = pyotp.TOTP(env['HOREKA_TOTP_SECRET'])
remaining = totp.interval - (int(time.time()) % totp.interval)
if remaining < 15:
    time.sleep(remaining + 1)
otp = totp.now()

cmd = (
    'ssh -o ConnectTimeout=15 -tt '
    '-o ProxyCommand="ssh -o ControlPath=/tmp/ssh-mux/aadlogin -W %h:%p ferreira@aadlogin.informatik.uni-freiburg.de" '
    'fr_ff1042@horeka.scc.kit.edu "YOUR_COMMAND; echo XDONE"'
)
child = pexpect.spawn(cmd, timeout=60, encoding='utf-8')
child.expect('OTP', timeout=20)
child.sendline(otp)
time.sleep(3)
child.read_nonblocking(size=10000, timeout=3)
child.sendline(env['HOREKA_PW'])
time.sleep(8)
out = child.read_nonblocking(size=30000, timeout=8)
clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]|\[\?[0-9]+[hl]', '', out)
print(clean.strip())
child.close()
```

## autoresearch-automl on HoreKa

- **Code synced to**: `/hkfs/work/workspace/scratch/fr_ff1042-autoresearch/autoresearch-automl/`
- **Status**: Code synced, venv + models + SLURM scripts still need setup

Still need:
1. Install venv on HoreKa (`uv venv && uv pip install -e ".[dev,all]"`)
2. Download Qwen3.5-0.8B and Qwen3.5-27B models to workspace
3. Adapt SLURM scripts for HoreKa partition names and account flags
4. Submit with `--account=hk-project-p0024002 -p accelerated-h200`
