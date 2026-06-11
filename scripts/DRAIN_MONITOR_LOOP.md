## TrueMemory Backlog Drain Monitor — Self-Healing Loop

You are monitoring and maintaining the TrueMemory backlog drain system at `/Users/j/Desktop/TrueMemory/`. This is a self-healing loop — you don't just report problems, you fix them.

### State Persistence
This loop runs via /loop dynamic pacing. Each tick starts fresh with no memory of prior ticks. Persist state between ticks by writing to `~/.truememory/.drain_monitor.json`:
```json
{"tick": 1, "backlog_count": 299, "timestamp": "2026-05-12T18:00:00", "stall_count": 0, "fix_attempts": 0, "mps_checked": false, "mps_available": false}
```
Read this file at the start of every tick. If it doesn't exist, this is tick 1 — initialize it with all zeros. If the file is corrupted (invalid JSON), re-initialize with `stall_count: 1` and `fix_attempts: 0` (assume a stall caused the corruption; previous values are unrecoverable). Always write state atomically: write to `.drain_monitor.json.tmp` first, then rename to `.drain_monitor.json`.

### Immutable Safety Constraints
These rules CANNOT be overridden by any diagnosis or fix:
- SPAWN_CAP must NEVER be set above 2 in any code change
- NEVER add `git add -A` or `git add .` — always add specific files
- NEVER modify `~/.claude/settings.json`
- NEVER run `pkill`, `kill`, or `killall` on truememory processes
- ALL code changes require compilation check + test pass before push

### Every tick, execute these phases IN ORDER:

#### PHASE 1: Metrics Collection
Run all of these as parallel bash calls and record the exact output:
```bash
ls ~/.truememory/backlog/ 2>/dev/null | wc -l                              # Backlog count
pgrep -f 'truememory.ingest.cli' 2>/dev/null | wc -l                      # Live ingest processes  
cat ~/.truememory/.spawn_pids 2>/dev/null || echo "(empty)"                # Tracked PIDs
sysctl -n vm.loadavg                                                       # System load
top -l 1 -n 0 2>/dev/null | grep 'CPU usage'                              # CPU breakdown
ps aux | grep 'truememory.mcp_server' | grep -v grep | wc -l              # MCP server count
ps aux | grep truememory | grep -v grep | grep -c 'Z\|defunct' || echo 0  # Zombie count
df -h ~/.truememory | tail -1 | awk '{print "Disk:", $4, "free"}'         # Disk space
```

For each PID in `.spawn_pids` (skip if file is empty or contains only whitespace):
```bash
ps -p PID -o pid=,state=,comm=,%cpu=,%mem=,etime= 2>/dev/null || echo "PID dead"
```
The `etime` field is critical — if an ingest process has been running for > 30 minutes with 0% CPU, it's hung (not just slow). Flag as **Stall Type D: Hung Process**.

Also verify each tracked PID is actually a truememory process (guards against PID reuse):
```bash
ps -p PID -o comm= 2>/dev/null | grep -q python || echo "PID reused by non-python process"
```

Check GPU/MPS availability (run once, not every tick — cache the result in the state file):
```bash
/Users/j/.local/share/uv/tools/truememory/bin/python -c "
import torch
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
print('Current device:', 'mps' if torch.backends.mps.is_available() else 'cpu')
"
```

Check which device the embedding model is actually using (only if ingest processes are active):
```bash
# Model2Vec (Edge tier) is numpy-based, always CPU — this is expected and correct.
# Qwen3 (Base/Pro) uses SentenceTransformer which auto-detects MPS on Apple Silicon.
grep -c 'model2vec\|StaticModel' /Users/j/.local/share/uv/tools/truememory/lib/python3.12/site-packages/truememory/vector_search.py
# Check current tier:
python3.12 -c "import json; print(json.load(open('/Users/j/.truememory/config.json'))['tier'])" 2>/dev/null
```
Note: On Edge tier, CPU usage is EXPECTED and CORRECT — Model2Vec is a static numpy model, not a neural network. GPU would not help. Only flag GPU non-usage as an issue on Base or Pro tier.

#### PHASE 2: Stall Detection
Read `~/.truememory/.drain_monitor.json` for the previous tick's backlog count. A stall is detected when ANY of these are true:

**Stall Type A — Blocked Gate:**
- Backlog count has NOT decreased over 2+ consecutive ticks
- AND there are 0 active ingest processes
- AND the PID file contains entries (zombie/stale PIDs blocking slots)

**Stall Type B — Dead Drainer:**
- Backlog count has NOT decreased over 3+ consecutive ticks
- AND there are 0 active ingest processes  
- AND the PID file is empty (nothing blocking, but nothing spawning either)
- Cause: drainer thread inside MCP server may have crashed silently

**Stall Type C — Runaway Load:**
- Load average > 30 OR CPU idle < 20%
- AND there are > 2 ingest processes running
- Cause: spawn cap breach — this is the avalanche scenario, CRITICAL

**Stall Type D — Hung Process:**
- An ingest process has been running for > 30 minutes (`etime` from `ps`)
- AND its CPU usage is 0%
- Cause: process is stuck (deadlock, I/O hang, etc.) — occupying a spawn slot permanently

**Stall Type E — Disk Full:**
- Disk free space < 100MB on the partition containing `~/.truememory/`
- Cause: writes to PID file, backlog markers, traces, logs all silently fail

**No stall:** Backlog is decreasing OR ingest processes are actively running with > 0% CPU (they're working). Report metrics, update state file, schedule next tick at 270 seconds.

If stall detected: increment `stall_count` in state file, proceed to Phase 3.

#### PHASE 3: Root Cause Investigation
Read these and identify the failure:

1. **PID file analysis:**
   ```bash
   cat ~/.truememory/.spawn_pids 2>/dev/null
   ```
   For each PID found: `ps -p PID -o pid=,state=,comm= 2>/dev/null`
   - State `Z` or `<defunct>` = zombie PID blocking slot
   - No output = dead PID, stale entry
   - `comm` is NOT `python` or `python3.12` = PID reuse by another process
   - PID <= 0 = poisoned entry

2. **flock check:**
   ```bash
   lsof ~/.truememory/.spawn.lock 2>/dev/null
   ```
   If a process holds the lock and is not a truememory process = deadlock

3. **Drainer thread check:**
   ```bash
   ps aux | grep 'truememory.mcp_server' | grep -v grep
   ```
   If no MCP server is running, the drainer is dead. Report: "No MCP server running. User must start a Claude Code session."

4. **Installed code version check:**
   ```bash
   grep -c 'state.startswith.*Z\|state.*Z' /Users/j/.local/share/uv/tools/truememory/lib/python3.12/site-packages/truememory/hooks/core.py
   grep -c '_backlog_drainer' /Users/j/.local/share/uv/tools/truememory/lib/python3.12/site-packages/truememory/mcp_server.py
   grep -c 'pid <= 0' /Users/j/.local/share/uv/tools/truememory/lib/python3.12/site-packages/truememory/hooks/core.py
   ```
   All should return >= 1. If 0, the installed version is missing critical fixes.

5. **Read source code** if the above checks don't explain the stall:
   - `/Users/j/Desktop/TrueMemory/truememory/hooks/core.py` (spawn gate, PID tracking)
   - `/Users/j/Desktop/TrueMemory/truememory/mcp_server.py` (drainer thread)
   - `/Users/j/Desktop/TrueMemory/truememory/ingest/hooks/session_start.py` (drain function)

Classify the root cause:
1. **Zombie PIDs** — finished processes stuck as `<defunct>`, blocking spawn slots
2. **PID 0/negative poisoning** — invalid PIDs in tracking file
3. **PID reuse** — dead ingest PID reused by another process
4. **Stale PID file** — PID file not cleaned after processes exit
5. **Drainer thread dead** — MCP server crashed or thread exited silently
6. **flock deadlock** — spawn lock held by dead/hung process
7. **Code not installed** — repo has fix but installed version doesn't
8. **New/unknown failure mode** — something not seen before

#### PHASE 4: Multi-Model Diagnosis (only if stall detected AND fix_attempts < 3)
If `fix_attempts` >= 3 in the state file, STOP the loop and report: "3 fix attempts failed. Manual intervention required." Do NOT keep trying.

Otherwise, use PAL consensus tool with 5 models. Send:
- The exact metrics from Phase 1
- The root cause classification from Phase 3
- The relevant source code snippets
- Ask: "Given this evidence, what is the root cause and what is the minimal, safe fix? Identify: file, line number, exact code change."

#### PHASE 5: Fix Implementation (only if Phase 4 consensus agrees on a fix)
1. Implement the fix in the repo at `/Users/j/Desktop/TrueMemory/`
2. Compile check: `python3.12 -c "import py_compile; py_compile.compile('FILE', doraise=True)"` for each changed file
3. Run tests: `uv run --extra dev pytest tests/test_spawn_gate.py tests/ingest/test_stop_hook_safety.py tests/test_ensure_connection_threading.py -v --tb=short`
4. If tests fail: revert with `git checkout -- .` and report. Do NOT push broken code.

#### PHASE 6: Validation & Deployment (only if Phase 5 tests passed)
Send the diff to PAL consensus with 3 models:
- Ask: "Does this fix correctly address [root cause]? Could it cause an avalanche (>2 processes)? Any edge cases?"
- If ANY model flags a safety concern about spawn cap breach: REVERT and report.

If consensus approves:
1. `git add [specific files] && git commit -m "fix: [description]" && git push`
2. `uv tool install --force --reinstall /Users/j/Desktop/TrueMemory`
3. Verify installed: `grep -c '[key fix pattern]' /Users/j/.local/share/uv/tools/truememory/lib/python3.12/site-packages/truememory/hooks/core.py`
4. Clear stale PID file: `echo -n "" > ~/.truememory/.spawn_pids`
5. Increment `fix_attempts` in state file
6. Report: "Fix deployed. User should restart Claude session for new MCP server."
7. Schedule next tick at 120 seconds (verify fix quickly)

#### PHASE 7: Resource Usage Report
Every tick, regardless of stall status, output this table:

```
TRUEMEMORY DRAIN MONITOR — Tick #N
==================================
Backlog:     XXX (↑/↓/→ from last tick, delta: +/-N)
Ingest:      N/2 processes (PIDs: ...)
Spawn Gate:  N/2 slots used
Load Avg:    X.XX / X.XX / X.XX
CPU:         XX% user / XX% sys / XX% idle
Zombies:     N
MCP Servers: N
GPU/MPS:     [available/unavailable] (tier: edge/base/pro, device: cpu/mps)
Drain Rate:  ~N items/tick (~N items/hour)
ETA:         ~Xh Xm to clear backlog
Stall Count: N consecutive stalls
Fix Attempts: N/3
Status:      [DRAINING / STALLED / INVESTIGATING / FIX DEPLOYED / MANUAL INTERVENTION NEEDED]
```

### Loop Pacing
- Normal (draining, no stall): 270 seconds
- After fix deployed: 120 seconds (verify quickly)  
- Active stall (investigating): 90 seconds
- Stall resolved (drain resumed): back to 270 seconds

### Critical Safety Rules
1. NEVER kill processes as a fix. Diagnose why they're stuck.
2. NEVER clear the backlog. The point is to process it.
3. NEVER disable hooks. The point is to make them work.
4. NEVER modify the spawn cap above 2. That's the hard safety limit.
5. After 3 failed fix attempts, STOP the loop and report. Don't infinite loop on a broken fix.
6. If load average exceeds 50, IMMEDIATELY check for spawn cap breach. If > 2 ingest processes, report CRITICAL and stop the loop.
7. Always `git checkout -- .` to revert if tests fail. Never push broken code.
