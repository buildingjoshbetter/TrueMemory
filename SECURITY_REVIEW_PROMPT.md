# TrueMemory Security Review Prompt

> Paste this into a fresh Claude Code session from the TrueMemory repo root.
> Run with: `/loop` for iterative fixing, or as a single-shot review.
> Estimated time: 2-4 hours for full review + fixes.

---

## Instructions

You are conducting a comprehensive security review of the TrueMemory codebase. TrueMemory is a persistent memory system for AI agents. It runs as an MCP server (`truememory-mcp`) with a SQLite backend, an ingestion pipeline with lifecycle hooks, and a Python SDK (`truememory.Memory`).

**Why this matters NOW:** TrueMemory is adding network-facing features — usage telemetry (issue #190), cloud corpus sync with multi-device replication and collaborative sharing (issue #199), and a web visualization dashboard. Code that was safe when local-only becomes a real attack surface once it touches the network. Fix everything now, before those features ship.

**Your role:** You are a senior security engineer. You are not looking for style issues or minor code quality problems. You are looking for vulnerabilities that could:
1. Allow data exfiltration (memories contain PII, preferences, relationships)
2. Allow code execution via crafted input
3. Allow denial of service
4. Compromise the SQLite database
5. Leak credentials (API keys stored in config.json)
6. Enable supply chain attacks via dependencies

---

## Phase 1: SQL Injection Sweep

A third-party scanner (Semgrep via MseeP.ai) flagged 20 medium-severity SQL string concatenation issues. Two were explicitly identified in `truememory/clustering.py` at lines 172 and 258. The remaining 18 were not disclosed.

### Task 1.1: Find ALL SQL string concatenation

Search the ENTIRE codebase for SQL queries built via string concatenation or f-strings instead of parameterized queries.

```bash
# Run these searches:
grep -rn "f\"SELECT\|f\"INSERT\|f\"UPDATE\|f\"DELETE\|f\"DROP\|f\"CREATE\|f\"ALTER\|f\"PRAGMA" truememory/ --include="*.py"
grep -rn "\"SELECT.*%s\|\"INSERT.*%s\|\"DELETE.*%s" truememory/ --include="*.py"
grep -rn "\"SELECT.*\" +\|\"INSERT.*\" +\|\"DELETE.*\" +" truememory/ --include="*.py"
grep -rn "\.format(" truememory/ --include="*.py" | grep -i "select\|insert\|delete\|update\|drop\|create"
grep -rn "execute(f\"" truememory/ --include="*.py"
grep -rn "\.execute(\".*\+\|\.execute(\".*%\|\.execute(f\"" truememory/ --include="*.py"
```

For EACH finding:
1. Read the surrounding function to understand the context
2. Determine: is the interpolated value from user input, or from internal/trusted code?
3. If from user input (MCP tool parameters, CLI args, env vars, transcript content): **HIGH priority — fix immediately**
4. If from internal code (table names, column names that are hardcoded strings): **LOW priority — fix for hygiene but not exploitable**
5. Convert ALL instances to parameterized queries (`?` placeholders) regardless of risk level

### Task 1.2: Fix each instance

For each SQL string concatenation:
- Replace `f"SELECT ... WHERE x = {value}"` with `"SELECT ... WHERE x = ?", (value,)`
- Replace `"SELECT ... WHERE x = '%s'" % value` with `"SELECT ... WHERE x = ?", (value,)`
- For table/column names that can't be parameterized (SQLite limitation), validate against an allowlist:
  ```python
  ALLOWED_TABLES = {"messages", "vec_messages", "vec_messages_sep", "entity_profiles", ...}
  if table_name not in ALLOWED_TABLES:
      raise ValueError(f"Invalid table name: {table_name}")
  ```

### Task 1.3: Verify the specific findings

Read `truememory/clustering.py` lines 172 and 258 specifically. These were the two explicitly flagged locations. Fix them and verify the fix is correct.

---

## Phase 2: Input Validation Audit

TrueMemory accepts input from multiple sources. Each needs validation.

### Task 2.1: MCP tool inputs

Read `truememory/mcp_server.py`. For each `@mcp.tool()` function, check:
- What parameters does it accept?
- Are string parameters length-bounded?
- Are numeric parameters range-validated?
- Could a malicious Claude prompt craft inputs that cause harm?

Specific tools to audit:
- `truememory_store(content, user_id, metadata)` — can `content` be arbitrarily large? Can `metadata` contain SQL? Can `user_id` contain path traversal?
- `truememory_search(query, user_id, limit)` — can `query` cause SQL injection via FTS5? Can `limit` be negative or extremely large?
- `truememory_forget(memory_id)` — can `memory_id` be crafted to delete unintended records?
- `truememory_configure(tier, api_key, api_provider)` — is `api_key` validated? Could a malicious key value cause issues?
- `truememory_entity_profile(entity_name)` — is `entity_name` sanitized?

### Task 2.2: Hook inputs

Read all 4 hook scripts in `truememory/ingest/hooks/`:
- `session_start.py` — reads stdin JSON. Can crafted JSON cause issues?
- `stop.py` — reads `transcript_path` from stdin. Path traversal risk?
- `compact.py` — same transcript_path concern
- `user_prompt_submit.py` — reads `prompt` from stdin, writes to file. Can crafted prompt content cause issues when written to the buffer file?

### Task 2.3: CLI inputs

Read `truememory/ingest/cli.py`:
- `--threshold` is `type=float` but no range validation in argparse (the EncodingGate now clamps, but the CLI should validate too)
- `--user` is a free-form string — is it used in SQL queries or file paths?
- `--db` is a file path — path traversal risk?
- `--transcript` is a file path — same concern

### Task 2.4: Environment variable inputs

Search for all `os.environ.get()` calls:
```bash
grep -rn "os.environ.get\|os.environ\[" truememory/ --include="*.py"
```

For each env var:
- What does it control?
- Is the value validated before use?
- Could a malicious env var value cause harm? (SQL injection via env var → query parameter)

---

## Phase 3: File System Security

### Task 3.1: Path traversal

Search for file operations where the path comes from user input:
```bash
grep -rn "Path(\|open(\|read_text\|write_text\|read_bytes\|write_bytes" truememory/ --include="*.py" | grep -v "test\|__pycache__"
```

Specific concerns:
- `stop.py` reads `transcript_path` from stdin JSON — is this validated?
- `user_prompt_submit.py` constructs buffer file paths from `session_id` — is session_id sanitized? (Check: it already sanitizes with `"".join(c for c in session_id if c.isalnum() or c in "-_")[:64]` — verify this is sufficient)
- `compact.py` reads `transcript_path` — same concern as stop.py
- Config files at `~/.truememory/config.json` — chmod 600 already applied, verify

### Task 3.2: Credential storage

`config.json` stores API keys in plaintext:
```json
{"tier": "pro", "anthropic_api_key": "sk-ant-..."}
```

Verify:
- File permissions: chmod 600 on the file, chmod 700 on the directory
- Are API keys ever logged? Search for log statements that might include key values
- Are API keys ever included in error messages shown to users?
- Are API keys sent anywhere other than the intended API endpoint?

```bash
grep -rn "api_key\|API_KEY\|_key" truememory/ --include="*.py" | grep -i "log\.\|print\|warn\|error\|debug"
```

### Task 3.3: Temp files and traces

TrueMemory writes to several directories:
- `~/.truememory/traces/` — decision traces (contain memory content)
- `~/.truememory/logs/` — ingestion logs
- `~/.truememory/buffers/` — user message buffers
- `~/.truememory/backlog/` — queued ingestions

Verify:
- Are these directories created with restrictive permissions?
- Are old files cleaned up? (buffer pruning exists — verify traces and logs too)
- Could trace files leak sensitive memory content?

---

## Phase 4: Dependency Security

### Task 4.1: Known vulnerabilities

Check for known CVEs in dependencies:
```bash
pip install pip-audit
pip-audit --requirement <(pip freeze) 2>/dev/null || echo "pip-audit not available"
```

Or manually check the key dependencies:
- `torch` — any known RCE vulnerabilities in the installed version?
- `sentence-transformers` — any known issues?
- `httpx` — any request smuggling or SSRF issues?
- `mcp[cli]` — any known MCP protocol vulnerabilities?
- `sqlite-vec` — any buffer overflow issues in the C extension?
- `model2vec` — any model deserialization vulnerabilities?

### Task 4.2: Model deserialization

TrueMemory loads ML models from HuggingFace Hub. PyTorch model files can contain arbitrary Python code (pickle-based serialization).

Verify:
- Are models loaded with `weights_only=True` (safe) or `torch.load()` (unsafe)?
- Does `sentence-transformers` use safe loading by default?
- Could a compromised HuggingFace model execute arbitrary code on the user's machine?

```bash
grep -rn "torch.load\|pickle.load\|pickle.loads\|joblib.load" truememory/ --include="*.py"
```

### Task 4.3: Supply chain

Check `pyproject.toml` dependency version ranges:
- Are any dependency ranges too broad? (e.g., `>=1.0` without an upper bound)
- Could a malicious new version of a dependency be pulled in?
- Is `torch>=2.0.0,<3.0` safe? What about `sentence-transformers>=3.0.0,<6.0`?

---

## Phase 5: FTS5 Security

SQLite FTS5 (full-text search) has its own query syntax that could be abused.

### Task 5.1: FTS5 injection

Read `truememory/fts_search.py`:
- How are search queries passed to FTS5?
- Can a user craft an FTS5 query that causes excessive CPU usage (ReDoS equivalent)?
- Can FTS5 special syntax (`NEAR`, `*`, `OR`, `AND`, `NOT`, `"..."`) be injected via the search query parameter?
- Are FTS5 queries parameterized or string-concatenated?

```bash
grep -rn "MATCH\|fts5\|FTS" truememory/ --include="*.py"
```

### Task 5.2: FTS5 denial of service

Can a crafted query cause FTS5 to scan the entire table? For example:
- `"*"` — match everything
- Very long queries with many OR clauses
- Nested parentheses

Check if there are any query length limits or complexity limits.

---

## Phase 6: Network Security (pre-telemetry / pre-sync)

These checks are for the CURRENT codebase, to establish a baseline before network features ship.

### Task 6.1: Outbound connections

Search for all HTTP/network calls:
```bash
grep -rn "httpx\.\|requests\.\|urllib\|urlopen\|socket\." truememory/ --include="*.py"
```

For each:
- What endpoint is being contacted?
- Is TLS enforced?
- Is the response validated?
- Could a MITM attack inject malicious data?
- Are there any SSRF risks (user-controlled URLs)?

### Task 6.2: HuggingFace Hub calls

When `HF_HUB_OFFLINE` is disabled (during tier switch, first install):
- What endpoints are contacted?
- Is model integrity verified (checksums)?
- Could a compromised CDN serve a malicious model?

### Task 6.3: LLM API calls

For OpenRouter, Anthropic, and OpenAI backends:
- Are API keys sent only to the correct endpoint?
- Is the response from the LLM API trusted? Could a malicious LLM response cause issues in the extractor?
- Is there any prompt injection risk where memory CONTENT is included in LLM prompts?

Read `truememory/ingest/extractor.py` and `truememory/ingest/models.py`:
- How is transcript content included in extraction prompts?
- Could stored memories influence future extraction in harmful ways?

---

## Phase 7: Concurrency and Race Conditions

### Task 7.1: SQLite concurrent access

TrueMemory uses WAL mode with a 10-second busy timeout. Check for:
- Can two processes write to the same table simultaneously?
- What happens if the MCP server and a CLI command (upgrade-tier) run at the same time?
- Is the busy timeout sufficient for long operations (re-embedding 100K memories)?

### Task 7.2: Thread safety

The MCP server uses background threads for model preloading. Check:
- Are all shared globals protected by locks?
- Can a search arrive before the preload threads finish? (Singleton locks should handle this)
- Are there any TOCTOU races (time-of-check-time-of-use) in file operations?

### Task 7.3: Hook concurrency

Multiple hooks can fire simultaneously (e.g., UserPromptSubmit + PreCompact). Check:
- `user_prompt_submit.py` uses `fcntl.flock` for buffer file locking — is this sufficient?
- `stop.py` has a spawn cap (SPAWN_CAP=2) — is it enforced correctly?
- Can two Stop hooks race and both spawn ingestion processes?

---

## Phase 8: Denial of Service

### Task 8.1: Memory exhaustion

Can any of these cause OOM?
- Storing an extremely large memory (`truememory_store` with a 10MB content string)
- Searching with `limit=999999999`
- Ingesting a transcript with 100K messages
- Loading multiple embedding models simultaneously

### Task 8.2: Disk exhaustion

Can any of these fill the disk?
- Unlimited trace file growth (`~/.truememory/traces/`)
- Unlimited log file growth (`~/.truememory/logs/`)
- Buffer file growth (pruning exists but verify)
- SQLite WAL file growth during long transactions

### Task 8.3: CPU exhaustion

Can any of these cause CPU spin?
- FTS5 query with pathological pattern
- Re-embedding a very large corpus (no progress indicator, no timeout)
- Encoding gate with many candidates

---

## Output Format

For each finding, create a structured report:

```markdown
### [SEVERITY] Finding title

**Location:** `file.py:line_number`
**Category:** SQL Injection / Path Traversal / Input Validation / etc.
**Risk:** HIGH / MEDIUM / LOW
**Exploitable:** Yes (remote) / Yes (local) / No (theoretical)

**Description:** What the vulnerability is and how it could be exploited.

**Current code:**
\`\`\`python
# the vulnerable code
\`\`\`

**Fix:**
\`\`\`python
# the fixed code
\`\`\`

**Verification:** How to verify the fix is correct.
```

After documenting all findings, fix them. Create a single PR with all fixes. Commit with:
```
fix(security): comprehensive SQL parameterization + input validation sweep

Addresses 20 SQL string concatenation findings from Semgrep scan plus
additional input validation, path traversal, and credential handling
improvements identified during full security review.
```

---

## Checklist

Before declaring the review complete, verify:

- [ ] Zero SQL string concatenation in production code (all parameterized)
- [ ] All MCP tool inputs validated (length, range, type)
- [ ] All file paths validated (no traversal)
- [ ] API keys never logged or included in error messages
- [ ] FTS5 queries cannot cause DoS
- [ ] All outbound HTTP uses TLS
- [ ] No pickle/torch.load without weights_only=True
- [ ] Dependency versions have upper bounds
- [ ] Trace/log directories have restrictive permissions
- [ ] Buffer files are pruned (verified, not just assumed)
- [ ] Thread safety verified for all shared globals
- [ ] Spawn cap enforced correctly in stop.py
