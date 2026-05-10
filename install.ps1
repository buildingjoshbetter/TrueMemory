# TrueMemory installer for Windows — https://github.com/buildingjoshbetter/TrueMemory
#
# One-line install (PowerShell):
#   irm https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.ps1 | iex
#
# What this does:
#   1. Installs uv (Astral's Python tool manager) if missing.
#   2. Fetches a managed Python 3.12 (system Python untouched).
#   3. Installs truememory as an isolated uv tool.
#   4. Runs truememory-mcp --setup to auto-configure Claude Code / Claude Desktop.
#   5. Runs truememory-ingest install to wire up lifecycle hooks.
#   6. Pre-downloads all tier models (Edge + Base + Pro).
#
# Environment overrides:
#   $env:TRUEMEMORY_PY = "3.12"         # pin a specific Python (default: 3.12)
#   $env:TRUEMEMORY_SOURCE = "..."      # install from a local path instead of PyPI
#   $env:TRUEMEMORY_SKIP_SETUP = "1"    # skip the Claude auto-config step
#
# Safety:
#   - No admin/elevation required. Everything lands under $env:LOCALAPPDATA.
#   - Source: https://github.com/buildingjoshbetter/TrueMemory/blob/main/install.ps1

$ErrorActionPreference = "Stop"

# ---------- pretty output helpers ----------
function Say($msg)  { Write-Host "[truememory] $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "[truememory] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[truememory] $msg" -ForegroundColor Red }
function Die($msg)  { Warn "error: $msg"; exit 1 }

# ---------- main ----------
$TRUEMEMORY_PY = if ($env:TRUEMEMORY_PY) { $env:TRUEMEMORY_PY } else { "3.12" }
$TRUEMEMORY_SOURCE = if ($env:TRUEMEMORY_SOURCE) { $env:TRUEMEMORY_SOURCE } else { "" }

if ($TRUEMEMORY_PY -notmatch '^\d+\.\d+$') {
    Die "invalid TRUEMEMORY_PY: '$TRUEMEMORY_PY' (expected digits and dots, e.g. 3.12)"
}

if ($TRUEMEMORY_SOURCE) {
    $PKG_SPEC = $TRUEMEMORY_SOURCE
    Say "using custom source: $TRUEMEMORY_SOURCE"
} else {
    $PKG_SPEC = "truememory"
}

# ---------- step 1: install uv if missing ----------
$uvPath = Get-Command uv -ErrorAction SilentlyContinue
if ($uvPath) {
    $uvVer = & uv --version 2>$null
    Say "uv already installed ($uvVer)"
} else {
    Say "installing uv (Astral) — https://docs.astral.sh/uv/"
    try {
        irm https://astral.sh/uv/install.ps1 | iex
    } catch {
        Die "uv install failed — try: irm https://astral.sh/uv/install.ps1 | iex"
    }
    # Refresh PATH so we can find uv
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $uvPath = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvPath) {
        Die "uv installed but not on PATH — close and reopen PowerShell, then re-run this script"
    }
}

# ---------- step 2: ensure Python is available ----------
Say "fetching managed Python $TRUEMEMORY_PY (system Python untouched)..."
& uv python install $TRUEMEMORY_PY > $null
if ($LASTEXITCODE -ne 0) {
    Die "failed to install managed Python $TRUEMEMORY_PY"
}

# ---------- step 3: install truememory as a uv tool ----------
Say "installing $PKG_SPEC (~3-5 min on first run, downloads all tier models)..."
& uv tool uninstall truememory *> $null
& uv tool install --python $TRUEMEMORY_PY --force --refresh $PKG_SPEC > $null
if ($LASTEXITCODE -ne 0) {
    Die "truememory install failed"
}

# Add uv tool bin dir to PATH for future sessions
Say "adding uv's tool dir to your PATH (reversible)..."
& uv tool update-shell *> $null

# Refresh PATH and add tool Scripts dir for this session
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$uvToolDir = & uv tool dir 2>$null
if ($uvToolDir) {
    $scriptsDir = Join-Path $uvToolDir "truememory\Scripts"
    if ($scriptsDir -and (Test-Path $scriptsDir)) {
        $env:Path = "$scriptsDir;$env:Path"
    }
}

# ---------- step 4: auto-configure Claude ----------
if ($env:TRUEMEMORY_SKIP_SETUP -eq "1") {
    Say "skipping Claude setup (TRUEMEMORY_SKIP_SETUP=1)"
} else {
    Say "configuring Claude Code / Claude Desktop..."
    & truememory-mcp --setup
    if ($LASTEXITCODE -ne 0) {
        Warn "auto-setup returned non-zero (you can re-run it with: truememory-mcp --setup)"
    }

    Say "installing hooks and CLAUDE.md instructions..."
    & truememory-ingest install
    if ($LASTEXITCODE -ne 0) {
        Warn "hook install returned non-zero (you can re-run it with: truememory-ingest install)"
    }
}

# ---------- step 5: pre-download models for all tiers ----------
Say "pre-downloading models for all tiers (Edge + Base + Pro)..."
Say "  this takes 2-5 min but means tier switching just works afterward."

$toolPython = Join-Path (& uv tool dir 2>$null) "truememory\Scripts\python.exe"
if (Test-Path $toolPython) {
    Say "  [1/3] Edge reranker (MiniLM-L-6-v2, ~22MB)..."
    & $toolPython -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
    if ($LASTEXITCODE -eq 0) { Ok "  [1/3] Edge reranker ready" }
    else { Warn "  [1/3] Edge reranker download failed (search still works without it)" }

    Say "  [2/3] Base/Pro embedder (Qwen3-Embedding-0.6B, ~1.2GB)..."
    & $toolPython -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', truncate_dim=256)"
    if ($LASTEXITCODE -eq 0) { Ok "  [2/3] Base/Pro embedder ready" }
    else { Warn "  [2/3] Base/Pro embedder download failed (you can retry later or use Edge tier)" }

    Say "  [3/3] Base/Pro reranker (gte-modernbert, ~600MB)..."
    & $toolPython -c "from sentence_transformers import CrossEncoder; CrossEncoder('Alibaba-NLP/gte-reranker-modernbert-base')"
    if ($LASTEXITCODE -eq 0) { Ok "  [3/3] Base/Pro reranker ready" }
    else { Warn "  [3/3] Base/Pro reranker download failed (you can retry later or use Edge tier)" }

    Ok "all models pre-downloaded — tier switching is instant."
} else {
    Warn "could not locate tool Python at $toolPython — skipping model pre-download"
    Warn "models will download on first use instead"
}

# ---------- done ----------
Write-Host ""
Write-Host @"
████████╗██████╗ ██╗   ██╗███████╗    ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗
╚══██╔══╝██╔══██╗██║   ██║██╔════╝    ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝
   ██║   ██████╔╝██║   ██║█████╗      ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝
   ██║   ██╔══██╗██║   ██║██╔══╝      ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝
   ██║   ██║  ██║╚██████╔╝███████╗    ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝
                                  a sauron company
"@ -ForegroundColor Green

$installedVer = & $toolPython -c "from importlib.metadata import version; print(version('truememory'))" 2>$null
if (-not $installedVer) { $installedVer = "unknown" }
Write-Host ""
Ok "TrueMemory v$installedVer installed successfully."
Write-Host ""
Write-Host "  First time? Start a new Claude session and type:" -ForegroundColor Green
Write-Host ""
Write-Host "    Set up TrueMemory" -ForegroundColor Green -NoNewline
Write-Host ""
Write-Host ""
Write-Host "  TrueMemory will walk you through choosing Edge, Base, or Pro."
Write-Host ""
Write-Host "  IMPORTANT — if Claude Desktop was already open:" -ForegroundColor Yellow
Write-Host "    Close it completely and reopen it."
Write-Host "    A new chat window is NOT enough — the config only loads at launch."
Write-Host ""
Write-Host "  Commands:" -ForegroundColor Green
Write-Host "    truememory-mcp --setup              " -NoNewline; Write-Host "# re-run Claude auto-config" -ForegroundColor DarkGray
Write-Host "    truememory-ingest install            " -NoNewline; Write-Host "# re-install hooks" -ForegroundColor DarkGray
Write-Host "    uv tool upgrade truememory     " -NoNewline; Write-Host "# update to latest" -ForegroundColor DarkGray
Write-Host "    uv tool uninstall truememory   " -NoNewline; Write-Host "# uninstall" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Note:" -ForegroundColor Yellow -NoNewline
Write-Host " If commands are not found, close and reopen PowerShell."
Write-Host ""
