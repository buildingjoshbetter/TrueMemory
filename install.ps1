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

# ---------- execution policy check ----------
$policy = Get-ExecutionPolicy -Scope CurrentUser
if ($policy -eq "Restricted" -or $policy -eq "Undefined") {
    $machinePolicy = Get-ExecutionPolicy -Scope LocalMachine
    if ($machinePolicy -eq "Restricted" -or $machinePolicy -eq "Undefined") {
        Warn "PowerShell execution policy is '$policy'. Scripts may be blocked."
        Warn "Run this to fix: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
        Warn "Then re-run the installer."
        exit 1
    }
}

# ---------- main ----------
$stepsDone = 0
function Step-Done { $script:stepsDone++ }

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
Say "installing $PKG_SPEC (~3-5 min on first run)..."
& uv tool uninstall truememory *> $null
if ($LASTEXITCODE -gt 1) {
    Warn "uv tool uninstall returned $LASTEXITCODE — proceeding, but result may be partial"
}
& uv tool install --python $TRUEMEMORY_PY --force --refresh "$PKG_SPEC" > $null
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

# Resolve tool venv python early (avoids Windows Defender ASR shim blocks)
$toolPython = $null
if ($uvToolDir) {
    $candidate = Join-Path $uvToolDir "truememory\Scripts\python.exe"
    if (Test-Path $candidate) { $toolPython = $candidate }
}

# ---------- step 4: auto-configure Claude ----------
$setupStatus = "skipped"
$hookStatus = "skipped"
if ($env:TRUEMEMORY_SKIP_SETUP -eq "1") {
    Say "skipping Claude setup (TRUEMEMORY_SKIP_SETUP=1)"
} elseif (-not $toolPython) {
    Warn "could not locate tool venv python — skipping Claude setup."
    Warn "Re-run manually: python -m truememory.mcp_server --setup"
} else {
    Say "configuring Claude Code / Claude Desktop..."
    & $toolPython -m truememory.mcp_server --setup
    if ($LASTEXITCODE -eq 0) { $setupStatus = "command succeeded" }
    else {
        $setupStatus = "failed"
        Warn "auto-setup returned non-zero (re-run: python -m truememory.mcp_server --setup)"
    }

    Say "installing hooks and CLAUDE.md instructions..."
    & $toolPython -m truememory.ingest.cli install
    if ($LASTEXITCODE -eq 0) { $hookStatus = "command succeeded" }
    else {
        $hookStatus = "failed"
        Warn "hook install returned non-zero (re-run: python -m truememory.ingest.cli install)"
    }
}

# ---------- step 5: pre-download models for all tiers ----------
Say "pre-downloading the Edge reranker and Base/Pro models..."
Say "  this can take 2-5 min; uncached models require network access."
$modelChecks = @(
    @{ Label = "Edge reranker"; Status = "not checked"; Code = "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" },
    @{ Label = "Base/Pro embedder"; Status = "not checked"; Code = "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', truncate_dim=256)" },
    @{ Label = "Base/Pro reranker"; Status = "not checked"; Code = "from sentence_transformers import CrossEncoder; CrossEncoder('Alibaba-NLP/gte-reranker-modernbert-base')" }
)
if ($toolPython -and (Test-Path -LiteralPath $toolPython)) {
    foreach ($modelCheck in $modelChecks) {
        Say "  Pre-downloading $($modelCheck.Label)..."
        $modelCheck.Status = "failed"
        try {
            & $toolPython -c $modelCheck.Code
            if ($LASTEXITCODE -eq 0) { $modelCheck.Status = "ready" }
        } catch {
            Warn "  $($modelCheck.Label) could not be loaded: $_"
        }
        if ($modelCheck.Status -eq "ready") {
            Ok "  $($modelCheck.Label) ready"
        } else {
            Warn "  $($modelCheck.Label) pre-download failed"
            Warn "Retry this model only: & (Join-Path (uv tool dir) 'truememory\Scripts\python.exe') -c `"$($modelCheck.Code)`""
        }
    }
} else {
    Warn "could not locate tool Python at $toolPython — skipping model pre-download"
    Warn "Model readiness is unverified. Locate the tool environment with: uv tool dir"
    Warn "After locating its Python, run: python -m truememory.ingest.cli setup"
}
$modelReadyCount = @($modelChecks | Where-Object { $_.Status -eq "ready" }).Count

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

$installedVer = $null
if ($toolPython) {
    try {
        $installedVer = & $toolPython -c "from importlib.metadata import version; print(version('truememory'))" 2>$null
    } catch {
        $installedVer = $null
    }
}
if (-not $installedVer) { $installedVer = "unknown" }
Write-Host ""
Ok "TrueMemory v$installedVer package installed."
Say "Claude registration: $setupStatus"
Say "Hooks: $hookStatus"
Say "Model pre-download checks: $modelReadyCount/3 succeeded."
foreach ($modelCheck in $modelChecks) {
    Say "  $($modelCheck.Label): $($modelCheck.Status)"
}
if ($modelReadyCount -eq 3) {
    Ok "All three requested model pre-download checks passed."
} else {
    Warn "Model readiness incomplete; package installation is retained. Retry the failed or skipped checks before relying on those models."
}
Say "The Edge embedder was not checked by this pre-download step."
Say "Registration and hook command success do not verify runtime operation."
Say "Changing embedding models can still require re-embedding stored memories."
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
