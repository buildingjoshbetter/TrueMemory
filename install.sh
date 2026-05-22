#!/bin/sh
# TrueMemory installer — https://github.com/buildingjoshbetter/TrueMemory
#
# One-line install:
#   curl -LsSf https://raw.githubusercontent.com/buildingjoshbetter/TrueMemory/main/install.sh | sh
#
# What this does:
#   1. Installs uv (Astral's Python tool manager) if missing — uv brings its own
#      Python runtime, so your system Python is never touched.
#   2. Fetches a managed Python 3.12 into ~/.local/share/uv/python/.
#   3. Installs truememory as an isolated uv tool.
#   4. Runs `truememory-mcp --setup` (code from PyPI) to auto-configure
#      Claude Code and/or Claude Desktop. Set TRUEMEMORY_SKIP_SETUP=1 to skip.
#   5. Runs `truememory-ingest install` to wire up lifecycle hooks
#      (SessionStart, Stop, UserPromptSubmit, PreCompact) and merge
#      CLAUDE.md instructions so Claude uses TrueMemory proactively.
#
# Environment overrides:
#   TRUEMEMORY_PY=3.12         # pin a specific Python (default: 3.12)
#   TRUEMEMORY_EXTRAS=          # (deprecated — gpu extras are now installed by default)
#   TRUEMEMORY_SOURCE=...      # install from a local path or git URL instead of PyPI
#                            # (useful for testing: TRUEMEMORY_SOURCE=/path/to/truememory)
#   TRUEMEMORY_SKIP_SETUP=1    # skip the Claude auto-config step
#
# Safety:
#   - No sudo required. Everything lands under $HOME.
#   - The script body is wrapped in a main() function, so a mid-download
#     network drop cannot execute partial logic — the file must parse
#     completely before anything runs.
#   - Source: https://github.com/buildingjoshbetter/TrueMemory/blob/main/install.sh
#     Read it first if you want: curl -LsSf <URL> -o install.sh && less install.sh

# ---------- pretty output helpers ----------
if [ -t 1 ]; then
  BLUE='\033[1;36m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'
  RED='\033[1;31m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'
else
  BLUE=''; GREEN=''; YELLOW=''; RED=''; BOLD=''; DIM=''; RESET=''
fi
say()  { printf '%b[truememory]%b %s\n' "$BLUE"  "$RESET" "$*"; }
ok()   { printf '%b[truememory]%b %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%b[truememory]%b %s\n' "$RED"   "$RESET" "$*" >&2; }
die()  { warn "error: $*"; exit 1; }

# ---------- main ----------
main() {
  set -euo pipefail

  TRUEMEMORY_PY="${TRUEMEMORY_PY:-3.12}"
  TRUEMEMORY_EXTRAS="${TRUEMEMORY_EXTRAS:-}"
  TRUEMEMORY_SOURCE="${TRUEMEMORY_SOURCE:-}"

  # Defend against hostile env vars (e.g. a malicious "paste this" blog post).
  case "$TRUEMEMORY_PY" in
    ''|*[!0-9.]*)
      die "invalid TRUEMEMORY_PY: '$TRUEMEMORY_PY' (expected digits and dots, e.g. 3.12)" ;;
  esac
  case "$TRUEMEMORY_EXTRAS" in
    *[!a-zA-Z0-9,_-]*)
      die "invalid TRUEMEMORY_EXTRAS: '$TRUEMEMORY_EXTRAS' (expected names like 'mcp' or 'gpu,mcp')" ;;
  esac

  if [ -n "$TRUEMEMORY_SOURCE" ]; then
    PKG_SPEC="${TRUEMEMORY_SOURCE}"
    say "using custom source: $TRUEMEMORY_SOURCE"
  else
    PKG_SPEC="truememory"
  fi

  # ---------- preflight ----------
  command -v curl >/dev/null 2>&1 || die "curl is required but not found on PATH"

  case "$(uname -s)" in
    Darwin|Linux) ;;
    *) die "unsupported OS: $(uname -s) — installer supports Mac and Linux. See README for Windows." ;;
  esac

  # Make sure common install dirs are on PATH for THIS shell so we can find
  # uv even if the user already has it but hasn't restarted their terminal.
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

  # ---------- step 1: install uv if missing ----------
  if command -v uv >/dev/null 2>&1; then
    say "uv already installed ($(uv --version 2>/dev/null || echo unknown))"
  else
    say "installing uv (Astral) — https://docs.astral.sh/uv/"
    # Astral's official installer — trusted source, same curl|sh pattern.
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || \
      die "uv install failed — try: curl -LsSf https://astral.sh/uv/install.sh | sh"
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || \
      die "uv installed but not on PATH — restart your shell and re-run this script"
  fi

  # ---------- step 2: ensure Python TRUEMEMORY_PY is available ----------
  say "fetching managed Python $TRUEMEMORY_PY (system Python untouched, ~30s on first run)..."
  # stderr is NOT suppressed — you see uv's progress output so a slow download
  # doesn't look like a frozen terminal.
  uv python install "$TRUEMEMORY_PY" >/dev/null || \
    die "failed to install managed Python $TRUEMEMORY_PY (see error above)"

  # ---------- step 3: install truememory as a uv tool ----------
  say "installing $PKG_SPEC (~3-5 min on first run, downloads all tier models)..."
  # Remove any existing install first to guarantee a clean slate.
  # Without this, uv may serve a cached older version even with --refresh.
  uv tool uninstall truememory >/dev/null 2>&1 || true
  # --force makes re-runs idempotent. --python pins the interpreter to avoid
  # astral-sh/uv#14110. --refresh bypasses the resolver cache.
  uv tool install --python "$TRUEMEMORY_PY" --force --refresh "$PKG_SPEC" >/dev/null || \
    die "truememory install failed (see error above)"

  # Future shells should see ~/.local/bin. Reversible via 'uv tool update-shell --uninstall'.
  say "adding uv's tool dir to your shell rc (reversible)..."
  uv tool update-shell >/dev/null 2>&1 || true

  # Resolve tool venv python for steps 4 and 5 (avoids Windows Defender ASR shim blocks)
  TOOL_PYTHON="$(uv tool dir)/truememory/bin/python"

  # ---------- step 4: auto-configure Claude ----------
  if [ "${TRUEMEMORY_SKIP_SETUP:-}" = "1" ]; then
    say "skipping Claude setup (TRUEMEMORY_SKIP_SETUP=1)"
  elif [ ! -x "$TOOL_PYTHON" ]; then
    warn "could not locate tool venv python at $TOOL_PYTHON — skipping Claude setup."
    warn "Re-run manually: python -m truememory.mcp_server --setup"
  else
    say "configuring Claude Code / Claude Desktop..."
    "$TOOL_PYTHON" -m truememory.mcp_server --setup || \
      warn "auto-setup returned non-zero (re-run: python -m truememory.mcp_server --setup)"

    say "installing hooks and CLAUDE.md instructions..."
    "$TOOL_PYTHON" -m truememory.ingest.cli install || \
      warn "hook install returned non-zero (re-run: python -m truememory.ingest.cli install)"
  fi

  # ---------- step 5: pre-download models for all tiers ----------
  say "pre-downloading models for all tiers (Edge + Base + Pro)..."
  say "  this takes 2-5 min but means tier switching just works afterward."
  say "  you'll see download progress bars below."
  # Use the tool's Python (resolved in step 4) to run downloads inside the uv venv.
  if [ -x "$TOOL_PYTHON" ]; then
    # Edge: Model2Vec embedder (usually bundled) + MiniLM reranker
    say "  [1/3] Edge reranker (MiniLM-L-6-v2, ~22MB)..."
    "$TOOL_PYTHON" -c "
from sentence_transformers import CrossEncoder
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
" && ok "  [1/3] Edge reranker ready" || warn "  [1/3] Edge reranker download failed (search still works without it)"

    # Base/Pro: Qwen3 embedder
    say "  [2/3] Base/Pro embedder (Qwen3-Embedding-0.6B, ~1.2GB)..."
    "$TOOL_PYTHON" -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', truncate_dim=256)
" && ok "  [2/3] Base/Pro embedder ready" || warn "  [2/3] Base/Pro embedder download failed (you can retry later or use Edge tier)"

    # Base/Pro: gte-reranker
    say "  [3/3] Base/Pro reranker (gte-modernbert, ~600MB)..."
    "$TOOL_PYTHON" -c "
from sentence_transformers import CrossEncoder
CrossEncoder('Alibaba-NLP/gte-reranker-modernbert-base')
" && ok "  [3/3] Base/Pro reranker ready" || warn "  [3/3] Base/Pro reranker download failed (you can retry later or use Edge tier)"

    ok "all models pre-downloaded — tier switching is instant."
  else
    warn "could not locate tool Python at $TOOL_PYTHON — skipping model pre-download"
    warn "models will download on first use instead"
  fi

  # ---------- done ----------
  printf '\n'
  printf '%b' "$GREEN"
  cat << 'BANNER'
████████╗██████╗ ██╗   ██╗███████╗    ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗   ██╗
╚══██╔══╝██╔══██╗██║   ██║██╔════╝    ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗ ██╔╝
   ██║   ██████╔╝██║   ██║█████╗      ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ╚████╔╝
   ██║   ██╔══██╗██║   ██║██╔══╝      ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗  ╚██╔╝
   ██║   ██║  ██║╚██████╔╝███████╗    ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║   ██║
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝
                                  a sauron company
BANNER
  printf '%b' "$RESET"
  printf '\n'
  # Show installed version
  INSTALLED_VER=$("$TOOL_PYTHON" -c "from importlib.metadata import version; print(version('truememory'))" 2>/dev/null || echo "unknown")
  ok "TrueMemory v${INSTALLED_VER} installed successfully."
  printf '\n'
  printf '  %bFirst time?%b Start a new Claude session and type:\n' "$GREEN" "$RESET"
  printf '\n'
  printf '    %b%bSet up TrueMemory%b\n' "$BOLD" "$GREEN" "$RESET"
  printf '\n'
  printf '  TrueMemory will walk you through choosing Edge, Base, or Pro.\n'
  printf '\n'
  printf '  %b%bIMPORTANT — if Claude Desktop was already open:%b\n' "$YELLOW" "$BOLD" "$RESET"
  printf '    Quit it completely with %bCmd+Q%b and reopen it.\n' "$BOLD" "$RESET"
  printf '    A new chat window is NOT enough — the config only loads at launch.\n'
  printf '\n'
  printf '  %bCommands:%b\n' "$GREEN" "$RESET"
  printf '    truememory-mcp --setup              %b# re-run Claude auto-config%b\n' "$DIM" "$RESET"
  printf '    truememory-ingest install            %b# re-install hooks%b\n' "$DIM" "$RESET"
  printf '    uv tool upgrade truememory     %b# update to latest%b\n' "$DIM" "$RESET"
  printf '    uv tool uninstall truememory   %b# uninstall%b\n' "$DIM" "$RESET"
  printf '\n'
  printf '  %bNote:%b If commands are not found, open a new terminal window\n' "$YELLOW" "$RESET"
  printf '        or run: %bsource ~/.zshrc%b  (or %bsource ~/.bashrc%b)\n' "$BOLD" "$RESET" "$BOLD" "$RESET"
  printf '\n'
}

main "$@"
