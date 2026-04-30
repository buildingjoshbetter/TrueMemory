"""
CLI entry point for truememory-ingest.

Usage:
    truememory-ingest /path/to/transcript.json --user alice
    truememory-ingest --install   # Install hooks into Claude Code settings
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from truememory.ingest import ingest, save_trace, IngestionResult
from truememory.ingest.models import LLMConfig, hydrate_config


def main():
    from truememory import __version__ as _tm_version
    parser = argparse.ArgumentParser(
        description="TrueMemory Ingestion — biomimetic memory encoding",
        prog="truememory-ingest",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"truememory-ingest {_tm_version}",
    )
    sub = parser.add_subparsers(dest="command")

    # --- ingest command ---
    p_ingest = sub.add_parser("ingest", help="Ingest a conversation transcript")
    p_ingest.add_argument("transcript", help="Path to transcript file")
    p_ingest.add_argument("--user", default="", help="User ID for memory scoping")
    p_ingest.add_argument("--db", default=None, help="Path to truememory database")
    p_ingest.add_argument("--threshold", type=float, default=0.30, help="Encoding gate threshold (0-1)")
    p_ingest.add_argument("--trace", default=None, help="Save decision trace to file")
    p_ingest.add_argument("--provider", default="auto", help="LLM provider (auto/ollama/claude_cli/openrouter/anthropic)")
    p_ingest.add_argument("--model", default="", help="LLM model name")
    p_ingest.add_argument("--session", default="", help="Session identifier to tag this ingestion trace with")
    p_ingest.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # --- install command ---
    p_install = sub.add_parser("install", help="Install hooks into Claude Code settings")
    p_install.add_argument("--user", default="", help="User ID for memory scoping")
    p_install.add_argument("--db", default=None, help="Path to truememory database")
    p_install.add_argument("--dry-run", action="store_true", help="Print settings without writing")

    # --- stats command ---
    p_stats = sub.add_parser("stats", help="Show ingestion statistics from a trace file")
    p_stats.add_argument("trace_file", help="Path to trace JSON file")

    # --- status command ---
    _p_status = sub.add_parser("status", help="Check whether ingestion is set up correctly")

    # --- uninstall command ---
    p_uninstall = sub.add_parser("uninstall", help="Remove truememory-ingest hooks from Claude Code settings")
    p_uninstall.add_argument("--dry-run", action="store_true", help="Show what would be removed without writing")

    # --- logs command ---
    p_logs = sub.add_parser("logs", help="Tail recent hook log files from ~/.truememory/logs/")
    p_logs.add_argument("--tail", type=int, default=50, help="Number of lines to show from the most recent log")
    p_logs.add_argument("--session", default="", help="Specific session ID to view (default: most recent)")
    p_logs.add_argument("--list", action="store_true", help="List available log files instead of tailing")

    # --- trace command ---
    p_trace = sub.add_parser("trace", help="Show the decision trace for a session")
    p_trace.add_argument("session", nargs="?", default="", help="Session ID (default: most recent)")
    p_trace.add_argument("--raw", action="store_true", help="Print raw JSON instead of formatted output")

    # --- facts command ---
    p_facts = sub.add_parser("facts", help="Show facts stored during a session (per-fact decisions)")
    p_facts.add_argument("session", nargs="?", default="", help="Session ID (default: most recent)")
    p_facts.add_argument("--all", action="store_true", help="Include facts that were skipped by the gate")
    p_facts.add_argument("--category", default="", help="Filter by category (personal/preference/decision/etc.)")

    # --- setup command (first-time onboarding) ---
    p_setup = sub.add_parser("setup", help="Interactive first-time setup wizard")
    p_setup.add_argument("--non-interactive", action="store_true", help="Skip prompts, use defaults + env vars")

    args = parser.parse_args()

    if args.command == "ingest":
        _run_ingest(args)
    elif args.command == "install":
        _run_install(args)
    elif args.command == "stats":
        _run_stats(args)
    elif args.command == "status":
        _run_status(args)
    elif args.command == "uninstall":
        _run_uninstall(args)
    elif args.command == "logs":
        _run_logs(args)
    elif args.command == "trace":
        _run_trace(args)
    elif args.command == "facts":
        _run_facts(args)
    elif args.command == "setup":
        _run_setup(args)
    else:
        parser.print_help()


def _run_ingest(args):
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    # Preflight: verify truememory is importable before we try to
    # construct a pipeline. This catches missing dependencies early with
    # an actionable error instead of a deep ModuleNotFoundError.
    try:
        import truememory  # noqa: F401
    except ImportError:
        print(
            "ERROR: truememory is not installed.\n"
            "       Install with: pip install truememory\n"
            "       (truememory-ingest depends on truememory>=0.1.3)",
            file=sys.stderr,
        )
        sys.exit(2)

    # Verify the transcript file exists AND is readable before spinning up
    # the pipeline. os.access catches chmod 000 / unreadable cases that
    # Path.exists() would happily let through (see Bug #1).
    import os
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"ERROR: transcript file not found: {args.transcript}", file=sys.stderr)
        sys.exit(2)
    if not os.access(transcript_path, os.R_OK):
        print(
            f"ERROR: transcript file not readable: {args.transcript}",
            file=sys.stderr,
        )
        sys.exit(2)

    # ---- Filesystem preflight (Bug #4) ----
    # Verify that the DB path and trace path are writable BEFORE we spend
    # any LLM calls on extraction. Bug #4 was that the pipeline would crash
    # with "unable to open database file" after the expensive extraction
    # call had already run — that's real money on paid providers. Exit 4 is
    # reserved for preflight filesystem failures so callers can distinguish
    # them from "bad args" (2) and "no LLM backend" (3).
    if not _preflight_writable_target(args.db, kind="db"):
        sys.exit(4)
    if args.trace is not None and not _preflight_writable_target(args.trace, kind="trace"):
        sys.exit(4)

    config = None
    if args.provider != "auto":
        # Hydrate provider defaults (api_key from env, base_url, default
        # model) so `--provider anthropic` works without forcing the user
        # to also pass `--model` and figure out how to inject a key.
        config = hydrate_config(LLMConfig(provider=args.provider, model=args.model))
        if not config.api_key and config.provider in ("anthropic", "openrouter", "openai"):
            print(
                f"ERROR: --provider {config.provider} requires an API key.\n"
                f"       Set the corresponding env var "
                f"({config.provider.upper()}_API_KEY) before running.",
                file=sys.stderr,
            )
            sys.exit(3)
        # Improvement A: fail fast (exit 3) when --provider claude_cli is
        # requested but the `claude` binary isn't on PATH. Previously we
        # would proceed, let extract_facts return [] due to LLMError, and
        # exit 0 — which made shell scripts think the ingest succeeded.
        if config.provider == "claude_cli":
            from truememory.ingest.models import _claude_cli_available
            if not _claude_cli_available():
                print(
                    "ERROR: --provider claude_cli requested but `claude` CLI is not on PATH.\n"
                    "       Install Claude Code or choose a different --provider.",
                    file=sys.stderr,
                )
                sys.exit(3)

    try:
        result = ingest(
            transcript_path=args.transcript,
            user_id=args.user,
            db_path=args.db,
            gate_threshold=args.threshold,
            llm_config=config,
            session_id=args.session,
        )
    except RuntimeError as e:
        # auto_detect raises RuntimeError when no LLM backend is available
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(3)

    _print_result(result)

    if args.trace:
        # save_trace never raises now — it logs a warning and returns False
        # on failure so the ingestion's exit code isn't contaminated by a
        # diagnostic write failure (Bug #3).
        save_trace(result, args.trace)


def _preflight_writable_target(target: str | None, *, kind: str) -> bool:
    """Verify that a file path's parent exists (or can be created) and is writable.

    Returns True on success, prints an ERROR to stderr and returns False
    on failure. Shared between the DB and trace preflight so they have
    identical error semantics.
    """
    if not target:
        # None / empty string → nothing to preflight (library default will
        # be used, e.g. ~/.truememory/memories.db which truememory owns).
        return True

    import os
    path = Path(target).expanduser()
    parent = path.parent if path.parent != Path("") else Path(".")

    # Try to create the parent dir if it's missing.
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        print(
            f"ERROR: cannot create {kind} directory {parent}: {e}",
            file=sys.stderr,
        )
        return False

    # Verify the parent is writable (catches chmod 555 and similar).
    # Use a temp-file probe for maximum correctness — os.access() lies on
    # some filesystems, but actually creating a file never does.
    probe = parent / f".truememory.ingest_preflight_{os.getpid()}"
    try:
        probe.write_text("x", encoding="utf-8")
    except (PermissionError, OSError) as e:
        print(
            f"ERROR: {kind} directory {parent} is not writable: {e}",
            file=sys.stderr,
        )
        return False
    finally:
        try:
            if probe.exists():
                probe.unlink()
        except OSError:
            pass

    # For DBs, also confirm we can actually open/create the sqlite file.
    # This catches corrupted existing DBs, bad permissions on an existing
    # file, and any other OperationalError that would surface later deep
    # in the storage layer — BEFORE we spend an LLM call.
    if kind == "db":
        import sqlite3
        try:
            conn = sqlite3.connect(str(path))
            conn.execute("PRAGMA user_version")
            conn.close()
        except sqlite3.OperationalError as e:
            print(
                f"ERROR: cannot open {kind} at {path}: {e}",
                file=sys.stderr,
            )
            return False

    return True


_SAURON_BANNER = """
\033[1;33m╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              ◉  T R U E M E M O R Y                          ║
║              Persistent Memory for AI Agents                 ║
║                                                              ║
║              A Sauron Company                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝\033[0m
"""

_TRUEMEMORY_CONFIG_PATH = Path.home() / ".truememory" / "config.json"


def _load_truememory_config() -> dict:
    """Load persistent config from ~/.truememory/config.json."""
    if _TRUEMEMORY_CONFIG_PATH.exists():
        try:
            return json.loads(_TRUEMEMORY_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_truememory_config(config: dict) -> None:
    """Save config to ~/.truememory/config.json.

    Hunter F28: chmod calls below are silent no-ops on Windows. When an
    API key is being persisted on Windows we warn to stderr so the user
    knows the plaintext file is readable by other local users and can
    route keys through env vars instead on shared machines.
    """
    _TRUEMEMORY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRUEMEMORY_CONFIG_PATH.parent.chmod(0o700)
    _TRUEMEMORY_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    _TRUEMEMORY_CONFIG_PATH.chmod(0o600)
    if sys.platform == "win32" and any(k.endswith("_api_key") for k in config):
        print(
            "truememory: warning — on Windows, ~/.truememory/config.json "
            "permissions are inherited from the parent directory and may be "
            "readable by other local users. If this is a shared machine, set "
            "the API key via the ANTHROPIC_API_KEY / OPENROUTER_API_KEY / "
            "OPENAI_API_KEY environment variable instead.",
            file=sys.stderr,
        )


def _run_setup(args):
    """Interactive first-time setup wizard for TrueMemory."""
    print(_SAURON_BANNER)
    print("Welcome to TrueMemory setup! Let's get you configured.\n")

    config = _load_truememory_config()

    # ── Step 1: Embedding tier ────────────────────────────────────────
    existing_tier = config.get("tier", "")
    print("  \033[1mEmbedding Tier\033[0m")
    print("  ─────────────")
    print("  [1] Edge — 90.1% LoCoMo, CPU-only, ~30MB install. Works anywhere.")
    print("  [2] Base — 91.5% LoCoMo, GPU recommended, ~1.5GB install. No API key needed.")
    print("  [3] Pro  — 91.8% LoCoMo, GPU recommended, ~1.5GB install. Requires LLM API key (HyDE).")
    print()

    _TIER_NUM = {"edge": "1", "base": "2", "pro": "3"}
    if args.non_interactive:
        tier = existing_tier or "edge"
    else:
        default_num = _TIER_NUM.get(existing_tier, "1")
        choice = input(f"  Choose tier [1/2/3] (default: {default_num}): ").strip() or default_num
        tier = {"1": "edge", "2": "base", "3": "pro"}.get(choice, "edge")

    if tier in ("base", "pro"):
        try:
            import sentence_transformers  # noqa: F401
            print(f"  \033[32m✓ {tier.capitalize()} dependencies already installed\033[0m")
        except ImportError:
            print(f"  \033[33m⚠ {tier.capitalize()} tier requires: pip install truememory[gpu]\033[0m")
            if not args.non_interactive:
                do_install = input("  Install now? [y/N]: ").strip().lower()
                if do_install == "y":
                    import subprocess
                    # Hunter F25: bound pip with a 10-minute timeout —
                    # long enough for model downloads from slow mirrors,
                    # short enough that a dead mirror doesn't wedge setup.
                    try:
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install",
                             "truememory[gpu]"],
                            timeout=600,
                        )
                    except subprocess.TimeoutExpired:
                        print(
                            "  \033[33m⚠ pip install timed out after "
                            "10 minutes. Try `pip install "
                            "truememory[gpu]` directly, then re-run "
                            "`truememory-ingest setup`.\033[0m",
                            file=sys.stderr,
                        )
                        print("  Falling back to Edge tier.")
                        tier = "edge"
                else:
                    print("  Falling back to Edge tier.")
                    tier = "edge"

    # Pre-download the embedding model so first search isn't slow
    print()
    print("  \033[1mDownloading embedding model...\033[0m")
    try:
        os.environ["TRUEMEMORY_EMBED_MODEL"] = tier
        from truememory.vector_search import set_embedding_model, get_model
        set_embedding_model(tier)
        get_model()  # triggers download if not cached
        if tier in ("base", "pro"):
            print("  \033[32m✓ Qwen3-Embedding-0.6B @ 256d Matryoshka ready\033[0m")
        else:
            print("  \033[32m✓ potion-base-8M (256-dim) ready\033[0m")
    except Exception as e:
        print(f"  \033[33m⚠ Model download failed: {e}\033[0m")
        print("  The model will download on first use instead.")

    # Also pre-download the cross-encoder reranker if available
    try:
        from truememory.reranker import get_reranker
        print("  Downloading reranker model...")
        get_reranker()
        print("  \033[32m✓ Cross-encoder reranker ready\033[0m")
    except Exception:
        pass  # Optional component, don't fail setup

    # ── Step 2: API key for HyDE / deep search ───────────────────────
    print()
    print("  \033[1mAPI Key for Deep Search (HyDE)\033[0m")
    print("  ────────────────────────────")
    print("  Deep search uses a small LLM to expand queries for better recall.")
    print("  Without a key, basic search still works — you just skip HyDE.")
    print()
    print("  Supported providers:")
    print("  [1] Anthropic  — Claude Haiku 4.5 (recommended)")
    print("  [2] OpenRouter — one key, many models")
    print("  [3] OpenAI     — GPT-4o-mini")
    print("  [4] Skip       — no API key, basic search only")
    print()

    # Check what's already configured
    has_anthropic = bool(config.get("anthropic_api_key"))
    has_openrouter = bool(config.get("openrouter_api_key"))
    has_openai = bool(config.get("openai_api_key"))

    if args.non_interactive:
        provider_choice = "4"
        if os.environ.get("ANTHROPIC_API_KEY") or has_anthropic:
            provider_choice = "1"
        elif os.environ.get("OPENROUTER_API_KEY") or has_openrouter:
            provider_choice = "2"
        elif os.environ.get("OPENAI_API_KEY") or has_openai:
            provider_choice = "3"
    else:
        default_provider = "4"
        if has_anthropic:
            default_provider = "1"
        elif has_openrouter:
            default_provider = "2"
        elif has_openai:
            default_provider = "3"

        provider_choice = input(f"  Choose provider [1/2/3/4] (default: {default_provider}): ").strip()
        if not provider_choice:
            provider_choice = default_provider

    api_key_field = None
    llm_provider = None
    api_key = ""

    if provider_choice in ("1", "2", "3"):
        _provider_map = {
            "1": ("anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY", "Anthropic API key (sk-ant-...)"),
            "2": ("openrouter", "openrouter_api_key", "OPENROUTER_API_KEY", "OpenRouter API key (sk-or-...)"),
            "3": ("openai", "openai_api_key", "OPENAI_API_KEY", "OpenAI API key (sk-...)"),
        }
        llm_provider, api_key_field, env_var, prompt_text = _provider_map[provider_choice]
        existing_key = config.get(api_key_field, "") or os.environ.get(env_var, "")

        if existing_key and not args.non_interactive:
            masked = existing_key[:8] + "..." + existing_key[-4:]
            print(f"  Found existing key: {masked}")
            use_existing = input("  Use this key? [Y/n]: ").strip().lower()
            if use_existing != "n":
                api_key = existing_key
            else:
                # getpass: don't echo the key to the terminal or shell history.
                from getpass import getpass as _getpass
                api_key = _getpass(f"  {prompt_text}: ").strip()
        elif existing_key:
            api_key = existing_key
        elif not args.non_interactive:
            from getpass import getpass as _getpass
            api_key = _getpass(f"  {prompt_text}: ").strip()

    # ── Save config ───────────────────────────────────────────────────
    config["tier"] = tier
    if llm_provider:
        config["llm_provider"] = llm_provider
    if api_key_field and api_key:
        config[api_key_field] = api_key

    _save_truememory_config(config)
    print()
    print(f"  \033[32m✓ Config saved to {_TRUEMEMORY_CONFIG_PATH}\033[0m")

    # ── Step 3: Install hooks ─────────────────────────────────────────
    print()
    if not args.non_interactive:
        do_hooks = input("  Install Claude Code hooks now? [Y/n]: ").strip().lower()
    else:
        do_hooks = "y"

    if do_hooks != "n":
        class _InstallArgs:
            user = config.get("user_id", "")
            db = None
            dry_run = False
        _run_install(_InstallArgs())

    # ���─ Done ──────────────────────────────────────────────────────────
    print()
    print("  \033[1;33m══════════════════════════════════════════════\033[0m")
    print("  \033[1mSetup complete!\033[0m")
    print(f"  Tier:     {tier}")
    print(f"  Provider: {llm_provider or 'none (basic search only)'}")
    print()
    print("  Run \033[1mtruememory-ingest status\033[0m to verify everything.")
    print()
    print("  \033[2mThanks for using TrueMemory, a Sauron company.\033[0m")
    print()


def _run_install(args):
    """Install all 4 Claude Code hooks for full lifecycle coverage.

    Hooks installed:
    - SessionStart: injects relevant memories as additionalContext
    - UserPromptSubmit: buffers user messages (kept for future use)
    - Stop: triggers background fact extraction after each session
    - PreCompact: saves context snapshot before Claude compresses conversation

    Note: the event is named ``PreCompact`` in Claude Code's settings.json
    schema — earlier versions of this installer registered ``compact`` which
    was silently ignored by Claude Code (the hook never fired). See
    https://code.claude.com/docs/en/hooks for the canonical event names.
    """
    # Hooks now live inside the package namespace so the wheel ships them
    # cleanly and ``import hooks`` doesn't collide with unrelated user code.
    hooks_dir = Path(__file__).parent / "hooks"

    # Verify all hook files exist before writing anything.
    # Event names (keys) MUST match Claude Code's canonical hook event names
    # exactly — any other value is silently ignored by the runtime.
    hook_files = {
        "SessionStart": hooks_dir / "session_start.py",
        "UserPromptSubmit": hooks_dir / "user_prompt_submit.py",
        "Stop": hooks_dir / "stop.py",
        "PreCompact": hooks_dir / "compact.py",
    }
    missing = [name for name, path in hook_files.items() if not path.exists()]
    if missing:
        print(f"ERROR: Missing hook files: {', '.join(missing)}", file=sys.stderr)
        print(f"Looked in: {hooks_dir}", file=sys.stderr)
        sys.exit(1)

    # Build the hook command list with proper shell quoting so paths
    # containing spaces (e.g. ``/Users/Jane Doe/.venv/bin/python``) survive
    # being embedded into a shell command string. Each hook script accepts
    # ``--user`` and ``--db`` flags via argparse so these args are not dead
    # code — they override the env var fallbacks.
    import shlex
    py = sys.executable

    def _build_command(hook_path: Path) -> str:
        parts: list[str] = [py, str(hook_path)]
        if args.user:
            parts.extend(["--user", args.user])
        if args.db:
            parts.extend(["--db", args.db])
        return " ".join(shlex.quote(p) for p in parts)

    settings = {
        "hooks": {
            event: [{
                "matcher": "",
                "hooks": [{
                    "type": "command",
                    "command": _build_command(path),
                }],
            }]
            for event, path in hook_files.items()
        }
    }

    settings_json = json.dumps(settings, indent=2)

    if args.dry_run:
        print("Add the following to your Claude Code settings.json:\n")
        print(settings_json)
        return

    # Merge into existing settings (preserves other config)
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text())
        except json.JSONDecodeError as e:
            print(f"ERROR: Existing settings.json is invalid JSON: {e}", file=sys.stderr)
            print(f"Fix or move {settings_path} and retry.", file=sys.stderr)
            sys.exit(1)
    else:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}

    if not isinstance(existing, dict):
        existing = {}

    existing.setdefault("hooks", {})
    if not isinstance(existing["hooks"], dict):
        existing["hooks"] = {}

    # Migration: earlier versions of this installer registered the compact
    # hook under the event name "compact", but Claude Code's canonical event
    # name is "PreCompact". Any hook registered under "compact" is silently
    # ignored by the runtime. Strip stale "compact" entries that point at
    # our hook file so users upgrading from 0.1.0 don't end up with dead
    # config alongside the correct PreCompact entry.
    _legacy_compact = existing["hooks"].get("compact")
    if isinstance(_legacy_compact, list):
        _cleaned = [
            h for h in _legacy_compact
            if not (
                isinstance(h, dict)
                and "truememory" in str(h.get("command", "")).lower()
            )
        ]
        if _cleaned:
            existing["hooks"]["compact"] = _cleaned
        else:
            del existing["hooks"]["compact"]
        if len(_cleaned) != len(_legacy_compact):
            print(
                "Migrated legacy 'compact' hook entry to 'PreCompact' "
                "(earlier versions registered the wrong event name)."
            )

    # Migration: earlier versions wrote hooks in the flat format
    # {type, command} instead of the required {matcher, hooks: [{type, command}]}.
    # Claude Code rejects the entire settings.json when any hook uses the
    # old format. Upgrade in-place before merging new entries.
    for event in list(existing["hooks"].keys()):
        entries = existing["hooks"][event]
        if not isinstance(entries, list):
            continue
        migrated = []
        did_migrate = False
        for h in entries:
            if not isinstance(h, dict):
                migrated.append(h)
                continue
            if "type" in h and "command" in h and "hooks" not in h:
                migrated.append({
                    "matcher": "",
                    "hooks": [h],
                })
                did_migrate = True
            else:
                migrated.append(h)
        if did_migrate:
            existing["hooks"][event] = migrated
            print(f"Migrated '{event}' hook entries from old format to new {{matcher, hooks}} schema.")

    for event, hooks in settings["hooks"].items():
        existing["hooks"].setdefault(event, [])
        if not isinstance(existing["hooks"][event], list):
            existing["hooks"][event] = []
        # Don't add duplicates (match on stop.py / session_start.py etc.)
        for hook in hooks:
            hook_file = str(hook_files[event])
            already_present = False
            for h in existing["hooks"][event]:
                if not isinstance(h, dict):
                    continue
                # Check new schema: {matcher, hooks: [{type, command}]}
                inner_hooks = h.get("hooks", [])
                if isinstance(inner_hooks, list):
                    for ih in inner_hooks:
                        if isinstance(ih, dict) and hook_file in ih.get("command", ""):
                            already_present = True
                            break
                # Check old schema: {type, command} (for migration)
                if hook_file in h.get("command", ""):
                    already_present = True
                if already_present:
                    break
            if not already_present:
                existing["hooks"][event].append(hook)

    settings_path.write_text(json.dumps(existing, indent=2))
    print(f"Hooks installed in {settings_path}")
    print(f"Events configured: {', '.join(settings['hooks'].keys())}")

    # Merge CLAUDE_TEMPLATE.md into the user's CLAUDE.md so Claude knows
    # how to use truememory's MCP tools during conversations.
    #
    # The template file is force-included inside the package at wheel-build
    # time (see pyproject.toml [tool.hatch.build.targets.wheel.force-include])
    # so it lives next to this module after installation. In editable/dev
    # installs we fall back to the repo-root copy so ``pip install -e .``
    # still finds it.
    template_path = Path(__file__).parent / "CLAUDE_TEMPLATE.md"
    if not template_path.exists():
        template_path = Path(__file__).parent.parent / "CLAUDE_TEMPLATE.md"
    claude_md_path = Path.home() / ".claude" / "CLAUDE.md"
    _merge_claude_md(template_path, claude_md_path)

    print("\nTo verify: truememory-ingest status")


_CLAUDE_MD_MARKER_START = "<!-- BEGIN truememory-ingest managed section -->"
_CLAUDE_MD_MARKER_END = "<!-- END truememory-ingest managed section -->"


def _merge_claude_md(template_path: Path, target_path: Path) -> None:
    """Merge the CLAUDE_TEMPLATE.md content into the user's CLAUDE.md.

    Uses marker comments to delimit the managed section so it can be
    safely updated or removed later. The user's existing content outside
    the markers is preserved untouched.
    """
    if not template_path.exists():
        print(f"  [WARN] Template not found at {template_path}, skipping CLAUDE.md merge")
        return

    try:
        template_content = template_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        print(f"  [WARN] Cannot read template: {e}")
        return

    managed_block = (
        f"\n{_CLAUDE_MD_MARKER_START}\n"
        f"{template_content}\n"
        f"{_CLAUDE_MD_MARKER_END}\n"
    )

    # Create ~/.claude/ if needed
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"  [WARN] Cannot create {target_path.parent}: {e}")
        return

    # Read existing content if present
    existing = ""
    if target_path.exists():
        try:
            existing = target_path.read_text(encoding="utf-8")
        except OSError as e:
            print(f"  [WARN] Cannot read existing CLAUDE.md: {e}")
            return

    # Back up the existing file before mutating it
    if existing and target_path.exists():
        backup_path = target_path.with_suffix(".md.bak")
        try:
            backup_path.write_text(existing, encoding="utf-8")
        except OSError:
            pass  # Non-fatal

    # If a managed block already exists, replace it; otherwise append
    if _CLAUDE_MD_MARKER_START in existing and _CLAUDE_MD_MARKER_END in existing:
        before, _, rest = existing.partition(_CLAUDE_MD_MARKER_START)
        _, _, after = rest.partition(_CLAUDE_MD_MARKER_END)
        new_content = before.rstrip() + managed_block + after.lstrip()
    else:
        new_content = existing.rstrip() + "\n" + managed_block if existing else managed_block

    try:
        target_path.write_text(new_content, encoding="utf-8")
        print(f"  [OK] Merged truememory instructions into {target_path}")
    except OSError as e:
        print(f"  [WARN] Cannot write CLAUDE.md: {e}")


def _run_status(args):
    """Print a health summary: hooks installed, LLM backend available, memory count."""
    print("TrueMemory Ingestion — Status Check")
    print("=" * 40)

    # 1. truememory availability
    try:
        import truememory
        version = getattr(truememory, "__version__", "unknown")
        print(f"  [OK] truememory {version} importable")
    except ImportError as e:
        print(f"  [FAIL] truememory not importable: {e}")
        print("         Install with: pip install truememory")
        return

    # 2. LLM backend
    try:
        from truememory.ingest.models import auto_detect
        cfg = auto_detect()
        print(f"  [OK] LLM backend: {cfg.provider} / {cfg.model}")
    except RuntimeError as e:
        print(f"  [WARN] No LLM backend: {e}")
        print("         Without an LLM, extraction falls back to regex heuristics.")

    # 3. Hooks installed?
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            hooks = settings.get("hooks", {})
            expected = ["SessionStart", "UserPromptSubmit", "Stop", "PreCompact"]
            installed = []
            missing = []
            for event in expected:
                event_hooks = hooks.get(event, [])
                has_truememory = any(
                    "truememory" in str(h.get("command", "")).lower()
                    for h in event_hooks
                    if isinstance(h, dict)
                )
                if has_truememory:
                    installed.append(event)
                else:
                    missing.append(event)
            if installed:
                print(f"  [OK] Hooks installed: {', '.join(installed)}")
            if missing:
                print(f"  [WARN] Hooks missing: {', '.join(missing)}")
                print("         Run: truememory-ingest install")
        except json.JSONDecodeError:
            print(f"  [FAIL] settings.json at {settings_path} is invalid JSON")
    else:
        print(f"  [WARN] No Claude Code settings.json at {settings_path}")
        print("         Run: truememory-ingest install")

    # 4. Database + memory count
    #
    # truememory.Memory exposes stats() which returns a dict including
    # ``message_count``. Previously this block looked for ``get_count`` /
    # ``count`` attributes that don't exist on the Memory class, so the
    # "N memories stored" annotation in the README sample output never
    # actually printed.
    try:
        from truememory import Memory
        memory = Memory()
        try:
            stats = memory.stats() if hasattr(memory, "stats") else {}
            count = None
            if isinstance(stats, dict):
                for key in ("message_count", "total_memories", "count", "messages"):
                    if key in stats and isinstance(stats[key], int):
                        count = stats[key]
                        break
            if count is not None:
                print(f"  [OK] Memory database accessible — {count} memories stored")
            else:
                print("  [OK] Memory database accessible (count unavailable)")
        except Exception:
            print("  [OK] Memory database accessible (stats unavailable)")
    except Exception as e:
        print(f"  [WARN] Cannot access memory database: {e}")

    # 5. Trace and log directories
    trace_dir = Path.home() / ".truememory" / "traces"
    log_dir = Path.home() / ".truememory" / "logs"
    if trace_dir.exists():
        traces = list(trace_dir.glob("*.json"))
        print(f"  [INFO] Traces: {len(traces)} files in {trace_dir}")
    if log_dir.exists():
        logs = list(log_dir.glob("*.log"))
        print(f"  [INFO] Logs:   {len(logs)} files in {log_dir}")
        if logs:
            latest = max(logs, key=lambda p: p.stat().st_mtime)
            print(f"         Most recent: {latest.name}")


def _run_uninstall(args):
    """Remove truememory hooks from Claude Code settings."""
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        print(f"No settings file at {settings_path} — nothing to uninstall")
        return

    try:
        settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: settings.json is invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    hooks = settings.get("hooks", {})
    removed = []

    for event in list(hooks.keys()):
        event_hooks = hooks[event]
        if not isinstance(event_hooks, list):
            continue
        kept = []
        for h in event_hooks:
            if isinstance(h, dict) and "truememory" in str(h.get("command", "")).lower():
                removed.append(f"{event}: {h.get('command', '')[:80]}")
            else:
                kept.append(h)
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]

    if not removed:
        print("No truememory hooks found in settings.json")
        return

    if args.dry_run:
        print("Would remove:")
        for r in removed:
            print(f"  - {r}")
        return

    settings["hooks"] = hooks
    settings_path.write_text(json.dumps(settings, indent=2))
    print(f"Removed {len(removed)} truememory hooks from {settings_path}")
    for r in removed:
        print(f"  - {r}")

    # Also remove the managed CLAUDE.md block if present
    claude_md_path = Path.home() / ".claude" / "CLAUDE.md"
    if claude_md_path.exists():
        try:
            content = claude_md_path.read_text(encoding="utf-8")
            if _CLAUDE_MD_MARKER_START in content and _CLAUDE_MD_MARKER_END in content:
                before, _, rest = content.partition(_CLAUDE_MD_MARKER_START)
                _, _, after = rest.partition(_CLAUDE_MD_MARKER_END)
                new_content = (before.rstrip() + "\n" + after.lstrip()).strip() + "\n"
                claude_md_path.write_text(new_content, encoding="utf-8")
                print(f"Removed truememory-managed section from {claude_md_path}")
        except OSError:
            pass


def _run_stats(args):
    data = json.loads(Path(args.trace_file).read_text())
    summary = data.get("summary", {})
    trace = data.get("trace", [])

    print(f"Facts extracted:    {summary.get('facts_extracted', 0)}")
    print(f"Passed gate:        {summary.get('facts_encoded', 0)}")
    print(f"Stored (new):       {summary.get('facts_stored', 0)}")
    print(f"Updated (existing): {summary.get('facts_updated', 0)}")
    print(f"Skipped (gate):     {summary.get('facts_skipped_gate', 0)}")
    print(f"Skipped (dedup):    {summary.get('facts_skipped_dedup', 0)}")
    print(f"Time:               {summary.get('elapsed_seconds', 0):.1f}s")

    print(f"\n--- Decision Trace ({len(trace)} facts) ---\n")
    for entry in trace:
        action = entry.get("action", "?")
        fact = entry.get("fact", "")[:70]
        gate = entry.get("gate", {})
        print(f"  [{action:>13}] {fact}")
        if gate:
            print(f"               novelty={gate.get('novelty',0):.2f} "
                  f"salience={gate.get('salience',0):.2f} "
                  f"pred_error={gate.get('prediction_error',0):.2f} "
                  f"score={gate.get('score',0):.2f}")


# ---------------------------------------------------------------------------
# logs / trace / facts commands — visibility into background ingestion
# ---------------------------------------------------------------------------

_TRACE_DIR = Path.home() / ".truememory" / "traces"
_LOG_DIR = Path.home() / ".truememory" / "logs"


def _most_recent(directory: Path, pattern: str = "*") -> Path | None:
    """Return the most recently modified file in a directory, or None."""
    if not directory.exists():
        return None
    files = [p for p in directory.glob(pattern) if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _sanitize_session(session: str) -> str:
    """Sanitize a user-supplied session string for safe filesystem lookup.

    Session IDs passed on the command line are used to look up files inside
    ``~/.truememory/traces`` and ``~/.truememory/logs``. A value like
    ``../../../etc/passwd`` would traverse out of those directories on an
    unsanitized glob. We restrict to alphanumeric + dash/underscore and cap
    at 64 chars — the same rule UserPromptSubmit already applies when it
    writes session-scoped buffer files (see hooks/user_prompt_submit.py).
    """
    return "".join(c for c in session if c.isalnum() or c in "-_")[:64]


def _find_session_file(directory: Path, session: str, suffix: str) -> Path | None:
    """Find a specific session file, or fall back to the most recent.

    The ``session`` argument is sanitized to prevent path traversal — if the
    caller passes a string containing shell metacharacters or ``..`` we
    silently drop those characters and look up the cleaned string.
    """
    if session:
        safe = _sanitize_session(session)
        if not safe:
            return None
        # Try exact match first, then prefix match
        exact = directory / f"{safe}{suffix}"
        if exact.exists():
            return exact
        matches = list(directory.glob(f"{safe}*{suffix}"))
        if matches:
            return matches[0]
        return None
    return _most_recent(directory, f"*{suffix}")


def _run_logs(args):
    """Tail recent hook log output from ~/.truememory/logs/."""
    if not _LOG_DIR.exists():
        print(f"No logs directory at {_LOG_DIR}")
        print("(Logs are created when the Stop hook fires in Claude Code.)")
        return

    if args.list:
        files = sorted(_LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            print(f"No log files in {_LOG_DIR}")
            return
        print(f"Log files in {_LOG_DIR}:")
        for f in files[:20]:
            size = f.stat().st_size
            mtime = f.stat().st_mtime
            from datetime import datetime
            when = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = size / 1024
            print(f"  {when}  {size_kb:>7.1f} KB  {f.name}")
        return

    log_path = _find_session_file(_LOG_DIR, args.session, ".log")
    if log_path is None:
        if args.session:
            print(f"No log file found for session '{args.session}' in {_LOG_DIR}")
        else:
            print(f"No log files yet in {_LOG_DIR}")
        return

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        print(f"ERROR reading {log_path}: {e}", file=sys.stderr)
        return

    print(f"--- {log_path.name} (last {args.tail} of {len(lines)} lines) ---")
    for line in lines[-args.tail:]:
        print(line)


def _run_trace(args):
    """Show the per-fact decision trace for a session."""
    trace_path = _find_session_file(_TRACE_DIR, args.session, ".json")
    if trace_path is None:
        if args.session:
            print(f"No trace file found for session '{args.session}' in {_TRACE_DIR}")
        else:
            print(f"No trace files yet in {_TRACE_DIR}")
            print("(Traces are created when the Stop hook processes a conversation.)")
        return

    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR reading {trace_path}: {e}", file=sys.stderr)
        return

    if args.raw:
        print(json.dumps(data, indent=2))
        return

    # Pretty-print (same as stats command, but auto-resolves the file)
    summary = data.get("summary", {})
    trace = data.get("trace", [])

    print(f"Trace: {trace_path.name}")
    print(f"{'=' * 60}")
    print(f"  Extracted:   {summary.get('facts_extracted', 0)}")
    print(f"  Encoded:     {summary.get('facts_encoded', 0)}")
    print(f"  Stored:      {summary.get('facts_stored', 0)}")
    print(f"  Updated:     {summary.get('facts_updated', 0)}")
    print(f"  Skip (gate): {summary.get('facts_skipped_gate', 0)}")
    print(f"  Skip (dedup):{summary.get('facts_skipped_dedup', 0)}")
    print(f"  Elapsed:     {summary.get('elapsed_seconds', 0):.1f}s")

    print(f"\n--- Per-fact decisions ({len(trace)} facts) ---\n")
    for entry in trace:
        action = entry.get("action", "?")
        fact = entry.get("fact", "")[:80]
        category = entry.get("category", "")
        gate = entry.get("gate", {})

        icon = {
            "stored": "+",
            "updated": "~",
            "skipped_gate": "×",
            "skipped_dedup": "=",
        }.get(action, "?")

        print(f"  [{icon}] {fact}")
        if category:
            print(f"      category={category}", end="")
            if gate:
                print(f"  score={gate.get('score', 0):.2f}  "
                      f"(n={gate.get('novelty', 0):.2f}, "
                      f"s={gate.get('salience', 0):.2f}, "
                      f"p={gate.get('prediction_error', 0):.2f})")
            else:
                print()


def _run_facts(args):
    """Show the facts stored during a session, optionally filtered.

    This is the "visible memory encoding" differentiator — users can see
    exactly which facts made it through the encoding gate and why.
    """
    trace_path = _find_session_file(_TRACE_DIR, args.session, ".json")
    if trace_path is None:
        if args.session:
            print(f"No trace file found for session '{args.session}'")
        else:
            print(f"No trace files yet in {_TRACE_DIR}")
        return

    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR reading {trace_path}: {e}", file=sys.stderr)
        return

    trace = data.get("trace", [])

    # Filter based on flags
    shown_actions = {"stored", "updated"}
    if args.all:
        shown_actions |= {"skipped_gate", "skipped_dedup"}

    filtered = [
        e for e in trace
        if e.get("action", "") in shown_actions
        and (not args.category or e.get("category", "") == args.category)
    ]

    if not filtered:
        if args.category:
            print(f"No facts matched category={args.category}")
        elif not args.all:
            print("No facts were stored in this session.")
            print("(Use --all to also show facts skipped by the encoding gate.)")
        else:
            print("No facts in this trace.")
        return

    # Group by category for readable output
    by_category: dict[str, list] = {}
    for entry in filtered:
        cat = entry.get("category", "general")
        by_category.setdefault(cat, []).append(entry)

    print(f"Session: {trace_path.stem}")
    print(f"{'=' * 60}")
    total_stored = sum(1 for e in filtered if e.get("action") == "stored")
    total_updated = sum(1 for e in filtered if e.get("action") == "updated")
    total_skipped = sum(1 for e in filtered if e.get("action", "").startswith("skipped"))
    print(f"Stored: {total_stored}  Updated: {total_updated}"
          + (f"  Skipped: {total_skipped}" if args.all else ""))

    for category in sorted(by_category.keys()):
        print(f"\n## {category} ({len(by_category[category])})")
        for entry in by_category[category]:
            action = entry.get("action", "?")
            icon = {
                "stored": "+",
                "updated": "~",
                "skipped_gate": "×",
                "skipped_dedup": "=",
            }.get(action, "?")
            fact = entry.get("fact", "")
            gate = entry.get("gate", {})
            print(f"  [{icon}] {fact}")
            if args.all and gate:
                print(f"        score={gate.get('score', 0):.2f}  "
                      f"reason={gate.get('reason', '')[:60]}")


def _print_result(result: IngestionResult):
    print(f"\nIngestion complete in {result.elapsed_seconds:.1f}s:")
    print(f"  Extracted:  {result.facts_extracted} facts")
    print(f"  Stored:     {result.facts_stored} new memories")
    print(f"  Updated:    {result.facts_updated} existing memories")
    print(f"  Skipped:    {result.facts_skipped_gate} (gate) + {result.facts_skipped_dedup} (dedup)")
    total_kept = result.facts_stored + result.facts_updated
    if result.facts_extracted > 0:
        rate = total_kept / result.facts_extracted * 100
        print(f"  Retention:  {rate:.0f}% ({total_kept}/{result.facts_extracted})")


if __name__ == "__main__":
    main()
