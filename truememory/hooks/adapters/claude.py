"""Claude Code adapter — wraps the existing install logic.

Delegates to truememory.ingest.cli._run_install() for hook installation.
The existing `truememory-ingest install` command continues working unchanged.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter


class ClaudeAdapter(CLIAdapter):
    """Adapter for Claude Code CLI."""

    @property
    def name(self) -> str:
        return "Claude Code"

    @property
    def cli_id(self) -> str:
        return "claude"

    @property
    def config_path(self) -> Path:
        return Path.home() / ".claude" / "settings.json"

    def detect(self) -> bool:
        return (
            (Path.home() / ".claude").is_dir()
            or shutil.which("claude") is not None
        )

    def is_configured(self) -> bool:
        if not self.config_path.exists():
            return False
        try:
            settings = json.loads(self.config_path.read_text(encoding="utf-8"))
            hooks = settings.get("hooks", {})
            for event_hooks in hooks.values():
                if not isinstance(event_hooks, list):
                    continue
                for h in event_hooks:
                    if not isinstance(h, dict):
                        continue
                    for inner in h.get("hooks", []):
                        if isinstance(inner, dict) and "truememory" in inner.get("command", "").lower():
                            return True
                    if "truememory" in h.get("command", "").lower():
                        return True
        except (json.JSONDecodeError, OSError):
            pass
        return False

    def install_mcp(self, python_path: str | None = None) -> None:
        py = python_path or sys.executable
        try:
            import subprocess
            subprocess.run(
                ["claude", "mcp", "add", "--scope", "user", "truememory",
                 "--", py, "-m", "truememory.mcp_server"],
                check=False,
                capture_output=True,
            )
        except FileNotFoundError:
            pass

    def install_hooks(
        self,
        python_path: str | None = None,
        user_id: str = "",
        db_path: str = "",
    ) -> None:
        # Hooks now live inside the package namespace so the wheel ships them
        # cleanly.
        hooks_dir = Path(__file__).parent.parent.parent / "ingest" / "hooks"

        hook_files = {
            "SessionStart": hooks_dir / "session_start.py",
            "UserPromptSubmit": hooks_dir / "user_prompt_submit.py",
            "SessionEnd": hooks_dir / "stop.py",
            "PreCompact": hooks_dir / "compact.py",
        }
        missing = [name for name, path in hook_files.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing hook files: {', '.join(missing)}")

        import shlex
        py = python_path or sys.executable

        _HOOK_MODULES = {
            "session_start.py": "truememory.ingest.hooks.session_start",
            "user_prompt_submit.py": "truememory.ingest.hooks.user_prompt_submit",
            "stop.py": "truememory.ingest.hooks.stop",
            "compact.py": "truememory.ingest.hooks.compact",
        }

        def _build_command(hook_path: Path) -> str:
            module = _HOOK_MODULES.get(hook_path.name)
            if module:
                parts: list[str] = [py, "-m", module]
            else:
                parts = [py, str(hook_path)]
            if user_id:
                parts.extend(["--user", user_id])
            if db_path:
                parts.extend(["--db", db_path])
            if sys.platform == "win32":
                import subprocess as _sp
                return _sp.list2cmdline(parts)
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

        # Merge into existing settings (preserves other config)
        settings_path = self.config_path
        existing: dict = {}
        if settings_path.exists():
            try:
                existing = json.loads(settings_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise ValueError(f"Existing settings.json is invalid JSON: {e}")
        else:
            settings_path.parent.mkdir(parents=True, exist_ok=True)

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

        # Migration: earlier versions wired the transcript extraction hook to
        # Claude Code's "Stop" event, but "Stop" fires after every assistant
        # response (per-turn), not once at session end. The correct event is
        # "SessionEnd". Strip stale "Stop" entries that point at our hook file
        # so upgrading users don't get double-fires.
        _legacy_stop = existing["hooks"].get("Stop")
        if isinstance(_legacy_stop, list):
            _cleaned = [
                h for h in _legacy_stop
                if not (
                    isinstance(h, dict)
                    and "truememory" in str(h.get("command", "")).lower()
                )
            ]
            if _cleaned:
                existing["hooks"]["Stop"] = _cleaned
            else:
                del existing["hooks"]["Stop"]

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
                    inner_hooks = h.get("hooks", [])
                    if isinstance(inner_hooks, list):
                        for ih in inner_hooks:
                            if isinstance(ih, dict) and hook_file in ih.get("command", ""):
                                already_present = True
                                break
                    if hook_file in h.get("command", ""):
                        already_present = True
                    if already_present:
                        break
                if not already_present:
                    existing["hooks"][event].append(hook)

        settings_tmp = settings_path.with_suffix(".json.tmp")
        settings_tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        try:
            settings_tmp.replace(settings_path)
        except OSError:
            settings_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            settings_tmp.unlink(missing_ok=True)

    def uninstall(self) -> None:
        try:
            import subprocess
            subprocess.run(
                ["claude", "mcp", "remove", "truememory"],
                check=False,
                capture_output=True,
            )
        except FileNotFoundError:
            pass
        if not self.config_path.exists():
            return
        try:
            settings = json.loads(self.config_path.read_text(encoding="utf-8"))
            hooks = settings.get("hooks", {})
            for event in list(hooks.keys()):
                entries = hooks[event]
                if not isinstance(entries, list):
                    continue
                cleaned = []
                for h in entries:
                    if not isinstance(h, dict):
                        cleaned.append(h)
                        continue
                    inner_hooks = h.get("hooks", [])
                    if isinstance(inner_hooks, list):
                        has_tm = any(
                            isinstance(ih, dict) and "truememory" in ih.get("command", "").lower()
                            for ih in inner_hooks
                        )
                        if has_tm:
                            continue
                    if "truememory" in h.get("command", "").lower():
                        continue
                    cleaned.append(h)
                if cleaned:
                    hooks[event] = cleaned
                else:
                    del hooks[event]
            settings["hooks"] = hooks

            # Also remove the MCP server entry (JSON-level fallback for when
            # `claude mcp remove` fails or the CLI is not on PATH)
            mcp_servers = settings.get("mcpServers", {})
            if "truememory" in mcp_servers:
                del mcp_servers["truememory"]
                settings["mcpServers"] = mcp_servers

            self.config_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except (json.JSONDecodeError, OSError):
            pass

    def verify(self) -> bool:
        if not self.config_path.exists():
            return False
        return self.is_configured()

    def get_system_prompt_path(self) -> Path | None:
        return Path.home() / ".claude" / "CLAUDE.md"

    def get_system_prompt_content(self) -> str:
        template_path = Path(__file__).parent.parent.parent / "ingest" / "CLAUDE_TEMPLATE.md"
        if not template_path.exists():
            template_path = Path(__file__).parent.parent.parent.parent / "CLAUDE_TEMPLATE.md"
        if template_path.exists():
            try:
                return template_path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        return ""
