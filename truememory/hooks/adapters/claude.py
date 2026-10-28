"""Claude Code adapter — wraps the existing install logic.

Delegates to truememory.ingest.cli._run_install() for hook installation.
The existing `truememory-ingest install` command continues working unchanged.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from truememory.hooks.adapters.base import CLIAdapter


_HOOK_MODULES = {
    "SessionStart": "truememory.ingest.hooks.session_start",
    "UserPromptSubmit": "truememory.ingest.hooks.user_prompt_submit",
    "SessionEnd": "truememory.ingest.hooks.stop",
    "Stop": "truememory.ingest.hooks.stop",
    "PreCompact": "truememory.ingest.hooks.compact",
}


def claude_hook_settings(python_path: str, user_id: str = "", db_path: str = "", *,
                         preserve_stop: bool = False) -> dict:
    import shlex
    import subprocess
    events = ["SessionStart", "UserPromptSubmit", "SessionEnd", "PreCompact"]
    if preserve_stop or os.environ.get("TRUEMEMORY_CLAUDE_STOP_FALLBACK") == "1":
        events.append("Stop")
    settings = {"hooks": {}}
    for event in events:
        parts = [python_path, "-m", _HOOK_MODULES[event], "--host", "claude", "--event", event]
        if user_id:
            parts.extend(["--user", user_id])
        if db_path:
            parts.extend(["--db", db_path])
        command = subprocess.list2cmdline(parts) if sys.platform == "win32" else " ".join(shlex.quote(part) for part in parts)
        settings["hooks"][event] = [{"matcher": "", "hooks": [{"type": "command", "command": command}]}]
    return settings


def _owned_hook_module(command: object, hooks_dir: Path, python_path: str) -> str | None:
    import re
    import shlex
    if type(command) is not str:
        return None
    try:
        parts = shlex.split(command, posix=sys.platform != "win32")
        if sys.platform == "win32":
            parts = [part[1:-1] if len(part) >= 2 and part[0] == part[-1] == '"' else part for part in parts]
    except ValueError:
        return None
    if len(parts) < 2:
        return None
    executable = Path(parts[0]).name.lower()
    if parts[0] != python_path and re.fullmatch(r"(?:python(?:[0-9]+(?:\.[0-9]+)*)?|py)(?:\.exe)?", executable) is None:
        return None
    modules = set(_HOOK_MODULES.values())
    if len(parts) >= 3 and parts[1] == "-m" and parts[2] in modules:
        module, tail = parts[2], parts[3:]
    else:
        scripts = {str((hooks_dir / (module.rsplit(".", 1)[1] + ".py")).resolve()): module for module in modules}
        try:
            module = scripts.get(str(Path(parts[1]).resolve()))
        except (OSError, ValueError):
            return None
        tail = parts[2:]
        if module is None:
            return None
    if len(tail) % 2 or any(tail[index] not in ("--user", "--db", "--host", "--event") for index in range(0, len(tail), 2)):
        return None
    return module


def _group_has_module(group: object, module: str, hooks_dir: Path, python_path: str) -> bool:
    if type(group) is not dict:
        return False
    commands = group.get("hooks", [group])
    return type(commands) is list and any(type(hook) is dict and hook.get("type") == "command"
        and _owned_hook_module(hook.get("command"), hooks_dir, python_path) == module for hook in commands)


class ClaudeAdapter(CLIAdapter):
    """Adapter for Claude Code CLI."""

    @property
    def has_hooks(self) -> bool:
        return True

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

    @property
    def mcp_config_path(self) -> Path:
        # `claude mcp add --scope user` writes the server entry to ~/.claude.json,
        # NOT ~/.claude/settings.json. The alwaysLoad patch must target the same
        # file the CLI actually reads, or it's a silent no-op.
        return Path.home() / ".claude.json"

    def install_mcp(self, python_path: str | None = None) -> None:
        py = python_path or sys.executable
        added = False
        try:
            import subprocess
            result = subprocess.run(
                ["claude", "mcp", "add", "--scope", "user", "truememory",
                 "--", py, "-m", "truememory.mcp_server"],
                check=False,
                capture_output=True,
            )
            if result.returncode == 0:
                added = True
            else:
                stderr = (result.stderr or b"").decode("utf-8", "replace").strip()
                print(
                    f"\033[33m⚠ `claude mcp add` exited {result.returncode}"
                    f"{f': {stderr}' if stderr else ''}.\033[0m",
                    file=sys.stderr,
                )
        except FileNotFoundError:
            print(
                "\033[33m⚠ `claude` CLI not found on PATH; skipping MCP "
                "registration.\033[0m",
                file=sys.stderr,
            )

        # `claude mcp add` doesn't support alwaysLoad, so patch it into the file
        # the CLI wrote (~/.claude.json). Only bother if the add succeeded;
        # otherwise there is no entry to patch.
        if not added:
            return
        mcp_path = self.mcp_config_path
        try:
            if mcp_path.exists():
                cfg = json.loads(mcp_path.read_text(encoding="utf-8"))
                if isinstance(cfg, dict):
                    mcp = cfg.get("mcpServers", {})
                    if (
                        isinstance(mcp, dict)
                        and isinstance(mcp.get("truememory"), dict)
                        and not mcp["truememory"].get("alwaysLoad")
                    ):
                        mcp["truememory"]["alwaysLoad"] = True
                        self._atomic_write_json(mcp_path, cfg)
        except (json.JSONDecodeError, OSError):
            pass

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        # tmp-in-same-dir + replace: a crash mid-write must never truncate the
        # user's config. A bare write_text() can leave a half-written file.
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            tmp.replace(path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise

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

        py = python_path or sys.executable
        settings = claude_hook_settings(py, user_id, db_path)

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

        # Only exact owned commands migrate from the unsupported event. Foreign
        # hooks remain untouched, including commands mentioning TrueMemory.
        legacy = existing["hooks"].get("compact")
        if isinstance(legacy, list):
            cleaned = []
            for group in legacy:
                if not isinstance(group, dict):
                    cleaned.append(group)
                    continue
                if "hooks" in group and isinstance(group["hooks"], list):
                    inner = [hook for hook in group["hooks"] if not (
                        isinstance(hook, dict) and hook.get("type") == "command"
                        and _owned_hook_module(hook.get("command"), hooks_dir, py) == _HOOK_MODULES["PreCompact"])]
                    if inner:
                        cleaned.append({**group, "hooks": inner})
                elif not _group_has_module(group, _HOOK_MODULES["PreCompact"], hooks_dir, py):
                    cleaned.append(group)
            if cleaned:
                existing["hooks"]["compact"] = cleaned
            else:
                existing["hooks"].pop("compact", None)

        # A supported Stop fallback may already be the only firing trigger on
        # a host. Preserve it while always installing SessionEnd as default.
        prior_stop = existing["hooks"].get("Stop", [])
        viable_stop = isinstance(prior_stop, list) and any(
            _group_has_module(group, _HOOK_MODULES["Stop"], hooks_dir, py) for group in prior_stop)
        settings = claude_hook_settings(py, user_id, db_path, preserve_stop=viable_stop)

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
                print(f"  Migrated '{event}' hook entries from old format to new {{matcher, hooks}} schema.")

        for event, desired_groups in settings["hooks"].items():
            entries = existing["hooks"].get(event, [])
            if not isinstance(entries, list):
                entries = []
            module = _HOOK_MODULES[event]
            desired = desired_groups[0]["hooks"][0]
            merged = []
            replaced = False
            for group in entries:
                if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                    merged.append(group)
                    continue
                inner = []
                for hook in group["hooks"]:
                    owned = (isinstance(hook, dict) and hook.get("type") == "command"
                             and _owned_hook_module(hook.get("command"), hooks_dir, py) == module)
                    if owned:
                        if not replaced:
                            inner.append({**hook, "command": desired["command"]})
                            replaced = True
                    else:
                        inner.append(hook)
                if inner:
                    merged.append({**group, "hooks": inner})
            if not replaced:
                merged.extend(desired_groups)
            existing["hooks"][event] = merged

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

            self._atomic_write_json(self.config_path, settings)
        except (json.JSONDecodeError, OSError):
            pass

    def _mcp_registered(self) -> bool:
        # The MCP server entry lives in ~/.claude.json (written by
        # `claude mcp add`), not in settings.json. Check the real target.
        mcp_path = self.mcp_config_path
        if not mcp_path.exists():
            return False
        try:
            cfg = json.loads(mcp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if not isinstance(cfg, dict):
            return False
        return "truememory" in cfg.get("mcpServers", {})

    def verify(self) -> bool:
        # A complete install is hooks (settings.json) AND the MCP server entry
        # (~/.claude.json). The old verify() only checked hooks, so a half
        # install where `claude mcp add` failed still reported success.
        if not self.config_path.exists():
            return False
        return self.is_configured() and self._mcp_registered()

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
