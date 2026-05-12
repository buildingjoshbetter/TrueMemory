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
        import argparse
        args = argparse.Namespace(
            user=user_id,
            db=db_path,
            dry_run=False,
        )
        from truememory.ingest.cli import _run_install
        _run_install(args)

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
