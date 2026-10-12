"""Check an installed wheel's CLI and MCP wire contract without starting models.

Run with the wheel environment's Python: python -I tests/mcp-wheel-smoke.py
This standalone probe deliberately avoids the source checkout and pytest fixtures.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
from contextlib import suppress
from pathlib import Path

_TOOLS = {
    "truememory_store", "truememory_directives", "truememory_search",
    "truememory_search_deep", "truememory_get", "truememory_forget",
    "truememory_stats", "truememory_configure", "truememory_status",
    "truememory_entity_profile", "truememory_consolidate",
}
_ALWAYS_LOADED = {
    "truememory_store", "truememory_directives", "truememory_search",
    "truememory_search_deep", "truememory_forget", "truememory_stats",
}
_TIMEOUT = 30
_PROTOCOL_VERSION = "2025-06-18"

# main() starts model services and maintenance. The registered server object
# exposes the real MCP transport and schemas without those inference services.
_SERVER = """
import importlib.util
import sys
from pathlib import Path

origin = importlib.util.find_spec("truememory").origin
if Path(origin).resolve() != Path(sys.argv[1]).resolve():
    raise RuntimeError("Installed wheel import provenance mismatch")
from truememory.mcp_server import mcp
mcp.run(transport="stdio")
"""


class SmokeFailure(Exception):
    """A safe diagnostic that contains no subprocess payload or private path."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _object(value: object, label: str) -> dict:
    _require(isinstance(value, dict), f"Expected a JSON object for {label}")
    return value


def _environment(root: Path) -> dict[str, str]:
    home = root / "home"
    cache = root / "cache"
    temp = root / "tmp"
    for directory in (home, cache, temp):
        directory.mkdir()
    # Build an allowlist instead of copying credentials or personal settings.
    env = {
        "PATH": os.pathsep.join((str(Path(sys.executable).parent), os.defpath)),
        "HOME": str(home), "USERPROFILE": str(home),
        "APPDATA": str(home), "LOCALAPPDATA": str(home),
        "XDG_CONFIG_HOME": str(home), "XDG_CACHE_HOME": str(cache),
        "HF_HOME": str(cache), "TORCH_HOME": str(cache),
        "TMPDIR": str(temp), "TMP": str(temp), "TEMP": str(temp),
        "USER": "synthetic", "LOGNAME": "synthetic", "USERNAME": "synthetic",
        "TRUEMEMORY_TELEMETRY": "off",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_MAX_THREADS": "1",
    }
    if os.name == "nt":
        for key in ("SYSTEMROOT", "WINDIR", "COMSPEC"):
            if key in os.environ:
                env[key] = os.environ[key]
    return env


def _check_cli(root: Path, env: dict[str, str], version: str) -> None:
    suffix = ".exe" if os.name == "nt" else ""
    executable = Path(sysconfig.get_path("scripts")) / f"truememory-mcp{suffix}"
    _require(executable.is_file(), "Installed console script is missing")
    for flag, expected in (("--help", "Usage: truememory-mcp"), ("--version", version)):
        # The executable belongs to the venv and flags are fixed, never shell input.
        result = subprocess.run(
            [str(executable), flag], cwd=root, env=env,
            capture_output=True, text=True, timeout=_TIMEOUT, check=False,
        )
        _require(result.returncode == 0, f"Console {flag} failed (exit {result.returncode})")
        _require(expected in result.stdout, f"Console {flag} output is incorrect")


async def _send(process: asyncio.subprocess.Process, message: dict) -> None:
    _require(process.stdin is not None, "MCP stdin is unavailable")
    process.stdin.write((json.dumps(message) + "\n").encode())
    await process.stdin.drain()


async def _response(process: asyncio.subprocess.Process, request_id: int) -> dict:
    _require(process.stdout is not None, "MCP stdout is unavailable")
    while True:
        line = await process.stdout.readline()
        _require(bool(line), "MCP server exited before its response")
        message = _object(json.loads(line), "MCP response")
        _require(message.get("jsonrpc") == "2.0", "Invalid JSON-RPC version")
        if "id" not in message:
            _require("method" in message, "Invalid MCP notification")
            continue
        _require(message["id"] == request_id, "Unexpected MCP response ID")
        _require("error" not in message, "MCP request returned an error")
        return _object(message.get("result"), "MCP result")


async def _exchange(process: asyncio.subprocess.Process) -> None:
    await _send(process, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": _PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "synthetic-wheel-smoke", "version": "1"},
        },
    })
    initialized = await _response(process, 1)
    _require(initialized.get("protocolVersion") == _PROTOCOL_VERSION,
             "MCP protocol negotiation failed")
    server = _object(initialized.get("serverInfo"), "server identity")
    _require(server.get("name") == "truememory", "Unexpected MCP server identity")
    capabilities = _object(initialized.get("capabilities"), "server capabilities")
    _require("tools" in capabilities, "MCP tools capability is missing")
    await _send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    await _send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    listed = await _response(process, 2)
    entries = listed.get("tools")
    _require(isinstance(entries, list) and len(entries) == len(_TOOLS),
             "Expected all 11 registered tools")
    tools = {}
    for entry in entries:
        tool = _object(entry, "tool")
        _require(isinstance(tool.get("name"), str), "Tool name is missing")
        tools[tool["name"]] = tool
    _require(set(tools) == _TOOLS, "Registered tool names differ from the contract")
    for name in _ALWAYS_LOADED:
        metadata = _object(tools[name].get("_meta"), "tool wire metadata")
        _require(metadata.get("anthropic/alwaysLoad") is True,
                 "alwaysLoad tool metadata was lost on the wire")
    schema = _object(tools["truememory_store"].get("inputSchema"), "store schema")
    properties = _object(schema.get("properties"), "store properties")
    directive = _object(properties.get("directive"), "directive schema")
    _require(directive.get("type") == "boolean" and directive.get("default") is False,
             "Directive input schema changed")
    _require("content" in schema.get("required", []), "Store content must be required")


async def _check_protocol(root: Path, env: dict[str, str], origin: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-I", "-c", _SERVER, str(origin), cwd=root, env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, limit=1024 * 1024,
    )
    try:
        await asyncio.wait_for(_exchange(process), timeout=_TIMEOUT)
        process.stdin.close()
        await asyncio.wait_for(process.wait(), timeout=5)
        _require(process.returncode == 0, "MCP server did not exit cleanly")
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await asyncio.wait_for(process.wait(), timeout=5)


def main() -> None:
    _require(bool(sys.flags.isolated), "Run this probe with python -I")
    distribution = importlib.metadata.distribution("truememory")
    # Report public package versions before subprocess checks, whose output is
    # intentionally redacted. Dependency import failures then remain traceable.
    print(json.dumps({
        "stage": "dependency_versions",
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "truememory_version": distribution.version,
        "mcp_version": importlib.metadata.version("mcp"),
        "pydantic_version": importlib.metadata.version("pydantic"),
        "pydantic_core_version": importlib.metadata.version("pydantic-core"),
    }), flush=True)
    origin = Path(distribution.locate_file("truememory/__init__.py")).resolve()
    spec = importlib.util.find_spec("truememory")
    _require(origin.is_file() and spec is not None and spec.origin is not None,
             "Installed wheel package is missing")
    _require(Path(spec.origin).resolve() == origin,
             "Probe imported a source checkout instead of the installed wheel")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    _require(not direct_url.get("dir_info", {}).get("editable", False),
             "An editable installation is not a wheel smoke test")
    with tempfile.TemporaryDirectory(prefix="synthetic-mcp-wheel-") as directory:
        root = Path(directory)
        env = _environment(root)
        _check_cli(root, env, distribution.version)
        asyncio.run(_check_protocol(root, env, origin))
    print(json.dumps({
        "status": "passed", "truememory_version": distribution.version,
        "mcp_version": importlib.metadata.version("mcp"),
        "tools": len(_TOOLS), "always_load_tools": len(_ALWAYS_LOADED),
    }))


if __name__ == "__main__":
    try:
        main()
    except SmokeFailure as error:
        print(f"MCP wheel smoke failed: {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        # This final boundary must redact unexpected errors as well as known ones.
        # Do not echo child output, credentials, or machine-specific paths.
        print(f"MCP wheel smoke failed: {type(error).__name__}", file=sys.stderr)
        sys.exit(1)
