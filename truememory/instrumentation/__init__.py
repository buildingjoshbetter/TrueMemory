"""
TrueMemory Instrumentation — Opt-in SQLite Telemetry Overlay
=============================================================

Monkey-patches engine/client/gate at runtime to emit structured telemetry
rows into a **separate** ``~/.truememory/instrumentation.db`` file.

**Never touches memories.db.** Controlled by ``TRUEMEMORY_INSTRUMENTATION=1``
(default: off). All wrappers swallow their own exceptions so a telemetry
bug can never break production code paths.

Usage::

    # In mcp_server.py main(), after engine is ready:
    from truememory.instrumentation import install
    install()

    # Query telemetry (MCP tool or CLI):
    from truememory.instrumentation.reader import query_telemetry
    rows = query_telemetry(signal="gate_decision", limit=50)
"""

from truememory.instrumentation.log import is_enabled  # noqa: F401
from truememory.instrumentation.patch import install, uninstall  # noqa: F401

__all__ = ["is_enabled", "install", "uninstall"]
