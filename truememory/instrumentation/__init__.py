"""truememory.instrumentation — opt-in, self-contained telemetry overlay.

This package emits structured telemetry to a ``telemetry`` table inside the
user's ``memories.db`` so dashboards can plot per-memory salience, retrieval
quality, gate decisions, and the model lifecycle. TrueMemory's engine does
NOT emit any of this natively today.

Design — sits on TOP of the engine (does not modify it):

- ``install()`` monkey-patches a handful of engine + MCP-server methods at
  runtime. There are no ``emit_*`` calls woven into the core. If the core
  changes (a method is renamed or moved), the corresponding patch simply
  no-ops — it can never break the host. Each patch is applied independently
  inside its own ``try/except`` so one failure cannot disturb the others.
- **Opt-in.** ``install()`` is a no-op unless the environment variable
  ``TRUEMEMORY_INSTRUMENTATION`` is set to ``"1"``. Default off → zero
  monkey-patching, zero writes, zero overhead, no behavior change.

The ONLY core-file footprint is a single guarded call in
``truememory.mcp_server.main()`` near where models preload::

    try:
        from truememory.instrumentation import install as _install_instrumentation
        _install_instrumentation()
    except Exception:
        pass

Everything else lives here and patches at runtime.

Public API:

- ``install()`` — apply all telemetry patches (idempotent, opt-in gated).
- ``is_enabled()`` — True when ``TRUEMEMORY_INSTRUMENTATION=1``.
"""
from __future__ import annotations

from truememory.instrumentation.patch import install
from truememory.instrumentation.log import is_enabled

__all__ = ["install", "is_enabled"]
