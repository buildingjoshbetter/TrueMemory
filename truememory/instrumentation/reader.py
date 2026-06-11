"""
Telemetry read path — query recent instrumentation data.

Used by the ``truememory_telemetry`` MCP tool and potentially a CLI command.
Opens the instrumentation DB in read-only mode.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from truememory.instrumentation.writer import _resolve_db_path
from truememory.storage import DEFAULT_BUSY_TIMEOUT_MS


def query_telemetry(
    signal: str = "",
    limit: int = 50,
    since_hours: float = 24.0,
) -> list[dict]:
    """Query recent telemetry signals from instrumentation.db.

    Args:
        signal: Filter by signal type (e.g. "gate_decision", "timing").
                Empty string returns all signals.
        limit: Maximum rows to return.
        since_hours: Only return rows from the last N hours.

    Returns:
        List of dicts with ts, signal, and parsed data fields.
    """
    db_path = _resolve_db_path()
    if not Path(db_path).exists():
        return []

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return []

    try:
        # M-86: align busy_timeout with the single-source constant used by the
        # writer so a concurrent prune/insert does not immediately error out.
        conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
        cutoff = time.time() - (since_hours * 3600)
        if signal:
            rows = conn.execute(
                "SELECT ts, signal, data FROM telemetry "
                "WHERE ts > ? AND signal = ? "
                "ORDER BY ts DESC LIMIT ?",
                (cutoff, signal, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ts, signal, data FROM telemetry "
                "WHERE ts > ? "
                "ORDER BY ts DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()

        results = []
        for ts, sig, data_str in rows:
            try:
                data = json.loads(data_str) if data_str else {}
            except (json.JSONDecodeError, ValueError):
                data = {"raw": data_str}
            results.append({
                "ts": ts,
                "signal": sig,
                "data": data,
            })
        return results
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def format_telemetry_text(rows: list[dict]) -> str:
    """Format telemetry rows as human-readable text for MCP tool output."""
    if not rows:
        return "No telemetry data found."

    import datetime
    lines = []
    for row in rows:
        ts = row.get("ts", 0)
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        sig = row.get("signal", "?")
        data = row.get("data", {})

        # Format data compactly
        parts = []
        for k, v in data.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            elif isinstance(v, bool):
                parts.append(f"{k}={'Y' if v else 'N'}")
            else:
                parts.append(f"{k}={v}")
        data_str = " ".join(parts)
        lines.append(f"[{time_str}] {sig}: {data_str}")

    return "\n".join(lines)
