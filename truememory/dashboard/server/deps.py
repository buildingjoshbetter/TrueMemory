from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from truememory.engine import TrueMemoryEngine

log = logging.getLogger(__name__)

_DEFAULT_DB = Path.home() / ".truememory" / "memories.db"
_CONFIG_PATH = Path.home() / ".truememory" / "config.json"

_engine: TrueMemoryEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> TrueMemoryEngine:
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        db_path = os.environ.get("TRUEMEMORY_DB_PATH", str(_DEFAULT_DB))
        _engine = TrueMemoryEngine(db_path=db_path)
        _engine._ensure_connection()
        return _engine


def get_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
