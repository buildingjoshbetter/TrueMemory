from __future__ import annotations

from fastapi import APIRouter
import httpx

import truememory
from truememory.dashboard.server.deps import get_engine, get_config

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    engine = get_engine()
    stats = engine.get_stats()
    config = get_config()
    return {
        "version": truememory.__version__,
        "tier": config.get("tier", "edge"),
        "db_path": str(engine.db_path),
        "db_size_kb": stats.get("db_size_kb", 0),
        "memory_count": stats.get("message_count", 0),
        "capabilities": stats.get("capabilities", {}),
    }


@router.get("/tier")
def tier_info():
    config = get_config()
    return {
        "tier": config.get("tier", "edge"),
        "has_api_key": bool(config.get("anthropic_api_key") or config.get("api_key")),
        "api_provider": config.get("api_provider", ""),
    }


@router.post("/update/check")
async def check_update():
    current = truememory.__version__
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://pypi.org/pypi/truememory/json")
            resp.raise_for_status()
            data = resp.json()
            latest = data["info"]["version"]
            return {
                "current": current,
                "latest": latest,
                "update_available": latest != current,
            }
    except Exception:
        return {
            "current": current,
            "latest": None,
            "update_available": False,
            "error": "Could not reach PyPI",
        }
