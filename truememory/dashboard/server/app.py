from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import truememory
from truememory.dashboard.server.routes import (
    analytics,
    entities,
    facts,
    memories,
    sessions,
    system,
)

_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(
        title="TrueMemory Dashboard",
        version=truememory.__version__,
        docs_url=None,
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(system.router)
    app.include_router(memories.router)
    app.include_router(sessions.router)
    app.include_router(entities.router)
    app.include_router(facts.router)
    app.include_router(analytics.router)

    @app.on_event("startup")
    async def _startup_index_sessions():
        import threading
        def _bg_index():
            try:
                from truememory.dashboard.server.session_index import (
                    get_dashboard_conn, index_sessions, get_session_count
                )
                conn = get_dashboard_conn()
                if get_session_count(conn) == 0:
                    index_sessions(conn, max_sessions=2000)
            except Exception:
                import logging
                logging.getLogger(__name__).exception("Background session indexing failed")
        threading.Thread(target=_bg_index, daemon=True).start()

    if _FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")

    return app
