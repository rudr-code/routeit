from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure all workspace packages are on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in ["engine", "reference", "roads", "traffic", "backend"]:
    _p_str = str(_ROOT / _p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from backend.app.db import get_db
from backend.app.queue import JobQueueWorker
from backend.app.routers.cost_profiles import router as cost_profiles_router
from backend.app.routers.customers import router as customers_router
from backend.app.routers.depots import router as depots_router
from backend.app.routers.orders import router as orders_router
from backend.app.routers.planning import router as planning_router
from backend.app.routers.plans import router as plans_router
from backend.app.routers.vehicles import router as vehicles_router

try:
    from api.database import get_db as get_delivery_db
    from api.routes.benchmark import router as benchmark_router
    from api.routes.delivery import router as delivery_router
    from api.routes.events import router as events_router
    from api.routes.health import router as health_router
    from api.routes.problems import router as problems_router
    from api.routes.ws import router as ws_router
    _HAS_UI_ROUTERS = True
except ImportError:
    _HAS_UI_ROUTERS = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("routeit.api")

worker: JobQueueWorker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker
    if os.environ.get("ROUTEIT_TESTING") != "1":
        db = get_db()
        worker = JobQueueWorker(db)
        worker.start()
        if _HAS_UI_ROUTERS:
            try:
                get_delivery_db()
            except Exception as e:
                logger.warning(f"Could not initialize delivery db: {e}")
        logger.info("[Route.it API] Database and JobQueueWorker initialized.")
    yield
    if worker:
        worker.stop()
    logger.info("[Route.it API] Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Route.it / Q-HGALS 5.0 Fleet Routing Platform",
        version="5.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount v1 normative routers
    app.include_router(depots_router)
    app.include_router(vehicles_router)
    app.include_router(customers_router)
    app.include_router(orders_router)
    app.include_router(cost_profiles_router)
    app.include_router(planning_router)
    app.include_router(plans_router)

    # Mount UI portal routers (/api prefix)
    if _HAS_UI_ROUTERS:
        ui_routers = [
            health_router,
            problems_router,
            events_router,
            ws_router,
            delivery_router,
            benchmark_router,
        ]
        for r in ui_routers:
            app.include_router(r)
            app.include_router(r, prefix="/api")

    @app.get("/v1/healthz", tags=["System"])
    def health_check():
        return {"status": "HEALTHY", "version": "5.0.0", "engine": "Q-HGALS 5.0"}

    # Mount static assets if dashboard/dist exists
    dist_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "dashboard", "dist")
    )
    if os.path.exists(dist_dir):
        assets_dir = os.path.join(dist_dir, "assets")
        if os.path.exists(assets_dir):
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/")
        async def serve_root():
            return FileResponse(os.path.join(dist_dir, "index.html"))

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            f_path = os.path.join(dist_dir, full_path)
            if full_path and os.path.exists(f_path) and os.path.isfile(f_path):
                return FileResponse(f_path)
            return FileResponse(os.path.join(dist_dir, "index.html"))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port, reload=True)
