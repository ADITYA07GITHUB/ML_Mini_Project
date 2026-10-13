"""
NetGuard AI - FastAPI Application Server
========================================
Main application entry point configuring routers, CORS middleware,
static files serving, and health checks.
"""

import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from backend.app.config import settings
from backend.app.api import health, analysis, metrics, explain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("network_analysis.main")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Network Traffic Analysis, Behavioral Anomaly Detection & Model Explainability Platform",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enable CORS for local client development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard docs redirects so /docs and /redoc both work alongside /api/docs and /api/redoc
@app.get("/docs", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(url="/api/docs")


@app.get("/redoc", include_in_schema=False)
def redoc_redirect():
    return RedirectResponse(url="/api/redoc")


# Include API Routers under /api
app.include_router(health.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(explain.router, prefix="/api")

# Also include routers at root for convenience (e.g. /health, /metrics, /models, /analyze, /predict)
app.include_router(health.router, include_in_schema=False)
app.include_router(analysis.router, include_in_schema=False)
app.include_router(metrics.router, include_in_schema=False)
app.include_router(explain.router, include_in_schema=False)

# Mount frontend static files
if os.path.exists(settings.FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=settings.FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_index():
        """Serve main dashboard SPA."""
        index_file = os.path.join(settings.FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse({"status": "Network Analysis Dashboard Backend Running", "docs": "/api/docs"})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
