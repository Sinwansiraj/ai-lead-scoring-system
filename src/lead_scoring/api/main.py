"""
FastAPI application factory.

Usage (development):
    uvicorn lead_scoring.api.main:app --reload --port 8000

Usage (production):
    uvicorn lead_scoring.api.main:app --workers 4 --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from lead_scoring import __version__
from lead_scoring.api.routes import router
from lead_scoring.config import settings
from lead_scoring.models.trainer import LeadScoringTrainer, ModelBundle
from lead_scoring.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load the model bundle on startup; release on shutdown.
    If no model is found, the API still starts — health check reports 503.
    """
    app.state.bundle = None
    try:
        bundle: ModelBundle = LeadScoringTrainer.load()
        app.state.bundle = bundle
        logger.info(
            "Model loaded successfully (ROC-AUC=%.4f)",
            bundle.metadata.get("production_roc_auc", 0.0),
        )
    except FileNotFoundError as exc:
        logger.warning("Model not found on startup: %s", exc)

    yield  # ── application runs here ──

    logger.info("API shutting down")


def create_app() -> FastAPI:
    application = FastAPI(
        title="Lead Quality Scoring API",
        description=(
            "AI-driven B2B CRM lead prioritisation.\n\n"
            "Scores leads 0–100, classifies them as Hot / Warm / Cold, "
            "and recommends a sales action."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — tighten origins in production via env vars
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Mount all routes under /api/v1
    application.include_router(router, prefix="/api/v1")

    # Root redirect to docs
    @application.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({"message": "Lead Scoring API", "docs": "/docs", "version": __version__})

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "lead_scoring.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
    )
