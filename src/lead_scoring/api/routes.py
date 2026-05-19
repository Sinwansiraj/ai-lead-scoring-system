"""
API route definitions.

All routes are mounted under /api/v1 by the application factory in main.py.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status

from lead_scoring import __version__
from lead_scoring.api.schemas import (
    BatchScoreRequest,
    HealthResponse,
    LeadFeatures,
    ScoreResponse,
    ScoredLead,
)
from lead_scoring.features.engineering import LeadFeatureEngineering
from lead_scoring.models.scorer import LeadScorer
from lead_scoring.models.trainer import ModelBundle
from lead_scoring.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ── Dependency ─────────────────────────────────────────────────────────────────

def get_bundle(request: Request) -> ModelBundle:
    """FastAPI dependency that retrieves the model bundle from app state."""
    bundle: ModelBundle | None = request.app.state.bundle
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run 'python scripts/train.py' then restart the API.",
        )
    return bundle


# ── Helpers ────────────────────────────────────────────────────────────────────

def _leads_to_df(leads: list[LeadFeatures]) -> pd.DataFrame:
    """Convert a list of Pydantic request objects to a pandas DataFrame."""
    return pd.DataFrame([lead.model_dump() for lead in leads])


def _df_to_scored_leads(scored_df: pd.DataFrame) -> list[ScoredLead]:
    """Convert scored DataFrame rows to a list of ScoredLead response objects."""
    results = []
    for _, row in scored_df.iterrows():
        results.append(
            ScoredLead(
                lead_id=str(row["lead_id"]),
                lead_quality_score=int(row["lead_quality_score"]),
                conversion_probability=round(float(row["conversion_probability"]), 4),
                lead_category=row["lead_category"],
                recommended_action=str(row["recommended_action"]),
                engagement_score=round(float(row.get("engagement_score", 0.0)), 2),
                recency_score=round(float(row.get("recency_score", 0.0)), 2),
            )
        )
    return results


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    """Liveness / readiness probe. Returns 200 whether or not a model is loaded."""
    bundle: ModelBundle | None = request.app.state.bundle
    return HealthResponse(
        status="ok",
        model_loaded=bundle is not None,
        model_roc_auc=bundle.metadata.get("production_roc_auc") if bundle else None,
        version=__version__,
    )


@router.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_lead(lead: LeadFeatures, bundle: ModelBundle = Depends(get_bundle)) -> ScoreResponse:
    """
    Score a single lead and return a quality score, category, and recommended action.

    **Real-time endpoint** — typical latency < 50 ms.
    """
    df = _leads_to_df([lead])
    df_fe = LeadFeatureEngineering.engineer_features(df)
    probabilities = bundle.predict_proba(df)
    scored = LeadScorer.score_dataframe(df_fe, probabilities)

    logger.info(
        "Scored lead %s → score=%d category=%s",
        lead.lead_id,
        int(scored.iloc[0]["lead_quality_score"]),
        scored.iloc[0]["lead_category"],
    )
    return ScoreResponse(
        scored_leads=_df_to_scored_leads(scored),
        model_version=__version__,
        model_roc_auc=bundle.metadata.get("production_roc_auc", 0.0),
    )


@router.post("/score/batch", response_model=ScoreResponse, tags=["Scoring"])
async def score_batch(request_body: BatchScoreRequest, bundle: ModelBundle = Depends(get_bundle)) -> ScoreResponse:
    """
    Score up to 1000 leads in a single request.

    **Batch endpoint** — ideal for nightly CRM re-scoring jobs.
    """
    df = _leads_to_df(request_body.leads)
    df_fe = LeadFeatureEngineering.engineer_features(df)
    probabilities = bundle.predict_proba(df)
    scored = LeadScorer.score_dataframe(df_fe, probabilities)

    hot = (scored["lead_category"] == "Hot").sum()
    warm = (scored["lead_category"] == "Warm").sum()
    cold = (scored["lead_category"] == "Cold").sum()
    logger.info(
        "Batch scored %d leads: Hot=%d Warm=%d Cold=%d",
        len(request_body.leads), hot, warm, cold,
    )
    return ScoreResponse(
        scored_leads=_df_to_scored_leads(scored),
        model_version=__version__,
        model_roc_auc=bundle.metadata.get("production_roc_auc", 0.0),
    )
