"""
Shared pytest fixtures.

All fixtures use small, fast datasets (200 rows) to keep the test suite
well under 60 seconds even without caching.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lead_scoring.data.generator import generate_crm_data
from lead_scoring.data.preprocessor import LeadDataPreprocessor
from lead_scoring.features.engineering import LeadFeatureEngineering
from lead_scoring.models.trainer import LeadScoringTrainer, ModelBundle

# ── Raw data ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """200-row synthetic CRM dataset (session-scoped for speed)."""
    return generate_crm_data(n_samples=200, conversion_rate=0.15, random_seed=0)


# ── Engineered data ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def engineered_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return LeadFeatureEngineering.engineer_features(raw_df)


# ── Fitted preprocessor ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def fitted_preprocessor(engineered_df: pd.DataFrame) -> LeadDataPreprocessor:
    preprocessor = LeadDataPreprocessor()
    preprocessor.fit_transform(engineered_df)
    return preprocessor


# ── Trained model bundle ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model_bundle(raw_df: pd.DataFrame) -> ModelBundle:
    """Full end-to-end trained bundle (session-scoped — trained once per run)."""
    trainer = LeadScoringTrainer()
    bundle, _ = trainer.train(raw_df)
    return bundle


# ── Single-lead DataFrame ──────────────────────────────────────────────────────

@pytest.fixture()
def single_lead_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "lead_id": "L00001",
                "lead_source": "Referral",
                "industry": "SaaS",
                "company_size": "Enterprise",
                "region": "North America",
                "website_visits": 10,
                "email_opens": 5,
                "email_clicks": 3,
                "demo_requested": 1,
                "days_since_interaction": 3,
                "followup_count": 2,
            }
        ]
    )
