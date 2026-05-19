"""
Business-driven feature engineering for lead scoring.

All features are designed so that a sales rep can understand *why* a lead
scored high — not just that the model said so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lead_scoring.utils.logging import get_logger

logger = get_logger(__name__)


class LeadFeatureEngineering:
    """
    Stateless feature-engineering transformer.

    All methods are static so they can be called both during training and
    at inference time without holding state.
    """

    # ── Individual feature constructors ────────────────────────────────────

    @staticmethod
    def create_engagement_score(df: pd.DataFrame) -> pd.Series:
        """
        Composite engagement metric weighted by interaction quality.

        Clicks carry more intent signal than opens, which carry more than visits.
        Normalised to [0, 100].
        """
        raw = (
            df["website_visits"] * 0.3
            + df["email_opens"] * 0.2
            + df["email_clicks"] * 0.5
        )
        max_val = raw.max()
        if max_val == 0:
            return pd.Series(np.zeros(len(df)), index=df.index)
        return (raw / max_val * 100).clip(0, 100)

    @staticmethod
    def create_recency_score(df: pd.DataFrame) -> pd.Series:
        """
        Exponential time-decay score based on days since last interaction.

        Score halves every ~14 days (decay_rate=0.05).
        """
        decay_rate = 0.05
        return (100 * np.exp(-decay_rate * df["days_since_interaction"])).clip(0, 100)

    @staticmethod
    def create_interaction_intensity(df: pd.DataFrame) -> pd.Series:
        """
        Total interactions per active week.

        Captures whether a lead engages *frequently* or just once long ago.
        """
        total = (
            df["website_visits"]
            + df["email_opens"]
            + df["email_clicks"]
            + df["followup_count"]
        )
        weeks_active = (df["days_since_interaction"] / 7).clip(lower=1)
        raw = total / weeks_active
        max_val = raw.max()
        if max_val == 0:
            return pd.Series(np.zeros(len(df)), index=df.index)
        return (raw / max_val * 100).clip(0, 100)

    @staticmethod
    def create_sales_readiness_score(
        df: pd.DataFrame,
        engagement_score: pd.Series,
        recency_score: pd.Series,
    ) -> pd.Series:
        """
        Composite "ready to buy" signal combining demo intent with recency and engagement.
        """
        return (
            df["demo_requested"] * 50
            + engagement_score * 0.3
            + recency_score * 0.2
        ).clip(0, 100)

    # ── Master transformer ─────────────────────────────────────────────────

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps and return an enriched DataFrame.

        Safe to call on both training data and single-row inference DataFrames.
        """
        df = df.copy()

        engagement = LeadFeatureEngineering.create_engagement_score(df)
        recency = LeadFeatureEngineering.create_recency_score(df)
        intensity = LeadFeatureEngineering.create_interaction_intensity(df)
        readiness = LeadFeatureEngineering.create_sales_readiness_score(df, engagement, recency)

        df["engagement_score"] = engagement
        df["recency_score"] = recency
        df["interaction_intensity"] = intensity
        df["sales_readiness_score"] = readiness

        # Interaction term: demo × engagement = strongest buy-signal combo
        df["demo_x_engagement"] = df["demo_requested"] * df["engagement_score"]

        # Cap outlier values (visits > 50 are likely bots or test data)
        df["website_visits_capped"] = df["website_visits"].clip(upper=50)
        df["email_opens_capped"] = df["email_opens"].clip(upper=20)

        n_features = len([c for c in df.columns if any(k in c for k in ("score", "intensity", "capped", "x_"))])
        logger.debug("Engineered %d new features", n_features)
        return df
