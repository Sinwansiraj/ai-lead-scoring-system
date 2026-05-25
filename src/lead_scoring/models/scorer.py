"""
Lead quality scoring and sales action recommendation logic.

Converts raw model probabilities into human-readable scores (0-100),
tiered categories (Hot / Warm / Cold), and prescriptive sales actions.
"""

from __future__ import annotations

import pandas as pd

from lead_scoring.config import settings


class LeadScorer:
    """
    Stateless post-processing layer that turns model probabilities into
    actionable CRM output.
    """

    # ── Core conversions ───────────────────────────────────────────────────

    @staticmethod
    def probability_to_score(probability: float) -> int:
        """Map a [0, 1] probability to an integer [0, 100] quality score (raw, no blending)."""
        return max(0, min(100, round(probability * 100)))

    @staticmethod
    def composite_score(
        probability: float,
        engagement_score: float = 50.0,
        recency_score: float = 50.0,
        *,
        prob_weight: float | None = None,
        engagement_weight: float | None = None,
        recency_weight: float | None = None,
    ) -> int:
        """
        Blended lead quality score combining model probability with behavioural signals.

        Prevents the pathological case where a lead with maximum engagement and
        recent interaction still scores near-zero because the XGBoost probability
        is low (e.g. cold-call startup leads that are nonetheless very active).

        Weights default to ``settings.score_*_weight`` values so they can be
        tuned via env vars at deploy time without a code change.

        Default weights: 60 % model probability · 25 % engagement · 15 % recency
        """
        pw = prob_weight if prob_weight is not None else settings.score_prob_weight
        ew = engagement_weight if engagement_weight is not None else settings.score_engagement_weight
        rw = recency_weight if recency_weight is not None else settings.score_recency_weight

        composite = pw * (probability * 100) + ew * engagement_score + rw * recency_score
        return max(0, min(100, round(composite)))

    @staticmethod
    def score_to_category(score: int) -> str:
        """
        Map a numeric score to a sales tier.

        Thresholds are driven by ``settings.hot_threshold`` and
        ``settings.warm_threshold`` so they can be tuned via env vars.
        """
        if score >= settings.hot_threshold:
            return "Hot"
        if score >= settings.warm_threshold:
            return "Warm"
        return "Cold"

    @staticmethod
    def recommend_action(category: str, engagement_score: float, recency_score: float) -> str:
        """
        Return a sales playbook action based on the lead's profile.

        Designed so a non-technical sales rep can act on the output immediately.
        """
        if category == "Hot":
            return "Immediate Call – Strike while hot!" if recency_score > 70 else "Re-engagement Email → Call"
        if category == "Warm":
            return "Personalized Demo Invite" if engagement_score > 60 else "Value-driven Nurture Campaign"
        # Cold
        return "Re-qualification Survey" if recency_score < 30 else "Automated Drip Campaign"

    # ── Batch scoring ──────────────────────────────────────────────────────

    @classmethod
    def score_dataframe(cls, df: pd.DataFrame, probabilities: pd.Series | list[float]) -> pd.DataFrame:
        """
        Enrich a DataFrame with score, category, and recommended action columns.

        Parameters
        ----------
        df:
            Original lead DataFrame (must contain ``engagement_score`` and
            ``recency_score`` columns if they should influence the action).
        probabilities:
            Model-predicted conversion probabilities, aligned to ``df``'s index.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with additional columns, sorted by quality score desc.
        """
        df = df.copy()
        df["conversion_probability"] = list(probabilities)
        df["lead_quality_score"] = df.apply(
            lambda row: cls.composite_score(
                row["conversion_probability"],
                row.get("engagement_score", 50.0),
                row.get("recency_score", 50.0),
            ),
            axis=1,
        )
        df["lead_category"] = df["lead_quality_score"].apply(cls.score_to_category)
        df["recommended_action"] = df.apply(
            lambda row: cls.recommend_action(
                row["lead_category"],
                row.get("engagement_score", 50.0),
                row.get("recency_score", 50.0),
            ),
            axis=1,
        )
        return df.sort_values("lead_quality_score", ascending=False)
