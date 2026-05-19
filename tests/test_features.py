"""Tests for feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from lead_scoring.features.engineering import LeadFeatureEngineering


class TestLeadFeatureEngineering:
    def test_engineer_features_returns_dataframe(self, raw_df):
        result = LeadFeatureEngineering.engineer_features(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_original_df_not_mutated(self, raw_df):
        original_cols = list(raw_df.columns)
        _ = LeadFeatureEngineering.engineer_features(raw_df)
        assert list(raw_df.columns) == original_cols

    def test_all_engineered_columns_present(self, engineered_df):
        expected = {
            "engagement_score",
            "recency_score",
            "interaction_intensity",
            "sales_readiness_score",
            "demo_x_engagement",
            "website_visits_capped",
            "email_opens_capped",
        }
        assert expected.issubset(set(engineered_df.columns))

    def test_engagement_score_range(self, engineered_df):
        assert engineered_df["engagement_score"].between(0, 100).all()

    def test_recency_score_range(self, engineered_df):
        assert engineered_df["recency_score"].between(0, 100).all()

    def test_interaction_intensity_range(self, engineered_df):
        assert engineered_df["interaction_intensity"].between(0, 100).all()

    def test_sales_readiness_range(self, engineered_df):
        assert engineered_df["sales_readiness_score"].between(0, 100).all()

    def test_website_visits_capped_at_50(self, engineered_df):
        assert (engineered_df["website_visits_capped"] <= 50).all()

    def test_email_opens_capped_at_20(self, engineered_df):
        assert (engineered_df["email_opens_capped"] <= 20).all()

    def test_demo_x_engagement_zero_when_no_demo(self, raw_df):
        df_no_demo = raw_df.copy()
        df_no_demo["demo_requested"] = 0
        result = LeadFeatureEngineering.engineer_features(df_no_demo)
        assert (result["demo_x_engagement"] == 0).all()

    def test_recency_score_decreases_with_days(self):
        """More days since interaction → lower recency score."""
        df = pd.DataFrame({"days_since_interaction": [0, 7, 14, 30, 60]})
        scores = LeadFeatureEngineering.create_recency_score(df)
        assert list(scores) == sorted(scores, reverse=True)

    def test_zero_engagement_all_zeros(self):
        df = pd.DataFrame(
            {
                "website_visits": [0],
                "email_opens": [0],
                "email_clicks": [0],
            }
        )
        score = LeadFeatureEngineering.create_engagement_score(df)
        assert score.iloc[0] == 0.0

    def test_works_on_single_row(self, single_lead_df):
        result = LeadFeatureEngineering.engineer_features(single_lead_df)
        assert len(result) == 1
        assert "engagement_score" in result.columns
