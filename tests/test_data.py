"""Tests for data generation and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_scoring.data.generator import generate_crm_data
from lead_scoring.data.preprocessor import LeadDataPreprocessor

# ── Generator ──────────────────────────────────────────────────────────────────

class TestGenerateCrmData:
    def test_returns_dataframe(self):
        df = generate_crm_data(n_samples=100)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = generate_crm_data(n_samples=150)
        assert len(df) == 150

    def test_has_required_columns(self):
        expected = {
            "lead_id", "lead_source", "industry", "company_size", "region",
            "website_visits", "email_opens", "email_clicks", "demo_requested",
            "days_since_interaction", "followup_count", "converted",
        }
        df = generate_crm_data(n_samples=50)
        assert expected.issubset(set(df.columns))

    def test_binary_target(self):
        df = generate_crm_data(n_samples=200)
        assert set(df["converted"].unique()).issubset({0, 1})

    def test_conversion_rate_close_to_target(self):
        df = generate_crm_data(n_samples=2000, conversion_rate=0.12)
        actual_rate = df["converted"].mean()
        assert 0.08 <= actual_rate <= 0.18, f"Conversion rate {actual_rate:.2%} out of expected range"

    def test_reproducible_with_same_seed(self):
        df1 = generate_crm_data(n_samples=100, random_seed=99)
        df2 = generate_crm_data(n_samples=100, random_seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_crm_data(n_samples=100, random_seed=1)
        df2 = generate_crm_data(n_samples=100, random_seed=2)
        assert not df1["lead_id"].equals(df2["lead_id"]) or not df1["converted"].equals(df2["converted"])

    def test_no_null_values(self):
        df = generate_crm_data(n_samples=100)
        assert df.isnull().sum().sum() == 0

    def test_non_negative_numeric_fields(self):
        df = generate_crm_data(n_samples=200)
        for col in ["website_visits", "email_opens", "email_clicks", "followup_count", "days_since_interaction"]:
            assert (df[col] >= 0).all(), f"Column {col} has negative values"


# ── Preprocessor ───────────────────────────────────────────────────────────────

class TestLeadDataPreprocessor:
    def test_fit_transform_returns_dataframe(self, engineered_df):
        preprocessor = LeadDataPreprocessor()
        result = preprocessor.fit_transform(engineered_df)
        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_sets_is_fitted(self, engineered_df):
        preprocessor = LeadDataPreprocessor()
        preprocessor.fit_transform(engineered_df)
        assert preprocessor.is_fitted

    def test_encoded_columns_created(self, engineered_df):
        preprocessor = LeadDataPreprocessor()
        result = preprocessor.fit_transform(engineered_df)
        expected_encoded = ["lead_source_encoded", "industry_encoded", "company_size_encoded", "region_encoded"]
        for col in expected_encoded:
            assert col in result.columns, f"Missing encoded column: {col}"

    def test_transform_without_fit_raises(self, engineered_df):
        preprocessor = LeadDataPreprocessor()
        with pytest.raises(RuntimeError, match="fitted"):
            preprocessor.transform(engineered_df)

    def test_transform_consistent_with_fit_transform(self, engineered_df):
        preprocessor = LeadDataPreprocessor()
        train_result = preprocessor.fit_transform(engineered_df)
        # transform on same data should equal fit_transform output
        test_result = preprocessor.transform(engineered_df)
        pd.testing.assert_frame_equal(
            train_result[preprocessor.numeric_cols],
            test_result[preprocessor.numeric_cols],
        )

    def test_handles_missing_numeric_values(self, engineered_df):
        df_with_nulls = engineered_df.copy()
        df_with_nulls.loc[df_with_nulls.index[0], "website_visits"] = np.nan
        preprocessor = LeadDataPreprocessor()
        result = preprocessor.fit_transform(df_with_nulls)
        assert result.isnull().sum().sum() == 0

    def test_unseen_category_does_not_raise(self, engineered_df):
        preprocessor = LeadDataPreprocessor()
        preprocessor.fit_transform(engineered_df)
        df_unseen = engineered_df.copy()
        df_unseen["lead_source"] = "Unknown Source"  # unseen category
        # Should not raise; falls back to first known class
        result = preprocessor.transform(df_unseen)
        assert "lead_source_encoded" in result.columns
