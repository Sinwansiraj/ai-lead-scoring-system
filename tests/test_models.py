"""Tests for model training, scoring, and persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lead_scoring.models.scorer import LeadScorer
from lead_scoring.models.trainer import LeadScoringTrainer, ModelBundle


# ── Scorer ─────────────────────────────────────────────────────────────────────

class TestLeadScorer:
    @pytest.mark.parametrize("prob,expected", [
        (0.0, 0), (0.5, 50), (1.0, 100), (0.856, 86), (0.001, 0),
    ])
    def test_probability_to_score(self, prob, expected):
        assert LeadScorer.probability_to_score(prob) == expected

    def test_score_to_category_hot(self):
        assert LeadScorer.score_to_category(90) == "Hot"

    def test_score_to_category_warm(self):
        assert LeadScorer.score_to_category(65) == "Warm"

    def test_score_to_category_cold(self):
        assert LeadScorer.score_to_category(30) == "Cold"

    def test_score_to_category_boundary_hot(self):
        assert LeadScorer.score_to_category(80) == "Hot"

    def test_score_to_category_boundary_warm(self):
        assert LeadScorer.score_to_category(50) == "Warm"

    def test_recommend_action_hot_recent(self):
        action = LeadScorer.recommend_action("Hot", engagement_score=70, recency_score=80)
        assert "Call" in action

    def test_recommend_action_hot_stale(self):
        action = LeadScorer.recommend_action("Hot", engagement_score=70, recency_score=50)
        assert "Re-engagement" in action

    def test_recommend_action_warm_high_engagement(self):
        action = LeadScorer.recommend_action("Warm", engagement_score=70, recency_score=60)
        assert "Demo" in action

    def test_recommend_action_cold_stale(self):
        action = LeadScorer.recommend_action("Cold", engagement_score=20, recency_score=10)
        assert "Re-qualification" in action

    def test_score_dataframe_columns(self, engineered_df):
        probs = np.random.default_rng(0).uniform(0, 1, len(engineered_df))
        result = LeadScorer.score_dataframe(engineered_df, probs)
        for col in ["conversion_probability", "lead_quality_score", "lead_category", "recommended_action"]:
            assert col in result.columns

    def test_score_dataframe_sorted_descending(self, engineered_df):
        probs = np.random.default_rng(1).uniform(0, 1, len(engineered_df))
        result = LeadScorer.score_dataframe(engineered_df, probs)
        scores = result["lead_quality_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_score_dataframe_no_nulls(self, engineered_df):
        probs = np.random.default_rng(2).uniform(0, 1, len(engineered_df))
        result = LeadScorer.score_dataframe(engineered_df, probs)
        assert result[["lead_quality_score", "lead_category", "recommended_action"]].isnull().sum().sum() == 0


# ── Trainer ────────────────────────────────────────────────────────────────────

class TestLeadScoringTrainer:
    def test_train_returns_bundle_and_results(self, raw_df):
        trainer = LeadScoringTrainer()
        bundle, results = trainer.train(raw_df)
        assert isinstance(bundle, ModelBundle)
        assert "baseline" in results
        assert "production" in results

    def test_production_auc_above_threshold(self, model_bundle, raw_df):
        trainer = LeadScoringTrainer()
        _, results = trainer.train(raw_df)
        assert results["production"].roc_auc >= 0.70, (
            f"ROC-AUC {results['production'].roc_auc:.4f} below floor of 0.70"
        )

    def test_bundle_predict_proba_shape(self, model_bundle, single_lead_df):
        probs = model_bundle.predict_proba(single_lead_df)
        assert len(probs) == 1
        assert 0.0 <= probs[0] <= 1.0

    def test_bundle_predict_proba_batch(self, model_bundle, raw_df):
        probs = model_bundle.predict_proba(raw_df)
        assert len(probs) == len(raw_df)
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_evaluation_result_summary_contains_auc(self, raw_df):
        trainer = LeadScoringTrainer()
        _, results = trainer.train(raw_df)
        summary = results["production"].summary()
        assert "ROC-AUC" in summary

    def test_save_and_load_roundtrip(self, model_bundle):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_bundle.joblib"
            LeadScoringTrainer.save(model_bundle, path)
            assert path.exists()

            loaded = LeadScoringTrainer.load(path)
            assert isinstance(loaded, ModelBundle)
            assert loaded.metadata == model_bundle.metadata

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="No model found"):
            LeadScoringTrainer.load(Path("/nonexistent/path/model.joblib"))
