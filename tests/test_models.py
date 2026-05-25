"""Tests for model training, scoring, and persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from lead_scoring.models.scorer import LeadScorer
from lead_scoring.models.trainer import LeadScoringTrainer, ModelBundle

# ── Scorer ─────────────────────────────────────────────────────────────────────


class TestLeadScorer:
    # ── Raw probability→score helper (unchanged utility) ───────────────────

    @pytest.mark.parametrize(
        "prob,expected",
        [
            (0.0, 0),
            (0.5, 50),
            (1.0, 100),
            (0.856, 86),
            (0.001, 0),
        ],
    )
    def test_probability_to_score(self, prob, expected):
        assert LeadScorer.probability_to_score(prob) == expected

    # ── Composite score blending ───────────────────────────────────────────

    def test_composite_score_high_prob_dominates(self):
        """High probability lead stays high even with average behavioural signals."""
        score = LeadScorer.composite_score(0.95, engagement_score=50.0, recency_score=50.0)
        # 0.60*95 + 0.25*50 + 0.15*50 = 57 + 12.5 + 7.5 = 77
        assert score == 77

    def test_composite_score_engagement_boosts_cold_lead(self):
        """L0080 pattern: prob≈4% but eng=100 and rec=61 should reach Warm via floor."""
        score = LeadScorer.composite_score(0.0414, engagement_score=100.0, recency_score=60.65)
        # composite = 0.60*4.14 + 0.25*100 + 0.15*60.65 ≈ 37
        # floor     = 0.60 * (0.60*100 + 0.40*60.65)    ≈ 51  ← floor wins
        assert score >= 50, f"Expected ≥50 (Warm) for highly engaged lead, got {score}"

    def test_composite_score_floor_does_not_affect_high_prob_leads(self):
        """Behavioural floor must never lower a score that the composite already earned."""
        score_no_floor = round(0.60 * 97.16 + 0.25 * 78.08 + 0.15 * 81.87)  # L0420 ≈ 90
        score = LeadScorer.composite_score(0.9716, engagement_score=78.08, recency_score=81.87)
        assert score >= score_no_floor, "Floor should never reduce a strong lead's score"

    def test_composite_score_floor_zero_for_disengaged_lead(self):
        """Lead with eng=0 and rec≈0 gets no floor boost."""
        score = LeadScorer.composite_score(0.009, engagement_score=0.0, recency_score=6.72)
        assert score <= 5, f"Dead lead should still score near-zero, got {score}"

    def test_composite_score_pure_model_via_explicit_weights(self):
        """With 100% weight on probability, composite == probability_to_score."""
        score = LeadScorer.composite_score(0.73, 0.0, 0.0, prob_weight=1.0, engagement_weight=0.0, recency_weight=0.0)
        assert score == 73

    def test_composite_score_clamps_to_0_100(self):
        assert LeadScorer.composite_score(0.0, 0.0, 0.0) == 0
        assert LeadScorer.composite_score(1.0, 100.0, 100.0) == 100

    def test_composite_score_recency_matters(self):
        """Same prob + engagement, but fresher lead should score higher."""
        fresh = LeadScorer.composite_score(0.40, engagement_score=60.0, recency_score=100.0)
        stale = LeadScorer.composite_score(0.40, engagement_score=60.0, recency_score=0.0)
        assert fresh > stale

    def test_composite_score_engagement_matters(self):
        """Same prob + recency, but more engaged lead should score higher."""
        active = LeadScorer.composite_score(0.30, engagement_score=100.0, recency_score=50.0)
        dormant = LeadScorer.composite_score(0.30, engagement_score=0.0, recency_score=50.0)
        assert active > dormant

    # ── Category thresholds ────────────────────────────────────────────────

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

    # ── Recommended action ─────────────────────────────────────────────────

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

    # ── score_dataframe ────────────────────────────────────────────────────

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

    def test_score_dataframe_uses_engagement_and_recency(self, engineered_df):
        """Two leads with identical probability but very different engagement should score differently."""
        base = engineered_df.iloc[:2].copy()
        probs = [0.20, 0.20]

        # Force divergent behavioural signals directly on the rows
        base.iloc[0, base.columns.get_loc("engagement_score")] = 100.0
        base.iloc[0, base.columns.get_loc("recency_score")] = 100.0
        base.iloc[1, base.columns.get_loc("engagement_score")] = 0.0
        base.iloc[1, base.columns.get_loc("recency_score")] = 0.0

        result = LeadScorer.score_dataframe(base, probs)
        assert result["lead_quality_score"].max() > result["lead_quality_score"].min()


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
        assert results["production"].roc_auc >= 0.70, f"ROC-AUC {results['production'].roc_auc:.4f} below floor of 0.70"

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
