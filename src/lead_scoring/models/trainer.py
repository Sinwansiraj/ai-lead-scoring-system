"""
Model training, evaluation, and persistence.

Two models are trained:
- Logistic Regression: interpretable baseline
- XGBoost: production-grade gradient-boosted ensemble

The ``ModelBundle`` dataclass holds everything needed to score new leads
and can be serialised / deserialised with ``joblib``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from lead_scoring.config import settings
from lead_scoring.data.preprocessor import LeadDataPreprocessor
from lead_scoring.features.engineering import LeadFeatureEngineering
from lead_scoring.utils.logging import get_logger

logger = get_logger(__name__)

# Columns that must be excluded from the feature matrix
_EXCLUDE_FROM_FEATURES = {
    "lead_id", "converted",
    "lead_source", "industry", "company_size", "region",
}


@dataclass
class EvaluationResult:
    """Structured evaluation output for one model."""

    model_name: str
    roc_auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    cv_roc_auc_mean: float = 0.0
    cv_roc_auc_std: float = 0.0

    def summary(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"{self.model_name} — Evaluation Summary",
            f"{'=' * 60}",
            f"  ROC-AUC  : {self.roc_auc:.4f}",
            f"  Precision: {self.precision:.4f}",
            f"  Recall   : {self.recall:.4f}",
            f"  F1       : {self.f1:.4f}",
            f"  CV AUC   : {self.cv_roc_auc_mean:.4f} ± {self.cv_roc_auc_std:.4f}",
        ]
        cm = self.confusion_matrix
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        lines += [
            "",
            "  Business Impact:",
            f"    True Positives  (hot leads found)   : {tp}",
            f"    False Positives (wasted sales calls) : {fp}",
            f"    False Negatives (missed revenue)     : {fn}",
            f"    True Negatives  (cold leads filtered): {tn}",
        ]
        return "\n".join(lines)


@dataclass
class ModelBundle:
    """
    Self-contained bundle that can score new leads end-to-end.

    Holds both the feature-engineering step and the fitted preprocessor so
    a single ``joblib.load`` call is all that is needed at inference time.
    """

    feature_engineer: LeadFeatureEngineering
    preprocessor: LeadDataPreprocessor
    production_model: XGBClassifier
    baseline_model: LogisticRegression
    feature_cols: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return conversion probabilities for a raw (un-engineered) DataFrame."""
        df_fe = self.feature_engineer.engineer_features(df)
        df_proc = self.preprocessor.transform(df_fe)
        X = df_proc[self.feature_cols].values
        return self.production_model.predict_proba(X)[:, 1]


class LeadScoringTrainer:
    """
    Orchestrates the full training pipeline and produces a ``ModelBundle``.
    """

    def train(self, df: pd.DataFrame) -> tuple[ModelBundle, dict[str, EvaluationResult]]:
        """
        Run end-to-end training on a raw CRM DataFrame.

        Returns
        -------
        bundle:
            Fitted ``ModelBundle`` ready for serialisation.
        results:
            Evaluation results keyed by ``"baseline"`` and ``"production"``.
        """
        logger.info("Starting training pipeline on %d rows", len(df))

        # 1. Feature engineering
        fe = LeadFeatureEngineering()
        df_fe = fe.engineer_features(df)

        # 2. Train / test split
        feature_cols = [
            c for c in df_fe.columns if c not in _EXCLUDE_FROM_FEATURES
        ]
        X_raw = df_fe[feature_cols]
        y = df_fe["converted"]

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y,
            test_size=settings.test_size,
            random_state=settings.random_seed,
            stratify=y,
        )
        logger.info("Train: %d  Test: %d", len(X_train_raw), len(X_test_raw))

        # 3. Preprocessing (fit on train only)
        preprocessor = LeadDataPreprocessor()
        train_proc = preprocessor.fit_transform(
            df_fe.loc[X_train_raw.index]
        )
        test_proc = preprocessor.transform(
            df_fe.loc[X_test_raw.index]
        )
        X_train = train_proc[feature_cols].values
        X_test = test_proc[feature_cols].values

        # 4. Train models
        baseline = self._train_baseline(X_train, y_train)
        production = self._train_production(X_train, y_train)

        # 5. Evaluate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.random_seed)
        results: dict[str, EvaluationResult] = {
            "baseline": self._evaluate(baseline, "Logistic Regression (Baseline)", X_train, X_test, y_train, y_test, cv),
            "production": self._evaluate(production, "XGBoost (Production)", X_train, X_test, y_train, y_test, cv),
        }

        for name, res in results.items():
            logger.info("%s\n%s", name.upper(), res.summary())

        # 6. Build bundle
        bundle = ModelBundle(
            feature_engineer=fe,
            preprocessor=preprocessor,
            production_model=production,
            baseline_model=baseline,
            feature_cols=feature_cols,
            metadata={
                "trained_on_rows": len(df),
                "production_roc_auc": results["production"].roc_auc,
                "baseline_roc_auc": results["baseline"].roc_auc,
            },
        )
        return bundle, results

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _train_baseline(X_train: np.ndarray, y_train: pd.Series) -> LogisticRegression:
        logger.info("Training Logistic Regression baseline…")
        model = LogisticRegression(
            random_state=settings.random_seed,
            max_iter=1000,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def _train_production(X_train: np.ndarray, y_train: pd.Series) -> XGBClassifier:
        logger.info("Training XGBoost production model…")
        scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        model = XGBClassifier(
            n_estimators=settings.xgb_n_estimators,
            max_depth=settings.xgb_max_depth,
            learning_rate=settings.xgb_learning_rate,
            subsample=settings.xgb_subsample,
            colsample_bytree=settings.xgb_colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=settings.random_seed,
            eval_metric="auc",
            verbosity=0,
        )
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def _evaluate(
        model: Any,
        name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series,
        cv: StratifiedKFold,
    ) -> EvaluationResult:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred).tolist()
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        return EvaluationResult(
            model_name=name,
            roc_auc=roc_auc_score(y_test, y_proba),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            confusion_matrix=cm,
            cv_roc_auc_mean=float(cv_scores.mean()),
            cv_roc_auc_std=float(cv_scores.std()),
        )

    # ── Persistence ────────────────────────────────────────────────────────

    @staticmethod
    def save(bundle: ModelBundle, path: Path | None = None) -> Path:
        """Serialise the model bundle with joblib."""
        save_path = path or settings.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, save_path)
        logger.info("Model bundle saved → %s", save_path)
        return save_path

    @staticmethod
    def load(path: Path | None = None) -> ModelBundle:
        """Deserialise a model bundle from disk."""
        load_path = path or settings.model_path
        if not load_path.exists():
            raise FileNotFoundError(
                f"No model found at '{load_path}'. Run the training script first:\n"
                "  python scripts/train.py"
            )
        bundle: ModelBundle = joblib.load(load_path)
        logger.info("Model bundle loaded ← %s  (AUC=%.4f)", load_path, bundle.metadata.get("production_roc_auc", 0))
        return bundle
