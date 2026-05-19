"""
Data preprocessing pipeline: missing-value handling, encoding, scaling.

The ``LeadDataPreprocessor`` follows a fit/transform pattern so the same
transformations learned on training data can be applied to inference data
without leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from lead_scoring.utils.logging import get_logger

logger = get_logger(__name__)

CATEGORICAL_COLS = ["lead_source", "industry", "company_size", "region"]
NON_FEATURE_COLS = {"lead_id", "converted"}


class LeadDataPreprocessor:
    """
    Stateful preprocessor that fits on training data and transforms any split.

    Attributes
    ----------
    label_encoders:
        Per-column ``LabelEncoder`` objects fitted during ``fit_transform``.
    scaler:
        ``StandardScaler`` fitted on numeric columns during ``fit_transform``.
    numeric_cols:
        List of numeric column names that will be scaled (set during fit).
    is_fitted:
        Whether ``fit_transform`` has been called.
    """

    def __init__(self) -> None:
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.numeric_cols: list[str] = []
        self.is_fitted: bool = False

    # ── Public API ─────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *training* data and return transformed copy."""
        df = df.copy()
        df = self._fill_missing(df)
        df = self._encode_categoricals(df, fit=True)
        df = self._scale_numerics(df, fit=True)
        self.is_fitted = True
        logger.info("Preprocessor fitted on %d rows, %d features", len(df), len(self.numeric_cols))
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously fitted transformations to *new* data."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform(). Call fit_transform() first.")
        df = df.copy()
        df = self._fill_missing(df)
        df = self._encode_categoricals(df, fit=False)
        df = self._scale_numerics(df, fit=False)
        return df

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Business rule: missing engagement metrics treated as zero activity."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    raise ValueError(f"No fitted encoder for column '{col}'.")
                # Handle unseen categories gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
                df[f"{col}_encoded"] = le.transform(df[col])
        return df

    def _scale_numerics(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit:
            self.numeric_cols = [
                c for c in df.columns
                if df[c].dtype in ("int64", "float64") and c not in NON_FEATURE_COLS
            ]
        if self.numeric_cols:
            if fit:
                df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])
            else:
                df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        return df
