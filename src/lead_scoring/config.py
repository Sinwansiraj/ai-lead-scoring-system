"""
Application configuration via environment variables.
All settings can be overridden with a .env file or shell exports.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=("settings_",),  # allow model_dir / model_filename fields
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    model_dir: Path = Field(default=Path("models"), description="Directory for saved model artifacts")
    model_filename: str = Field(default="lead_scorer.joblib", description="Filename for the serialized model bundle")

    # ── Data ───────────────────────────────────────────────────────────────
    n_samples: int = Field(default=5000, ge=100, description="Number of synthetic CRM records to generate")
    conversion_rate: float = Field(default=0.12, gt=0.0, lt=1.0, description="Target base conversion rate")
    random_seed: int = Field(default=42)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)

    # ── Model ──────────────────────────────────────────────────────────────
    xgb_n_estimators: int = Field(default=200)
    xgb_max_depth: int = Field(default=6)
    xgb_learning_rate: float = Field(default=0.05)
    xgb_subsample: float = Field(default=0.8)
    xgb_colsample_bytree: float = Field(default=0.8)

    # ── API ────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)
    api_workers: int = Field(default=1)
    log_level: str = Field(default="INFO")

    # ── Scoring thresholds ─────────────────────────────────────────────────
    hot_threshold: int = Field(default=80, ge=0, le=100)
    warm_threshold: int = Field(default=50, ge=0, le=100)

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename


# Singleton — import this everywhere
settings = Settings()
