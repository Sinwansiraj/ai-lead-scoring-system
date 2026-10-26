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

    # ── Security ───────────────────────────────────────────────────────────
    allowed_origins: str = Field(
        default="https://ai-lead-scoring-system.vercel.app,http://localhost:5173,http://localhost:3000",
        description="Comma-separated list of browser origins permitted by CORS",
    )
    rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        description="Max requests per client IP per minute on non-exempt endpoints",
    )

    # ── Scoring thresholds ─────────────────────────────────────────────────
    hot_threshold: int = Field(default=80, ge=0, le=100)
    warm_threshold: int = Field(default=50, ge=0, le=100)

    # ── Composite score blend weights (should sum to 1.0) ──────────────────
    score_prob_weight: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Weight for model conversion probability in the composite quality score",
    )
    score_engagement_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for engagement score (website visits, email clicks) in composite score",
    )
    score_recency_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for recency score (time-decay since last interaction) in composite score",
    )
    score_floor_multiplier: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description=(
            "Behavioural floor multiplier. A lead's final score is always at least "
            "floor_multiplier × (0.6 × engagement + 0.4 × recency), preventing highly "
            "engaged recent leads from being buried by a low model probability alone."
        ),
    )

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_filename

    @property
    def allowed_origins_list(self) -> list[str]:
        """CORS origins parsed from the comma-separated ``allowed_origins`` string."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


# Singleton — import this everywhere
settings = Settings()
