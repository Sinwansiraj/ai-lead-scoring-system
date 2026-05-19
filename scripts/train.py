"""
Standalone training script.

Run:
    python scripts/train.py

Environment overrides (via .env or shell):
    N_SAMPLES=10000 CONVERSION_RATE=0.15 python scripts/train.py
"""

import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lead_scoring.config import settings
from lead_scoring.data.generator import generate_crm_data
from lead_scoring.models.trainer import LeadScoringTrainer
from lead_scoring.utils.logging import get_logger

logger = get_logger("train")


def main() -> None:
    logger.info("=" * 60)
    logger.info("Lead Scoring — Training Pipeline")
    logger.info("=" * 60)

    # 1. Generate / load data
    logger.info("Generating synthetic CRM data…")
    df = generate_crm_data()
    logger.info("Dataset: %d rows, conversion rate=%.1f%%", len(df), df["converted"].mean() * 100)

    # 2. Train
    trainer = LeadScoringTrainer()
    bundle, results = trainer.train(df)

    # 3. Assert minimum quality bar before saving
    prod_auc = results["production"].roc_auc
    if prod_auc < 0.80:
        logger.error(
            "Production model ROC-AUC %.4f is below the 0.80 threshold. "
            "Model NOT saved. Check your features and data.",
            prod_auc,
        )
        sys.exit(1)

    # 4. Save
    saved_path = trainer.save(bundle)
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Production ROC-AUC : %.4f", prod_auc)
    logger.info("  Baseline  ROC-AUC  : %.4f", results["baseline"].roc_auc)
    logger.info("  Model saved → %s", saved_path)
    logger.info("=" * 60)
    logger.info("Start the API:  uvicorn lead_scoring.api.main:app --reload")


if __name__ == "__main__":
    main()
