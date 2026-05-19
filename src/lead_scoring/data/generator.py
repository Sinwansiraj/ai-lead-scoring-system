"""
Synthetic CRM lead data generator.

Simulates a realistic B2B SaaS pipeline where:
- Conversion rate is ~12%
- Different lead sources and industries carry different conversion propensity
- Engagement signals (visits, email clicks, demo requests) correlate with conversion
"""

import numpy as np
import pandas as pd

from lead_scoring.config import settings
from lead_scoring.utils.logging import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

LEAD_SOURCES = ["Website", "Referral", "LinkedIn Ads", "Cold Call", "Conference", "Partner"]
LEAD_SOURCE_PROBS = [0.35, 0.20, 0.25, 0.10, 0.05, 0.05]
LEAD_SOURCE_WEIGHTS = {
    "Referral": 25, "Website": 15, "Conference": 20,
    "LinkedIn Ads": 10, "Cold Call": 5, "Partner": 20,
}

INDUSTRIES = ["SaaS", "Fintech", "E-commerce", "Healthcare", "Manufacturing", "Consulting"]
INDUSTRY_PROBS = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
INDUSTRY_WEIGHTS = {
    "SaaS": 15, "Fintech": 18, "E-commerce": 12,
    "Healthcare": 10, "Manufacturing": 8, "Consulting": 7,
}

COMPANY_SIZES = ["SMB", "Mid-Market", "Enterprise"]
COMPANY_SIZE_PROBS = [0.50, 0.35, 0.15]
COMPANY_SIZE_WEIGHTS = {"SMB": 10, "Mid-Market": 20, "Enterprise": 30}

REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America"]
REGION_PROBS = [0.45, 0.30, 0.15, 0.10]


def generate_crm_data(
    n_samples: int | None = None,
    conversion_rate: float | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic CRM lead dataset.

    Parameters
    ----------
    n_samples:
        Number of lead records to generate. Defaults to ``settings.n_samples``.
    conversion_rate:
        Target proportion of converted leads. Defaults to ``settings.conversion_rate``.
    random_seed:
        NumPy random seed for reproducibility. Defaults to ``settings.random_seed``.

    Returns
    -------
    pd.DataFrame
        One row per lead with raw CRM features and a ``converted`` binary target.
    """
    n = n_samples or settings.n_samples
    rate = conversion_rate or settings.conversion_rate
    seed = random_seed if random_seed is not None else settings.random_seed

    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic CRM leads (target conversion rate=%.1f%%)", n, rate * 100)

    lead_sources = rng.choice(LEAD_SOURCES, size=n, p=LEAD_SOURCE_PROBS)
    industries = rng.choice(INDUSTRIES, size=n, p=INDUSTRY_PROBS)
    company_sizes = rng.choice(COMPANY_SIZES, size=n, p=COMPANY_SIZE_PROBS)
    regions = rng.choice(REGIONS, size=n, p=REGION_PROBS)

    website_visits = rng.poisson(lam=5, size=n)
    email_opens = rng.poisson(lam=3, size=n)
    email_clicks = rng.poisson(lam=1, size=n)
    demo_requested = rng.choice([0, 1], size=n, p=[0.70, 0.30])
    days_since_interaction = rng.exponential(scale=15, size=n).astype(int)
    followup_count = rng.poisson(lam=2, size=n)

    df = pd.DataFrame(
        {
            "lead_id": [f"L{str(i).zfill(5)}" for i in range(n)],
            "lead_source": lead_sources,
            "industry": industries,
            "company_size": company_sizes,
            "region": regions,
            "website_visits": website_visits,
            "email_opens": email_opens,
            "email_clicks": email_clicks,
            "demo_requested": demo_requested,
            "days_since_interaction": days_since_interaction,
            "followup_count": followup_count,
        }
    )

    # Deterministic conversion signal based on business rules + noise
    conversion_score = (
        df["demo_requested"] * 40
        + df["website_visits"].clip(upper=20) * 2
        + df["email_clicks"].clip(upper=10) * 5
        + df["company_size"].map(COMPANY_SIZE_WEIGHTS)
        + df["lead_source"].map(LEAD_SOURCE_WEIGHTS)
        + df["industry"].map(INDUSTRY_WEIGHTS)
        - df["days_since_interaction"].clip(upper=60) * 0.5
        + rng.normal(0, 15, n)
    )

    threshold = np.percentile(conversion_score, (1 - rate) * 100)
    df["converted"] = (conversion_score > threshold).astype(int)

    actual_rate = df["converted"].mean()
    logger.info(
        "Dataset ready: %d leads, actual conversion rate=%.1f%%",
        len(df),
        actual_rate * 100,
    )
    return df
