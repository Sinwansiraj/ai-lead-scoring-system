"""
Pydantic request / response schemas for the Lead Scoring API.

Keeping schemas in a dedicated module makes it easy to version the API
(v1 schemas live here; v2 schemas can live in schemas_v2.py).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class LeadSource(str, Enum):
    website = "Website"
    referral = "Referral"
    linkedin_ads = "LinkedIn Ads"
    cold_call = "Cold Call"
    conference = "Conference"
    partner = "Partner"


class Industry(str, Enum):
    saas = "SaaS"
    fintech = "Fintech"
    ecommerce = "E-commerce"
    healthcare = "Healthcare"
    manufacturing = "Manufacturing"
    consulting = "Consulting"


class CompanySize(str, Enum):
    smb = "SMB"
    mid_market = "Mid-Market"
    enterprise = "Enterprise"


class Region(str, Enum):
    north_america = "North America"
    europe = "Europe"
    asia_pacific = "Asia Pacific"
    latin_america = "Latin America"


class LeadCategory(str, Enum):
    hot = "Hot"
    warm = "Warm"
    cold = "Cold"


# ── Request schemas ────────────────────────────────────────────────────────────

class LeadFeatures(BaseModel):
    """
    Raw CRM fields for a single lead.

    All fields map 1-to-1 to the columns produced by ``generate_crm_data``.
    """

    lead_id: str = Field(..., description="Unique lead identifier, e.g. 'L00042'")
    lead_source: LeadSource
    industry: Industry
    company_size: CompanySize
    region: Region
    website_visits: Annotated[int, Field(ge=0, description="Number of website visits")] = 0
    email_opens: Annotated[int, Field(ge=0)] = 0
    email_clicks: Annotated[int, Field(ge=0)] = 0
    demo_requested: Annotated[int, Field(ge=0, le=1, description="1 if the lead requested a demo")] = 0
    days_since_interaction: Annotated[int, Field(ge=0)] = 0
    followup_count: Annotated[int, Field(ge=0)] = 0

    model_config = {"use_enum_values": True}


class BatchScoreRequest(BaseModel):
    leads: list[LeadFeatures] = Field(..., min_length=1, max_length=1000)


# ── Response schemas ───────────────────────────────────────────────────────────

class ScoredLead(BaseModel):
    lead_id: str
    lead_quality_score: Annotated[int, Field(ge=0, le=100)]
    conversion_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    lead_category: LeadCategory
    recommended_action: str
    engagement_score: float
    recency_score: float

    model_config = {"use_enum_values": True}


class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    scored_leads: list[ScoredLead]
    model_version: str
    model_roc_auc: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_roc_auc: float | None = None
    version: str
