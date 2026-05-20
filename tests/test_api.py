"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lead_scoring.api.main import create_app

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client_with_model(model_bundle):
    """Test client with a real trained model loaded into app state."""
    app = create_app()
    app.state.bundle = model_bundle

    # Override lifespan so the fixture-loaded bundle is not replaced
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="module")
def client_no_model():
    """Test client with no model — simulates cold start before training."""
    app = create_app()
    app.state.bundle = None
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


GOOD_LEAD = {
    "lead_id": "TEST001",
    "lead_source": "Referral",
    "industry": "SaaS",
    "company_size": "Enterprise",
    "region": "North America",
    "website_visits": 12,
    "email_opens": 6,
    "email_clicks": 3,
    "demo_requested": 1,
    "days_since_interaction": 2,
    "followup_count": 3,
}


# ── Health ─────────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200_with_model(self, client_with_model):
        resp = client_with_model.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_body_model_loaded(self, client_with_model):
        body = client_with_model.get("/api/v1/health").json()
        assert body["model_loaded"] is True
        assert body["status"] == "ok"

    def test_health_returns_200_without_model(self, client_no_model):
        resp = client_no_model.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_body_model_not_loaded(self, client_no_model):
        body = client_no_model.get("/api/v1/health").json()
        assert body["model_loaded"] is False


# ── Single score ───────────────────────────────────────────────────────────────


class TestScoreEndpoint:
    def test_score_returns_200(self, client_with_model):
        resp = client_with_model.post("/api/v1/score", json=GOOD_LEAD)
        assert resp.status_code == 200

    def test_score_response_structure(self, client_with_model):
        body = client_with_model.post("/api/v1/score", json=GOOD_LEAD).json()
        assert "scored_leads" in body
        assert len(body["scored_leads"]) == 1

    def test_score_fields_present(self, client_with_model):
        lead = client_with_model.post("/api/v1/score", json=GOOD_LEAD).json()["scored_leads"][0]
        assert "lead_id" in lead
        assert "lead_quality_score" in lead
        assert "conversion_probability" in lead
        assert "lead_category" in lead
        assert "recommended_action" in lead

    def test_score_range(self, client_with_model):
        lead = client_with_model.post("/api/v1/score", json=GOOD_LEAD).json()["scored_leads"][0]
        assert 0 <= lead["lead_quality_score"] <= 100
        assert 0.0 <= lead["conversion_probability"] <= 1.0

    def test_score_valid_category(self, client_with_model):
        lead = client_with_model.post("/api/v1/score", json=GOOD_LEAD).json()["scored_leads"][0]
        assert lead["lead_category"] in ("Hot", "Warm", "Cold")

    def test_score_without_model_returns_503(self, client_no_model):
        resp = client_no_model.post("/api/v1/score", json=GOOD_LEAD)
        assert resp.status_code == 503

    def test_score_invalid_payload_returns_422(self, client_with_model):
        resp = client_with_model.post("/api/v1/score", json={"lead_id": "X"})
        assert resp.status_code == 422

    def test_score_invalid_lead_source_returns_422(self, client_with_model):
        bad_lead = {**GOOD_LEAD, "lead_source": "Smoke Signal"}
        resp = client_with_model.post("/api/v1/score", json=bad_lead)
        assert resp.status_code == 422

    def test_score_negative_visits_returns_422(self, client_with_model):
        bad_lead = {**GOOD_LEAD, "website_visits": -5}
        resp = client_with_model.post("/api/v1/score", json=bad_lead)
        assert resp.status_code == 422


# ── Batch score ────────────────────────────────────────────────────────────────


class TestBatchScoreEndpoint:
    def test_batch_returns_200(self, client_with_model):
        payload = {"leads": [GOOD_LEAD, {**GOOD_LEAD, "lead_id": "TEST002", "demo_requested": 0}]}
        resp = client_with_model.post("/api/v1/score/batch", json=payload)
        assert resp.status_code == 200

    def test_batch_returns_correct_count(self, client_with_model):
        payload = {"leads": [GOOD_LEAD, {**GOOD_LEAD, "lead_id": "TEST002"}]}
        body = client_with_model.post("/api/v1/score/batch", json=payload).json()
        assert len(body["scored_leads"]) == 2

    def test_batch_sorted_by_score_desc(self, client_with_model):
        leads = [{**GOOD_LEAD, "lead_id": f"L{i}", "demo_requested": i % 2, "website_visits": i} for i in range(5)]
        body = client_with_model.post("/api/v1/score/batch", json={"leads": leads}).json()
        scores = [sl["lead_quality_score"] for sl in body["scored_leads"]]
        assert scores == sorted(scores, reverse=True)

    def test_batch_empty_list_returns_422(self, client_with_model):
        resp = client_with_model.post("/api/v1/score/batch", json={"leads": []})
        assert resp.status_code == 422


# ── Root ───────────────────────────────────────────────────────────────────────


class TestRoot:
    def test_root_returns_200(self, client_with_model):
        resp = client_with_model.get("/")
        assert resp.status_code == 200
