"""Comprehensive pytest test suite for EquiTriage.
Validates deterministic scoring tiers, multi-depot routing, trade-constrained bundling,
labor hour ceilings, wet-season passability, RAG context retrieval, and REST API endpoints.
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from starlette.testclient import TestClient

from api import app
from equitriage_engine import (
    Policy,
    compute_manifest_breakdown,
    generate_contractor_work_packet,
    price_of_equity,
    record_coordinator_feedback,
    record_job_resolution,
    retrieve_property_history,
    score_tickets,
)
from geo_utils import (
    find_closest_depot,
    get_coords,
    get_road_status,
    haversine_km,
)


@pytest.fixture
def sample_tickets():
    return pd.DataFrame([
        {
            "ticket_id": "TKT-001",
            "property_id": "PROP-WADEYE-01",
            "location": "Wadeye (Remote)",
            "distance_km": 420.0,
            "days_waiting": 10.0,
            "text": "Main water supply pipe burst in front yard, severe flooding.",
            "category": "plumbing",
            "safety_prob": 0.85,  # Tier 0 Safety Floor
            "det_score": 3.0,
            "chronic_prob": 0.0,
        },
        {
            "ticket_id": "TKT-002",
            "property_id": "PROP-WADEYE-02",
            "location": "Wadeye (Remote)",
            "distance_km": 420.0,
            "days_waiting": 52.0,  # Overdue
            "text": "Laundry drainage trap leaking slowly under tub.",
            "category": "plumbing",
            "safety_prob": 0.10,
            "det_score": 1.0,
            "chronic_prob": 0.0,
        },
        {
            "ticket_id": "TKT-003",
            "property_id": "PROP-WADEYE-03",
            "location": "Wadeye (Remote)",
            "distance_km": 420.0,
            "days_waiting": 15.0,
            "text": "Main electrical breaker sparking violently when stove turned on.",
            "category": "electrical",  # Incompatible trade with plumbing crew
            "safety_prob": 0.30,
            "det_score": 2.0,
            "chronic_prob": 0.0,
        },
        {
            "ticket_id": "TKT-004",
            "property_id": "PROP-ALICE-01",
            "location": "Alice Springs (Remote)",
            "distance_km": 1498.0,
            "days_waiting": 48.0,  # Tier 1 Ageing Guardrail
            "text": "Front door security lock jammed, door cannot be secured.",
            "category": "structural",
            "safety_prob": 0.40,
            "det_score": 2.0,
            "chronic_prob": 0.0,
        },
        {
            "ticket_id": "TKT-005",
            "property_id": "PROP-DARWIN-01",
            "location": "Darwin (Urban)",
            "distance_km": 8.0,
            "days_waiting": 4.0,
            "text": "Pest wasps nesting under front eave.",
            "category": "pest",
            "safety_prob": 0.15,
            "det_score": 1.0,
            "chronic_prob": 0.0,
        },
    ])


def test_tier_classification(sample_tickets):
    """Test hard policy guardrails: Tier 0 safety floor and Tier 1 ageing guardrail."""
    policy = Policy(safety_floor=0.70, max_wait_days=45, guardrail_min_urgency=2.0)
    scored = score_tickets(sample_tickets, policy=policy, jobs_per_week=2)

    # TKT-001 has safety_prob 0.85 >= 0.70 -> Tier 0
    tkt1 = scored.loc[scored.ticket_id == "TKT-001"].iloc[0]
    assert tkt1.tier == 0
    assert tkt1["rank"] == 1

    # TKT-004 has days_waiting 48 > 45 and urgency >= 2.0 -> Tier 1
    tkt4 = scored.loc[scored.ticket_id == "TKT-004"].iloc[0]
    assert tkt4.tier == 1


def test_closest_depot_assignment(sample_tickets):
    """Test multi-depot assignment routes Central Australia jobs to Alice Springs depot."""
    policy = Policy(depot_mode="closest")
    scored = score_tickets(sample_tickets, policy=policy, jobs_per_week=2)

    tkt_alice = scored.loc[scored.ticket_id == "TKT-004"].iloc[0]
    assert tkt_alice.depot_id == "alice_springs"
    assert "Alice Springs" in tkt_alice.depot_name
    # Rather than 1498 km from Darwin, local distance in Alice Springs is under 20 km
    assert tkt_alice.effective_distance_km < 50.0


def test_trade_constrained_trip_bundling(sample_tickets):
    """Test that a Plumbing run bundles secondary plumbing jobs, but rejects electrical jobs."""
    policy = Policy(enforce_trade_matching=True)
    scored = score_tickets(sample_tickets, policy=policy, jobs_per_week=1)

    # TKT-001 is primary dispatch (Plumber)
    tkt1 = scored.loc[scored.ticket_id == "TKT-001"].iloc[0]
    assert "PRIMARY DISPATCH (Plumber)" in tkt1.bundle_status

    # TKT-002 is plumbing in Wadeye -> BUNDLED
    tkt2 = scored.loc[scored.ticket_id == "TKT-002"].iloc[0]
    assert bool(tkt2.is_bundled) is True
    assert "BUNDLED ON RUN" in tkt2.bundle_status
    assert tkt2.avoided_travel_km > 0

    # TKT-003 is electrical in Wadeye -> INCOMPATIBLE TRADE
    tkt3 = scored.loc[scored.ticket_id == "TKT-003"].iloc[0]
    assert bool(tkt3.is_bundled) is False
    assert "Requires Electrician" in tkt3.bundle_status


def test_labor_capacity_ceiling(sample_tickets):
    """Test that jobs exceeding the max crew shift labor budget are not over-bundled."""
    policy = Policy(enforce_trade_matching=False, max_labor_hours_per_run=4.5)
    scored = score_tickets(sample_tickets, policy=policy, jobs_per_week=1)

    tkt1 = scored.loc[scored.ticket_id == "TKT-001"].iloc[0]
    # TKT-001 labor = 2.5 + 3.0*0.5 = 4.0h. Next job would push it to >4.5h
    bundled_count = scored.is_bundled.sum()
    # At most 1 or 0 jobs can be added without exceeding 4.5h
    assert bundled_count <= 1


def test_wet_season_road_passability():
    """Test seasonal road condition matrix for flood-prone NT crossings."""
    # Dry season
    dry_status = get_road_status("gunbalanya", is_wet_season=False)
    assert dry_status["status"] == "OPEN"

    # Wet season
    wet_status = get_road_status("gunbalanya", is_wet_season=True)
    assert wet_status["status"] == "IMPASSABLE"
    assert "Cahills Crossing" in wet_status["reason"]
    assert "Light Aircraft" in wet_status["transit_mode"]
    assert wet_status["access_factor"] > 1.5


def test_chronic_failure_escalation(sample_tickets):
    """Test that chronic failure probability boosts urgency score."""
    sample = sample_tickets.copy()
    sample.loc[sample.ticket_id == "TKT-005", "chronic_prob"] = 0.85

    policy = Policy(chronic_floor=0.60, chronic_weight=0.50)
    scored = score_tickets(sample, policy=policy)

    tkt5 = scored.loc[scored.ticket_id == "TKT-005"].iloc[0]
    assert bool(tkt5.is_chronic) is True
    assert tkt5.chronic_mult > 1.0


def test_record_resolution_and_feedback():
    """Test resolution writing to ledger and feedback writing to JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_ledger = Path(tmpdir) / "test_ledger.json"
        tmp_feedback = Path(tmpdir) / "test_feedback.jsonl"

        res = record_job_resolution(
            ticket_id="TKT-TEST",
            property_id="PROP-999",
            category="plumbing",
            summary="Burst water inlet pipe",
            resolution="Replaced 15mm brass valve and pressure tested",
            technician="Contractor NT-04",
            parts_used="15mm brass valve, teflon tape",
            ledger_path=tmp_ledger,
        )
        assert res["technician"] == "Contractor NT-04"
        assert tmp_ledger.exists()

        # Check retrieval works with newly recorded entry
        context_str, records = retrieve_property_history({"property_id": "PROP-999"}, json.loads(tmp_ledger.read_text()))
        assert len(records) == 1
        assert "Replaced 15mm brass valve" in context_str

        fb = record_coordinator_feedback(
            ticket_id="TKT-TEST",
            original_category="other",
            corrected_category="plumbing",
            original_safety=0.20,
            corrected_safety=0.85,
            reason="High pressure spray near sub-board",
            coordinator="Senior Coordinator Darwin",
            feedback_path=tmp_feedback,
        )
        assert fb["ticket_id"] == "TKT-TEST"
        assert tmp_feedback.exists()


def test_fastapi_rest_endpoints(sample_tickets):
    """Test REST API health, depots, and triage endpoints."""
    client = TestClient(app)

    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # Depots list
    res_depots = client.get("/api/depots")
    assert res_depots.status_code == 200
    assert "darwin" in res_depots.json()["depots"]
    assert "alice_springs" in res_depots.json()["depots"]

    # Triage endpoint
    payload = {
        "tickets": sample_tickets.to_dict(orient="records"),
        "policy": {
            "equity_weight": 0.6,
            "safety_floor": 0.70,
            "max_wait_days": 45,
            "depot_mode": "closest",
            "is_wet_season": False,
            "enforce_trade_matching": True,
            "max_labor_hours_per_run": 12.0,
            "jobs_per_week": 3,
        },
    }
    res_triage = client.post("/api/triage", json=payload)
    assert res_triage.status_code == 200
    data = res_triage.json()
    assert data["count"] == 5
    assert len(data["results"]) == 5


def test_price_of_equity_and_work_packet(sample_tickets):
    """Test price of equity metrics and contractor field work packet markdown generation."""
    policy = Policy(equity_weight=0.6)
    scored = score_tickets(sample_tickets, policy=policy, jobs_per_week=2)
    poe = price_of_equity(scored, jobs_per_week=2, policy=policy)

    # Verify columns and numeric metric values
    assert "Efficiency-first" in poe.columns
    assert "Equity policy" in poe.columns
    assert isinstance(int(poe["Equity policy"]["Fleet cost saved via bundling ($)"]), int)

    # Manifest breakdown
    manifest = compute_manifest_breakdown(scored, jobs_per_week=2, policy=policy)
    assert len(manifest) >= 1

    first_run = manifest.iloc[0].to_dict()
    packet_md = generate_contractor_work_packet(first_run, scored, policy)
    assert "NORTHERN TERRITORY HOUSING // FIELD CONTRACTOR WORK PACKET" in packet_md
    assert "ACCESS & SAFETY BRIEFING" in packet_md
    assert "SCHEDULED WORK ORDERS" in packet_md

