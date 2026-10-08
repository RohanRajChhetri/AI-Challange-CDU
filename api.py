"""EquiTriage FastAPI Microservice.
Exposes RESTful endpoints for enterprise housing ERP systems (SAP, Salesforce, GovCMS)
to run deterministic triage, query multi-depot routing, inspect wet season road status,
and generate contractor dispatch manifests.
"""
from __future__ import annotations

from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from equitriage_engine import (
    Policy,
    score_tickets,
    compute_manifest_breakdown,
    price_of_equity,
    record_job_resolution,
    record_coordinator_feedback,
    generate_contractor_work_packet,
)
from geo_utils import NT_DEPOTS, WET_SEASON_ROAD_STATUS, get_road_status

app = FastAPI(
    title="EquiTriage REST API",
    description="Autonomous decision-support and dispatch logistics engine for NT public housing maintenance.",
    version="2.0.0",
)


# --------------------------------------------------------------------------
# PYDANTIC SCHEMAS
# --------------------------------------------------------------------------
class TicketInput(BaseModel):
    ticket_id: str
    location: str
    distance_km: float = 0.0
    days_waiting: float = 0.0
    text: str
    property_id: Optional[str] = None
    category: str = "other"
    safety_prob: float = 0.0
    det_score: float = 1.0
    chronic_prob: float = 0.0
    history_text: Optional[str] = None


class PolicyInput(BaseModel):
    equity_weight: float = Field(0.5, ge=0.0, le=1.0)
    safety_floor: float = Field(0.70, ge=0.0, le=1.0)
    max_wait_days: int = Field(45, ge=1)
    depot_mode: str = "closest"
    is_wet_season: bool = False
    enforce_trade_matching: bool = True
    max_labor_hours_per_run: float = 12.0
    jobs_per_week: int = Field(5, ge=1)


class TriageRequest(BaseModel):
    tickets: List[TicketInput]
    policy: Optional[PolicyInput] = None


class ResolutionRequest(BaseModel):
    ticket_id: str
    property_id: str
    category: str
    summary: str
    resolution: str
    technician: str
    parts_used: str = ""


class FeedbackRequest(BaseModel):
    ticket_id: str
    original_category: str
    corrected_category: str
    original_safety_prob: float
    corrected_safety_prob: float
    reason: str
    coordinator: str


# --------------------------------------------------------------------------
# ENDPOINTS
# --------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "EquiTriage Decision & Logistics Engine",
        "depots_count": len(NT_DEPOTS),
        "version": "2.0.0",
    }


@app.get("/api/depots")
def list_depots():
    """List all Northern Territory fleet depots and coordinates."""
    return {"depots": NT_DEPOTS}


@app.get("/api/road-conditions")
def road_conditions(is_wet_season: bool = False):
    """Retrieve seasonal access conditions across NT communities."""
    summary = {}
    for comm in WET_SEASON_ROAD_STATUS:
        summary[comm] = get_road_status(comm, is_wet_season)
    return {
        "season": "wet" if is_wet_season else "dry",
        "road_conditions": summary,
    }


@app.post("/api/triage")
def run_triage(req: TriageRequest):
    """Run deterministic EquiTriage scoring and trip bundling over incoming tickets."""
    if not req.tickets:
        raise HTTPException(status_code=400, detail="Ticket list cannot be empty.")

    p_in = req.policy or PolicyInput()
    policy = Policy(
        equity_weight=p_in.equity_weight,
        safety_floor=p_in.safety_floor,
        max_wait_days=p_in.max_wait_days,
        depot_mode=p_in.depot_mode,
        is_wet_season=p_in.is_wet_season,
        enforce_trade_matching=p_in.enforce_trade_matching,
        max_labor_hours_per_run=p_in.max_labor_hours_per_run,
    )

    tickets_df = pd.DataFrame([t.model_dump() for t in req.tickets])
    scored = score_tickets(tickets_df, policy=policy, jobs_per_week=p_in.jobs_per_week)

    return {
        "count": len(scored),
        "depot_mode": policy.depot_mode,
        "is_wet_season": policy.is_wet_season,
        "results": scored.to_dict(orient="records"),
    }


@app.post("/api/manifest")
def generate_manifest(req: TriageRequest):
    """Generate vehicle dispatch manifests, bundled piggyback jobs, and travel cost recovery."""
    if not req.tickets:
        raise HTTPException(status_code=400, detail="Ticket list cannot be empty.")

    p_in = req.policy or PolicyInput()
    policy = Policy(
        equity_weight=p_in.equity_weight,
        safety_floor=p_in.safety_floor,
        max_wait_days=p_in.max_wait_days,
        depot_mode=p_in.depot_mode,
        is_wet_season=p_in.is_wet_season,
        enforce_trade_matching=p_in.enforce_trade_matching,
        max_labor_hours_per_run=p_in.max_labor_hours_per_run,
    )

    tickets_df = pd.DataFrame([t.model_dump() for t in req.tickets])
    scored = score_tickets(tickets_df, policy=policy, jobs_per_week=p_in.jobs_per_week)
    manifest = compute_manifest_breakdown(scored, jobs_per_week=p_in.jobs_per_week, policy=policy)
    poe = price_of_equity(scored, jobs_per_week=p_in.jobs_per_week, policy=policy)

    return {
        "manifest_runs": manifest.to_dict(orient="records"),
        "price_of_equity": poe.to_dict(),
    }


@app.post("/api/resolve")
def resolve_ticket(req: ResolutionRequest):
    """Mark a work order completed and update the RAG property maintenance ledger."""
    entry = record_job_resolution(
        ticket_id=req.ticket_id,
        property_id=req.property_id,
        category=req.category,
        summary=req.summary,
        resolution=req.resolution,
        technician=req.technician,
        parts_used=req.parts_used,
    )
    return {"status": "success", "message": f"Recorded resolution for {req.ticket_id}", "entry": entry}


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """Submit coordinator adjudication feedback for active learning."""
    entry = record_coordinator_feedback(
        ticket_id=req.ticket_id,
        original_category=req.original_category,
        corrected_category=req.corrected_category,
        original_safety=req.original_safety_prob,
        corrected_safety=req.corrected_safety_prob,
        reason=req.reason,
        coordinator=req.coordinator,
    )
    return {"status": "success", "message": f"Saved feedback for {req.ticket_id}", "entry": entry}
