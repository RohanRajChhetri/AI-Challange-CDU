"""EquiTriage engine.

Design rules
------------
1. The AI (Laya) only EXTRACTS signals from text + historical context (slow, run once, cacheable).
2. Everything after that is plain, deterministic maths (instant, auditable),
   so sliders in the UI can re-rank live and every score can be explained.
3. Two rankings are always computed side by side:
     - efficiency-only (what "cheapest trip first" quietly does today)
     - the policy ranking (safety floor + ageing guardrail + equity slider)
   The gap between them IS the equity trade-off the human must own.
4. Intelligent Trip Bundling dynamically attaches overdue/lower-tier jobs in the same community
   to emergency runs, constrained by onboard contractor trade skills and working hour budgets.
5. Context-aware RAG retrieves property maintenance history to detect chronic asset failures.
6. Multi-depot regional routing and seasonal wet-weather passability awareness.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from geo_utils import (
    DARWIN_DEPOT,
    NT_DEPOTS,
    find_closest_depot,
    get_coords,
    get_depot_info,
    get_road_status,
    haversine_km,
)

DATA_DIR = Path(__file__).parent / "data"
HISTORICAL_MAINTENANCE_FILE = (
    DATA_DIR / "historical_maintenance.json"
    if (DATA_DIR / "historical_maintenance.json").exists()
    else Path(__file__).with_name("historical_maintenance.json")
)
FEEDBACK_LOG_FILE = (
    DATA_DIR / "evaluation_feedback.jsonl"
    if (DATA_DIR / "evaluation_feedback.jsonl").exists()
    else Path(__file__).with_name("evaluation_feedback.jsonl")
)

# --------------------------------------------------------------------------
# 1. TRADE MAPPINGS & LABOR DURATION MODEL
# --------------------------------------------------------------------------
CATEGORY_TRADE_MAP = {
    "plumbing": "Plumber",
    "electrical": "Electrician",
    "structural": "Carpenter / Builder",
    "pest": "Pest Control Specialist",
    "other": "General Maintenance Technician",
}

CATEGORY_BASE_HOURS = {
    "plumbing": 2.5,
    "electrical": 2.0,
    "structural": 3.5,
    "pest": 1.5,
    "other": 2.0,
}

TRADE_COMPATIBILITY = {
    "Plumber": ["plumbing", "other"],
    "Electrician": ["electrical", "other"],
    "Carpenter / Builder": ["structural", "other"],
    "Pest Control Specialist": ["pest", "other"],
    "General Maintenance Technician": ["other", "pest"],
}

# --------------------------------------------------------------------------
# 2. AI EXTRACTION WITH CONTEXT-AWARE RAG (Typed Laya Questions)
# --------------------------------------------------------------------------
LAYA_QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "Categorize the primary issue in this maintenance request.",
        "criteria": {
            "plumbing": "Water, leaks, pipes, or drainage issues",
            "electrical": "Power, wiring, outlets, or appliance issues",
            "structural": "Roof, walls, windows, doors, or physical damage",
            "pest": "Insects, rodents, or animal issues",
            "other": "Everything else",
        },
    },
    "safety_hazard": {
        "type": "noul",
        "instructions": "Does this issue present an immediate physical safety, fire, or severe health hazard to the tenants?",
    },
    "deterioration_risk": {
        "type": "score",
        "instructions": "Rate the risk of this issue causing significant property damage if left unattended.",
        "criteria": [
            "Stable, will not get worse immediately",
            "Minor deterioration over time",
            "Will cause significant damage if left for a month",
            "Imminent failure or catastrophic damage",
        ],
    },
    "chronic_failure": {
        "type": "noul",
        "instructions": "Based on the maintenance history and current request, does this issue represent a chronic, repeat, or recurrent failure at this property?",
    },
}


def load_history_ledger() -> dict[str, list[dict]]:
    """Load authentic historical maintenance ledger for NT properties."""
    if HISTORICAL_MAINTENANCE_FILE.exists():
        try:
            with open(HISTORICAL_MAINTENANCE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def retrieve_property_history(ticket: dict, history_ledger: dict[str, list[dict]], max_records: int = 3) -> tuple[str, list[dict]]:
    """Retrieve historical maintenance context for a property (RAG context-retrieval)."""
    prop_id = ticket.get("property_id")
    if not prop_id:
        prop_id = ticket.get("ticket_id", "")

    records = history_ledger.get(prop_id, [])
    if not records and "location" in ticket:
        loc_clean = ticket["location"].split("(")[0].strip().upper()
        for k, v in history_ledger.items():
            if loc_clean in k.upper():
                records = v
                break

    records = records[:max_records]
    if not records:
        return "No prior maintenance work orders recorded on file for this address.", []

    formatted_lines = []
    for r in records:
        days = r.get("days_ago", "?")
        cat = r.get("category", "general")
        summary = r.get("summary", "")
        resolution = r.get("resolution", "")
        formatted_lines.append(f"- [{days} days ago | {cat.upper()}]: {summary} (Resolution: {resolution})")

    context_str = "Prior Maintenance History at this Address:\n" + "\n".join(formatted_lines)
    return context_str, records


def extract_signals(tickets: pd.DataFrame, router) -> pd.DataFrame:
    """Fast step using Laya's native batched inference with RAG context-awareness."""
    records = tickets.to_dict("records")
    history_ledger = load_history_ledger()

    # 1. Prepare RAG-augmented batch requests
    requests = []
    history_meta = []
    for t in records:
        hist_text, hist_list = retrieve_property_history(t, history_ledger)
        history_meta.append((hist_text, hist_list))

        augmented_state = f"{hist_text}\n\nCurrent Maintenance Request:\n{t['text']}"
        requests.append({"state": augmented_state, "questions": LAYA_QUESTIONS})

    # 2. Run all tickets through the router in a single batch call
    batch_results = router.predict_batch(requests)

    # 3. Extract answers
    rows = []
    for t, response, (hist_text, hist_list) in zip(records, batch_results, history_meta):
        ans = response["answers"]

        chronic_prob = 0.0
        if "chronic_failure" in ans:
            chronic_prob = float(ans["chronic_failure"].get("noul", 0.0))

        rows.append(
            {
                **t,
                "category": ans["category"]["choice"],
                "safety_prob": float(ans["safety_hazard"]["noul"]),
                "det_score": float(ans["deterioration_risk"]["score"]),
                "chronic_prob": chronic_prob,
                "history_text": hist_text,
                "has_history": len(hist_list) > 0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. POLICY DEFINITION (Named, visible, adjustable governance parameters)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    safety_weight: float = 6.0             # max points from safety probability
    deterioration_weight: float = 4 / 3    # points per det_score step (max 4)
    equity_weight: float = 0.5             # 0 = efficiency-only, 1 = distance-blind
    travel_cost_per_100km: float = 0.25    # how hard distance penalises a job (efficiency view)
    remote_uplift_per_100km: float = 0.10  # extra boost for remoteness, scaled by equity_weight
    wait_rate_per_day: float = 0.02        # fairness to EVERYONE who waits, any location
    wait_cap: float = 1.0                  # ageing can add at most +100%
    safety_floor: float = 0.70             # >= this probability => always Tier 0
    max_wait_days: int = 45                # guardrail: no job waits longer than this
    guardrail_min_urgency: float = 2.0     # ...but only jobs this urgent can jump the queue
    review_band: tuple = (0.40, 0.60)      # model "unsure" zone => human check
    remote_km: float = 100.0               # what counts as "remote" in summaries
    # Context-aware chronic failure & bundling knobs:
    chronic_weight: float = 0.50           # multiplier boost for recurring failure patterns
    chronic_floor: float = 0.60            # Laya probability threshold for chronic alert
    max_bundle_per_trip: int = 3           # max piggyback jobs appended to a remote run
    fleet_cost_per_km: float = 1.50        # NT fleet operational cost ($/km)
    # Logistics & Operational Knobs:
    depot_mode: str = "closest"            # "closest" or specific depot id e.g. "darwin", "katherine"
    is_wet_season: bool = False            # Wet season monsoon / road flood simulation
    wet_season_access_penalty: float = 0.35# Cost/delay multiplier for impassable/4WD routes
    enforce_trade_matching: bool = True    # Restrict bundled jobs to onboard technician trade
    max_labor_hours_per_run: float = 12.0  # Daily on-site labor ceiling per vehicle crew


# --------------------------------------------------------------------------
# 4. SCORING & SMART TRADE-CONSTRAINED BUNDLING
# --------------------------------------------------------------------------
def score_tickets(
    signals: pd.DataFrame,
    policy: Policy = Policy(),
    jobs_per_week: int = 5,
) -> pd.DataFrame:
    d = signals.copy()
    e = policy.equity_weight

    # Ensure auxiliary fields exist
    if "chronic_prob" not in d.columns:
        d["chronic_prob"] = 0.0
    if "history_text" not in d.columns:
        d["history_text"] = "No prior history records on file."

    # Map required trades and estimated labor hours
    d["trade_required"] = d["category"].map(CATEGORY_TRADE_MAP).fillna("General Maintenance Technician")
    d["base_labor_hours"] = d["category"].map(CATEGORY_BASE_HOURS).fillna(2.0)
    d["est_labor_hours"] = (d["base_labor_hours"] + d["det_score"] * 0.5).round(1)

    # Multi-depot routing and distance computation
    coords = [get_coords(loc) for loc in d["location"]]
    assigned_depots = []
    effective_distances = []

    for (lat, lon), orig_dist in zip(coords, d["distance_km"]):
        if policy.depot_mode == "closest":
            depot = find_closest_depot(lat, lon)
            # Outback driving curvature factor is ~1.28x over great-circle distance
            calc_dist = max(10.0, haversine_km(lat, lon, depot["lat"], depot["lon"]) * 1.28)
            effective_distances.append(round(calc_dist, 1))
        elif policy.depot_mode in NT_DEPOTS:
            depot = NT_DEPOTS[policy.depot_mode]
            calc_dist = max(10.0, haversine_km(lat, lon, depot["lat"], depot["lon"]) * 1.28)
            effective_distances.append(round(calc_dist, 1))
        else:
            depot = DARWIN_DEPOT
            effective_distances.append(float(orig_dist))
        assigned_depots.append(depot)

    d["depot_id"] = [dep["id"] for dep in assigned_depots]
    d["depot_name"] = [dep["name"] for dep in assigned_depots]
    d["effective_distance_km"] = effective_distances

    # Road conditions & wet season accessibility
    road_statuses = [get_road_status(loc, policy.is_wet_season) for loc in d["location"]]
    d["road_status"] = [r["status"] for r in road_statuses]
    d["road_status_label"] = [r["status_label"] for r in road_statuses]
    d["transit_mode"] = [r["transit_mode"] for r in road_statuses]
    d["access_factor"] = [r["access_factor"] for r in road_statuses]

    # Chronic failure escalation
    d["is_chronic"] = d.chronic_prob >= policy.chronic_floor
    d["chronic_mult"] = np.where(
        d.is_chronic,
        1.0 + ((d.chronic_prob - policy.chronic_floor) / max(0.01, (1.0 - policy.chronic_floor))) * policy.chronic_weight,
        1.0,
    )
    d["effective_det_score"] = (d.det_score * d.chronic_mult).clip(upper=3.0)

    # Base urgency
    d["base_urgency"] = (
        d.safety_prob * policy.safety_weight
        + d.effective_det_score * policy.deterioration_weight
    )

    # Effective distance under wet season road multiplier
    d["adj_distance_km"] = d["effective_distance_km"] * np.where(
        policy.is_wet_season & (d["road_status"] != "OPEN"),
        1.0 + policy.wet_season_access_penalty,
        1.0,
    )

    d["travel_penalty"] = 1 + policy.travel_cost_per_100km * d.adj_distance_km / 100
    d["wait_mult"] = 1 + (d.days_waiting * policy.wait_rate_per_day).clip(upper=policy.wait_cap)
    d["remote_mult"] = 1 + policy.remote_uplift_per_100km * d.adj_distance_km / 100 * e
    d["travel_discount"] = d.travel_penalty ** (1 - e)

    # Baseline efficiency score vs Policy score
    d["efficiency_score"] = d.base_urgency / d.travel_penalty
    d["final_score"] = d.base_urgency * d.wait_mult * d.remote_mult / d.travel_discount

    # Tiers: hard rules
    d["tier"] = np.select(
        [
            d.safety_prob >= policy.safety_floor,
            (d.days_waiting > policy.max_wait_days) & (d.base_urgency >= policy.guardrail_min_urgency),
        ],
        [0, 1],
        default=2,
    )

    d = d.sort_values(["tier", "final_score"], ascending=[True, False]).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    d["rank_efficiency"] = d.efficiency_score.rank(ascending=False, method="first").astype(int)
    d["rank_shift"] = d.rank_efficiency - d["rank"]

    d["bundle_with_trip"] = (d.days_waiting > policy.max_wait_days) & (d.tier == 2)

    # ----------------------------------------------------------------------
    # ACTIVE INTELLIGENT TRIP BUNDLING WITH TRADE & CAPACITY CONSTRAINTS
    # ----------------------------------------------------------------------
    d["is_bundled"] = False
    d["bundle_lead"] = ""
    d["bundle_status"] = "Pending in Queue"
    d["avoided_travel_km"] = 0.0

    primary_indices = d.head(jobs_per_week).index
    visited_manifests: dict[str, dict] = {}

    for idx in primary_indices:
        loc = d.at[idx, "location"]
        tkt = d.at[idx, "ticket_id"]
        trade = d.at[idx, "trade_required"]
        labor_hrs = float(d.at[idx, "est_labor_hours"])

        if loc not in visited_manifests:
            visited_manifests[loc] = {
                "lead_ticket": tkt,
                "lead_trade": trade,
                "allocated_hours": labor_hrs,
                "bundled_tickets": [],
            }
        else:
            visited_manifests[loc]["allocated_hours"] += labor_hrs

        d.at[idx, "bundle_status"] = f"PRIMARY DISPATCH ({trade})"

    # Scan queue for candidate secondary jobs at visited locations
    for idx in d.index:
        if idx in primary_indices:
            continue
        loc = d.at[idx, "location"]
        if loc in visited_manifests:
            manifest = visited_manifests[loc]
            lead_tkt = manifest["lead_ticket"]
            lead_trade = manifest["lead_trade"]
            job_cat = d.at[idx, "category"]
            job_trade = d.at[idx, "trade_required"]
            job_hrs = float(d.at[idx, "est_labor_hours"])

            # 1. Trade compatibility check
            compatible_cats = TRADE_COMPATIBILITY.get(lead_trade, ["other"])
            is_compatible = (not policy.enforce_trade_matching) or (job_cat in compatible_cats)

            if not is_compatible:
                d.at[idx, "bundle_status"] = f"Queued (Requires {job_trade}, Onsite: {lead_trade})"
                continue

            # 2. Labor capacity check
            if (manifest["allocated_hours"] + job_hrs) > policy.max_labor_hours_per_run:
                d.at[idx, "bundle_status"] = f"Queued (Exceeds {policy.max_labor_hours_per_run:.0f}h Crew Shift)"
                continue

            # 3. Maximum bundle count per trip
            if len(manifest["bundled_tickets"]) >= policy.max_bundle_per_trip:
                d.at[idx, "bundle_status"] = f"Queued (Max {policy.max_bundle_per_trip} Piggyback Cap)"
                continue

            # Attach to manifest
            d.at[idx, "is_bundled"] = True
            d.at[idx, "bundle_lead"] = lead_tkt
            d.at[idx, "bundle_status"] = f"BUNDLED ON RUN ({lead_tkt})"
            d.at[idx, "avoided_travel_km"] = float(2 * d.at[idx, "effective_distance_km"])

            manifest["allocated_hours"] += job_hrs
            manifest["bundled_tickets"].append(d.at[idx, "ticket_id"])

    n = len(d)
    d["review_reason"] = d.apply(lambda r: _review_reason(r, policy, n), axis=1)
    d["needs_review"] = d.review_reason != ""
    d["explanation"] = d.apply(lambda r: explain(r, n, policy), axis=1)
    return d


def _review_reason(r, policy: Policy, n: int) -> str:
    reasons = []
    lo, hi = policy.review_band
    if lo < r.safety_prob < hi:
        reasons.append("safety model is unsure")
    if abs(r.rank_shift) >= max(3, n // 5):
        reasons.append(f"equity rules moved it {abs(int(r.rank_shift))} places")
    if r.category == "other":
        reasons.append("fault category unclear")
    if getattr(r, "is_chronic", False):
        reasons.append(f"chronic failure flagged ({r.chronic_prob:.0%})")
    if getattr(r, "road_status", "OPEN") == "IMPASSABLE":
        reasons.append("wet season flooded road access")
    return "; ".join(reasons)


# --------------------------------------------------------------------------
# 5. EXPLANATIONS & TENANT ANSWERS
# --------------------------------------------------------------------------
def explain(r, n: int, policy: Policy) -> str:
    out = [f"Ranked #{int(r['rank'])} of {n}. Efficiency-only would have placed it #{int(r.rank_efficiency)}."]

    if r.tier == 0:
        out.append(
            f"SAFETY OVERRIDE: safety risk {r.safety_prob:.0%} is above the {policy.safety_floor:.0%} floor, "
            "so travel cost is not allowed to delay it."
        )
    elif r.tier == 1:
        out.append(f"AGEING GUARDRAIL: waiting {int(r.days_waiting)} days exceeds the {policy.max_wait_days}-day limit.")

    dist_km = r.get("effective_distance_km", r.distance_km)
    depot_name = r.get("depot_name", "Darwin Central Depot")
    out.append(
        f"Urgency {r.base_urgency:.1f} (safety {r.safety_prob:.0%}, damage level {r.effective_det_score:.1f}/3, {r.category}, "
        f"assigned trade: {r.trade_required}, ~{r.est_labor_hours:.1f}h work)."
    )

    if getattr(r, "is_chronic", False):
        out.append(
            f"CHRONIC ISSUE ALERT: Repeat failure pattern identified across prior address history "
            f"(Laya confidence: {r.chronic_prob:.0%}). Deterioration urgency boosted by +{(r.chronic_mult - 1):.0%}."
        )

    out.append(f"Waiting {int(r.days_waiting)} days adds +{(r.wait_mult - 1):.0%}.")

    if dist_km > 0:
        out.append(
            f"At {dist_km:.0f} km from {depot_name}, travel would cut its score by "
            f"{1 - 1 / r.travel_penalty:.0%} under efficiency-only; the equity setting "
            f"({policy.equity_weight:.0%}) offsets that and adds +{(r.remote_mult - 1):.0%} for remoteness."
        )

    if r.get("road_status") == "IMPASSABLE":
        out.append(f"WET SEASON NOTICE: Route impassable by road ({r.road_reason}); requires {r.transit_mode}.")

    if getattr(r, "is_bundled", False):
        out.append(
            f"SMART TRIP BUNDLE: Bundled onto scheduled run {r.bundle_lead} to {r.location} matching onboard {r.trade_required} skillset. "
            f"Resolved with 0 km marginal fleet travel (saving {r.avoided_travel_km:.0f} km future travel)."
        )
    elif r.bundle_with_trip:
        out.append("Overdue but low urgency: schedule on next community run.")

    if r.needs_review:
        out.append(f"Flagged for human review: {r.review_reason}.")
    return " ".join(out)


def tenant_answer(r, scored: pd.DataFrame, policy: Policy) -> str:
    """Plain-language reply a tenant (or their housing officer) can be given."""
    ahead = scored[scored["rank"] < r["rank"]]
    ahead_safety = int((ahead.tier == 0).sum())
    dist_km = r.get("effective_distance_km", r.distance_km)
    depot_name = r.get("depot_name", "Darwin Central Fleet Depot")

    lines = [
        f"Your request {r.ticket_id} ({r.category}) is currently number {int(r['rank'])} of {len(scored)} on the repair list.",
        f"{len(ahead)} jobs are ahead of it; {ahead_safety} of those are urgent safety jobs that must go first.",
    ]

    if getattr(r, "is_bundled", False):
        lines.append(
            f"DISPATCH UPDATE: A contractor crew has been dispatched to {r.location} for primary job {r.bundle_lead}, "
            f"and your repair ({r.category}) has been bundled directly onto their manifest for resolution on the same visit!"
        )

    if getattr(r, "is_chronic", False):
        lines.append(
            "Our system flagged repeat maintenance history at your address, escalating your request priority for preventative resolution."
        )

    if r.get("road_status") == "IMPASSABLE":
        lines.append(
            f"WEATHER ADVISORY: Access roads to your community are cut by seasonal floodwaters ({r.road_reason}). "
            f"Emergency access is operating via {r.transit_mode}."
        )

    if r.rank_shift > 0:
        lines.append(
            f"Because you live {dist_km:.0f} km from {depot_name}, a cost-only list would have put you at "
            f"#{int(r.rank_efficiency)}. We deliberately moved you up {int(r.rank_shift)} places."
        )
    elif r.rank_shift < 0:
        lines.append(
            f"Other jobs moved ahead because they are more urgent or have waited longer; "
            f"distance did not push you back (a cost-only list would have been #{int(r.rank_efficiency)})."
        )
    lines.append(f"You have waited {int(r.days_waiting)} days, and each extra day raises your priority.")
    lines.append(
        "What can change this: if the problem is getting worse, or there is any risk of fire, injury, "
        "no water, no power, or no working toilet, tell us straight away and it will be re-assessed by a person."
    )
    if r.needs_review:
        lines.append("A housing coordinator is personally checking this decision.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 6. PRICE OF EQUITY & TRIP BUNDLING METRICS
# --------------------------------------------------------------------------
def price_of_equity(scored: pd.DataFrame, jobs_per_week: int, policy: Policy) -> pd.DataFrame:
    """Compare what each ranking does, including dynamic trip bundling savings."""
    cols = {}

    for label, rank_col in (("Efficiency-first", "rank_efficiency"), ("Equity policy", "rank")):
        done_primary = scored.nsmallest(jobs_per_week, rank_col)
        left = scored.drop(done_primary.index)

        # Visited locations by primary jobs
        trips = done_primary.groupby("location")["effective_distance_km"].first()
        primary_travel_km = int(2 * trips.sum())

        # Bundled jobs
        bundled_jobs = left[left["location"].isin(trips.index)]
        if policy.enforce_trade_matching:
            # Only count trade-compatible bundled jobs
            primary_trades = set(done_primary["trade_required"])
            bundled_jobs = bundled_jobs[
                bundled_jobs.apply(
                    lambda b: any(b["category"] in TRADE_COMPATIBILITY.get(t, ["other"]) for t in primary_trades),
                    axis=1,
                )
            ]

        bundled_count = len(bundled_jobs)
        saved_travel_km = int(2 * bundled_jobs["effective_distance_km"].sum())
        cost_saved_aud = int(saved_travel_km * policy.fleet_cost_per_km)

        total_jobs_resolved = len(done_primary) + bundled_count
        effective_km_per_job = round(primary_travel_km / max(1, total_jobs_resolved), 1)

        cols[label] = {
            "Primary jobs dispatched": len(done_primary),
            "Bundled secondary jobs resolved": bundled_count,
            "Total repairs completed": total_jobs_resolved,
            "Safety-critical jobs completed": int((done_primary.safety_prob >= policy.safety_floor).sum()),
            "Remote jobs completed": int((done_primary.effective_distance_km >= policy.remote_km).sum()),
            "Communities visited": int(trips.size),
            "Direct fleet travel (km)": primary_travel_km,
            "Travel avoided via bundling (km)": saved_travel_km,
            "Fleet cost saved via bundling ($)": cost_saved_aud,
            "Effective travel per job (km/job)": effective_km_per_job,
            "Longest wait still in queue (days)": int(left.days_waiting.max()) if len(left) else 0,
            "Remote jobs still waiting": int((left.effective_distance_km >= policy.remote_km).sum()),
        }
    return pd.DataFrame(cols)


def compute_manifest_breakdown(scored: pd.DataFrame, jobs_per_week: int, policy: Policy) -> pd.DataFrame:
    """Generate detailed vehicle manifest showing primary dispatches, trades, and bundled jobs."""
    primary_done = scored.nsmallest(jobs_per_week, "rank")
    manifest_rows = []

    run_num = 1
    for loc, group in primary_done.groupby("location"):
        lead_job = group.iloc[0]
        other_in_primary = group.iloc[1:]["ticket_id"].tolist()
        queue_remaining = scored.drop(primary_done.index)
        bundled_at_loc = queue_remaining[
            (queue_remaining["location"] == loc) & (queue_remaining["is_bundled"])
        ]

        piggybacked_ids = other_in_primary + bundled_at_loc["ticket_id"].tolist()
        total_resolved = 1 + len(piggybacked_ids)
        distance = lead_job.get("effective_distance_km", lead_job["distance_km"])
        avoided_km = int(len(piggybacked_ids) * 2 * distance)
        saved_dollars = int(avoided_km * policy.fleet_cost_per_km)

        # Total on-site labor
        total_labor = round(lead_job["est_labor_hours"] + bundled_at_loc["est_labor_hours"].sum(), 1)

        manifest_rows.append({
            "Run #": f"RUN-{run_num:02d}",
            "Destination": loc,
            "Depot Base": lead_job.get("depot_name", "Darwin Central Depot"),
            "Lead Trade": lead_job["trade_required"],
            "Labor Hours Onsite": f"{total_labor}h (max {policy.max_labor_hours_per_run:.0f}h)",
            "Round-Trip Dist": f"{int(2 * distance)} km",
            "Transit Mode": lead_job.get("transit_mode", "Standard Road Fleet"),
            "Lead Safety/Equity Job": f"{lead_job['ticket_id']} ({lead_job['category'].title()})",
            "Bundled Manifest Repairs": ", ".join(piggybacked_ids) if piggybacked_ids else "None (Single Run)",
            "Total Jobs Resolved": total_resolved,
            "Future Travel Saved": f"{avoided_km} km",
            "Cost Recovered": f"${saved_dollars:,}",
        })
        run_num += 1

    return pd.DataFrame(manifest_rows)


# --------------------------------------------------------------------------
# 7. FIELD WORK ORDER PACKET GENERATOR (Contractor Run Sheet)
# --------------------------------------------------------------------------
def generate_contractor_work_packet(
    manifest_row: dict,
    scored_df: pd.DataFrame,
    policy: Policy,
) -> str:
    """Generate printable contractor work packet markdown for a scheduled remote dispatch run."""
    destination = manifest_row["Destination"]
    run_ref = manifest_row["Run #"]
    lead_trade = manifest_row["Lead Trade"]
    depot_base = manifest_row["Depot Base"]
    transit_mode = manifest_row["Transit Mode"]

    coords = get_coords(destination)
    road_status = get_road_status(destination, policy.is_wet_season)

    # Gather tickets for this run
    lead_tkt_id = manifest_row["Lead Safety/Equity Job"].split(" ")[0].strip()
    bundled_str = manifest_row["Bundled Manifest Repairs"]
    bundled_ids = [b.strip() for b in bundled_str.split(",") if b.strip() and b.strip() != "None (Single Run)"]

    all_tkt_ids = [lead_tkt_id] + bundled_ids
    run_tickets = scored_df[scored_df["ticket_id"].isin(all_tkt_ids)]

    lines = [
        f"# 📋 NORTHERN TERRITORY HOUSING // FIELD CONTRACTOR WORK PACKET",
        f"**Run Identifier:** `{run_ref}` | **Destination Community:** **{destination}** | **Status:** AUTHORIZED DISPATCH",
        f"**Fleet Depot Origin:** {depot_base} | **GPS Coordinates:** `{coords[0]:.4f}° S, {coords[1]:.4f}° E`",
        f"**Transit Mode:** `{transit_mode}` | **Road Condition:** `{road_status['status_label']}`",
        f"**Lead Crew Trade:** `{lead_trade}` | **Total Allocated Hours:** `{manifest_row['Labor Hours Onsite']}`",
        "",
        "---",
        "## 1. ACCESS & SAFETY BRIEFING",
        f"- **Access Route:** {road_status['reason']}",
        f"- **Seasonal Road Status:** {road_status['status']} (Access Factor: {road_status['access_factor']}x)",
        "- **Contractor Mandate:** High-clearance satellite comms (InReach/HF radio) required beyond regional sealed corridors.",
        "- **Community Protocol:** Register arrival with Local Council / Housing Authority Officer before lot entry.",
        "",
        "---",
        "## 2. SCHEDULED WORK ORDERS (MANIFEST)",
    ]

    for idx, row in run_tickets.iterrows():
        is_lead = (row["ticket_id"] == lead_tkt_id)
        role = "⭐ PRIMARY EMERGENCY/EQUITY JOB" if is_lead else "📦 BUNDLED SAME-VISIT JOB"
        lines.extend([
            f"### `{row['ticket_id']}` — {row.get('property_id', row['ticket_id'])} ({role})",
            f"- **Issue Category:** {row['category'].title()} | **Assigned Trade:** {row['trade_required']}",
            f"- **Safety Risk Assessment:** {row['safety_prob']:.0%} | **Estimated On-Site Labor:** {row['est_labor_hours']} hours",
            f"- **Tenant Reported Description:** *\"{row['text']}\"*",
            f"- **Asset Maintenance History Context:**\n  ```\n  {row.get('history_text', 'No prior record on file.')}\n  ```",
            f"- **Technician Action Plan:** Inspect {row['category']} assembly; test tenant isolation valve/mains switch; restore essential habitability.",
            "",
        ])

    lines.extend([
        "---",
        "## 3. RECOMMENDED TOOLS & PARTS MANIFEST",
        f"- **Primary Trade Toolset:** Standard {lead_trade} mobile kit & testing meters.",
        "- **Essential Spare Parts Stock:** PVC connectors, 15/20mm copper pipe joints, safety RCD breakers, heavy-duty sealants, replacement tapware.",
        "- **PPE & Safety Equipment:** Heat protection, steel-cap boots, emergency snake bite kit, water filtration packs.",
        "",
        "---",
        "## 4. FIELD COMPLETION SIGN-OFF",
        "| Ticket ID | Tenant Signature | Contractor Name & ID | Resolution Status | Sign-off Date |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for tkt in all_tkt_ids:
        lines.append(f"| `{tkt}` | ____________________ | ____________________ | [ ] Resolved  [ ] Deferral Needed | ____/____/2026 |")

    lines.append("\n*Report completed job tickets back to Housing Dispatch Console within 24 hours of returning to depot.*")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 8. RESOLUTION & EVALUATION FEEDBACK PERSISTENCE
# --------------------------------------------------------------------------
def record_job_resolution(
    ticket_id: str,
    property_id: str,
    category: str,
    summary: str,
    resolution: str,
    technician: str,
    parts_used: str = "",
    ledger_path: Path | str = HISTORICAL_MAINTENANCE_FILE,
) -> dict:
    """Record a completed work order into the authentic historical maintenance RAG ledger."""
    path = Path(ledger_path)
    ledger = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                ledger = json.load(f)
        except Exception:
            ledger = {}

    entry = {
        "days_ago": 0,
        "category": category.lower(),
        "summary": summary.strip(),
        "resolution": resolution.strip(),
        "technician": technician.strip(),
        "parts_used": parts_used.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if property_id not in ledger:
        ledger[property_id] = []

    # Insert at the top of history
    ledger[property_id].insert(0, entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

    return entry


def record_coordinator_feedback(
    ticket_id: str,
    original_category: str,
    corrected_category: str,
    original_safety: float,
    corrected_safety: float,
    reason: str,
    coordinator: str,
    feedback_path: Path | str = FEEDBACK_LOG_FILE,
) -> dict:
    """Record coordinator adjudication feedback for active learning & prompt refinement."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticket_id": ticket_id,
        "original_category": original_category,
        "corrected_category": corrected_category,
        "original_safety_prob": original_safety,
        "corrected_safety_prob": corrected_safety,
        "reason": reason.strip(),
        "coordinator": coordinator.strip(),
    }
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


# --------------------------------------------------------------------------
# 9. AUDIT LOG (Immutable Record of Human Discretion)
# --------------------------------------------------------------------------
def log_decision(
    path: str | Path,
    ticket_row,
    decision: str,
    justification: str,
    coordinator: str,
    policy: Policy,
) -> None:
    if not justification.strip():
        raise ValueError("Justification is required.")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticket_id": ticket_row.ticket_id,
        "system_rank": int(ticket_row["rank"]),
        "efficiency_rank": int(ticket_row.rank_efficiency),
        "decision": decision,
        "justification": justification.strip(),
        "coordinator": coordinator,
        "system_explanation": ticket_row.explanation,
        "policy": asdict(policy),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
