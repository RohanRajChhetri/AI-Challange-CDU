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
   to emergency runs, mathematically recovering travel efficiency.
5. Context-aware RAG retrieves property maintenance history to detect chronic asset failures.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
HISTORICAL_MAINTENANCE_FILE = (
    DATA_DIR / "historical_maintenance.json"
    if (DATA_DIR / "historical_maintenance.json").exists()
    else Path(__file__).with_name("historical_maintenance.json")
)

# --------------------------------------------------------------------------
# 1. AI EXTRACTION WITH CONTEXT-AWARE RAG (Typed Laya Questions)
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
        # Fallback to ticket_id if property_id missing
        prop_id = ticket.get("ticket_id", "")
    
    # Check direct match or community prefix
    records = history_ledger.get(prop_id, [])
    if not records and "location" in ticket:
        # Check by location identifier if present in keys
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
        
        # Augment the prompt state with real historical records
        augmented_state = f"{hist_text}\n\nCurrent Maintenance Request:\n{t['text']}"
        requests.append({"state": augmented_state, "questions": LAYA_QUESTIONS})

    # 2. Run all tickets through the router in a single batch call
    batch_results = router.predict_batch(requests)

    # 3. Extract answers
    rows = []
    for t, response, (hist_text, hist_list) in zip(records, batch_results, history_meta):
        ans = response["answers"]
        
        # Chronic failure probability from Laya
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
# 2. POLICY: every judgement call is a named, visible, adjustable number
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


# --------------------------------------------------------------------------
# 3. SCORING & INTELLIGENT BUNDLING
# --------------------------------------------------------------------------
def score_tickets(
    signals: pd.DataFrame,
    policy: Policy = Policy(),
    jobs_per_week: int = 5,
) -> pd.DataFrame:
    d = signals.copy()
    e = policy.equity_weight

    # Ensure chronic_prob exists
    if "chronic_prob" not in d.columns:
        d["chronic_prob"] = 0.0
    if "history_text" not in d.columns:
        d["history_text"] = "No prior history records on file."

    # Context-aware chronic failure escalation:
    # If Laya detects recurring failure above chronic_floor, escalate deterioration
    d["is_chronic"] = d.chronic_prob >= policy.chronic_floor
    d["chronic_mult"] = np.where(
        d.is_chronic,
        1.0 + ((d.chronic_prob - policy.chronic_floor) / max(0.01, (1.0 - policy.chronic_floor))) * policy.chronic_weight,
        1.0
    )
    d["effective_det_score"] = (d.det_score * d.chronic_mult).clip(upper=3.0)

    d["base_urgency"] = (
        d.safety_prob * policy.safety_weight
        + d.effective_det_score * policy.deterioration_weight
    )
    d["travel_penalty"] = 1 + policy.travel_cost_per_100km * d.distance_km / 100
    d["wait_mult"] = 1 + (d.days_waiting * policy.wait_rate_per_day).clip(upper=policy.wait_cap)
    d["remote_mult"] = 1 + policy.remote_uplift_per_100km * d.distance_km / 100 * e
    d["travel_discount"] = d.travel_penalty ** (1 - e)  # efficiency pressure left over

    # Baseline: what "cheapest trip first" does by default
    d["efficiency_score"] = d.base_urgency / d.travel_penalty
    # Policy score
    d["final_score"] = d.base_urgency * d.wait_mult * d.remote_mult / d.travel_discount

    # Tiers: hard rules that no multiplier is allowed to overturn
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
    d["rank_shift"] = d.rank_efficiency - d["rank"]  # + = moved UP because of equity rules
    
    # Overdue but low-urgency jobs don't jump the queue; they ride along on the next trip
    d["bundle_with_trip"] = (d.days_waiting > policy.max_wait_days) & (d.tier == 2)

    # ACTIVE INTELLIGENT TRIP BUNDLING ALGORITHM
    # When an urgent job is scheduled for a location, append pending jobs at that same location
    d["is_bundled"] = False
    d["bundle_lead"] = ""
    d["bundle_status"] = "Pending in Queue"
    d["avoided_travel_km"] = 0.0

    # Top primary jobs that dispatch this week
    primary_indices = d.head(jobs_per_week).index
    visited_locations = {}
    for idx in primary_indices:
        loc = d.at[idx, "location"]
        if loc not in visited_locations:
            visited_locations[loc] = d.at[idx, "ticket_id"]
        d.at[idx, "bundle_status"] = "PRIMARY DISPATCH RUN"

    # Scan queue for non-primary tickets at visited locations
    for idx in d.index:
        if idx in primary_indices:
            continue
        loc = d.at[idx, "location"]
        if loc in visited_locations:
            lead_tkt = visited_locations[loc]
            d.at[idx, "is_bundled"] = True
            d.at[idx, "bundle_lead"] = lead_tkt
            d.at[idx, "bundle_status"] = f"BUNDLED ON RUN ({lead_tkt})"
            # Marginal round-trip travel is 0 km, so avoided future travel = 2 * distance_km
            d.at[idx, "avoided_travel_km"] = float(2 * d.at[idx, "distance_km"])

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
    return "; ".join(reasons)


# --------------------------------------------------------------------------
# 4. EXPLANATIONS & TENANT ANSWERS
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

    out.append(
        f"Urgency {r.base_urgency:.1f} (safety {r.safety_prob:.0%}, damage-risk level {r.effective_det_score:.1f}/3, {r.category})."
    )

    if getattr(r, "is_chronic", False):
        out.append(
            f"CHRONIC ISSUE ALERT: Repeat failure pattern identified across prior address history "
            f"(Laya confidence: {r.chronic_prob:.0%}). Deterioration urgency boosted by +{(r.chronic_mult - 1):.0%}."
        )

    out.append(f"Waiting {int(r.days_waiting)} days adds +{(r.wait_mult - 1):.0%}.")

    if r.distance_km > 0:
        out.append(
            f"At {r.distance_km:.0f} km, travel would cut its score by "
            f"{1 - 1 / r.travel_penalty:.0%} under efficiency-only; the equity setting "
            f"({policy.equity_weight:.0%}) offsets that and adds +{(r.remote_mult - 1):.0%} for remoteness."
        )

    if getattr(r, "is_bundled", False):
        out.append(
            f"INTELLIGENT TRIP BUNDLE: Bundled onto scheduled run {r.bundle_lead} to {r.location}. "
            f"Cleared with 0 km marginal travel (saving {r.avoided_travel_km:.0f} km future travel)."
        )
    elif r.bundle_with_trip:
        out.append("Overdue but low urgency: schedule it on the next trip to this community rather than a special trip.")

    if r.needs_review:
        out.append(f"Flagged for human review: {r.review_reason}.")
    return " ".join(out)


def tenant_answer(r, scored: pd.DataFrame, policy: Policy) -> str:
    """Plain-language reply a tenant (or their housing officer) can be given."""
    ahead = scored[scored["rank"] < r["rank"]]
    ahead_safety = int((ahead.tier == 0).sum())
    lines = [
        f"Your request {r.ticket_id} ({r.category}) is currently number {int(r['rank'])} of {len(scored)} on the repair list.",
        f"{len(ahead)} jobs are ahead of it; {ahead_safety} of those are urgent safety jobs that must go first.",
    ]
    
    if getattr(r, "is_bundled", False):
        lines.append(
            f"UPDATE: A crew has been dispatched to {r.location} for primary job {r.bundle_lead}, "
            f"and your repair has been bundled directly onto their manifest for resolution on the same visit!"
        )

    if getattr(r, "is_chronic", False):
        lines.append(
            "Our system flagged repeat maintenance history at your address, escalating your request priority for preventative resolution."
        )

    if r.rank_shift > 0:
        lines.append(
            f"Because you live {r.distance_km:.0f} km from the depot, a cost-only list would have put you at "
            f"#{int(r.rank_efficiency)}. We deliberately moved you up {int(r.rank_shift)} places."
        )
    elif r.rank_shift < 0:
        lines.append(
            f"Other jobs moved ahead of yours because they are more urgent or have waited longer; "
            f"distance did not push you back (a cost-only list would have been #{int(r.rank_efficiency)})."
        )
    lines.append(f"You have waited {int(r.days_waiting)} days, and each extra day raises your priority.")
    lines.append(
        "What can change this: if the problem is getting worse, or there is any risk of fire, injury, "
        "no water, no power, or no working toilet, tell us straight away and it will be re-assessed by a person."
    )
    if r.needs_review:
        lines.append("A coordinator is personally checking this decision.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 5. THE PRICE OF EQUITY & DYNAMIC TRIP BUNDLING RECOVERY
# --------------------------------------------------------------------------
def price_of_equity(scored: pd.DataFrame, jobs_per_week: int, policy: Policy) -> pd.DataFrame:
    """Compare what each ranking does, including dynamic trip bundling savings."""
    cols = {}
    
    for label, rank_col in (("Efficiency-first", "rank_efficiency"), ("Equity policy", "rank")):
        done_primary = scored.nsmallest(jobs_per_week, rank_col)
        left = scored.drop(done_primary.index)
        
        # Visited locations by primary jobs
        trips = done_primary.groupby("location")["distance_km"].first()
        primary_travel_km = int(2 * trips.sum())
        
        # Bundled jobs: find tickets remaining in queue at these exact visited locations
        bundled_jobs = left[left["location"].isin(trips.index)]
        bundled_count = len(bundled_jobs)
        
        # Saved travel: if dispatched separately later, each would need a 2 * distance_km round trip
        saved_travel_km = int(2 * bundled_jobs["distance_km"].sum())
        cost_saved_aud = int(saved_travel_km * policy.fleet_cost_per_km)
        
        total_jobs_resolved = len(done_primary) + bundled_count
        net_travel_km = max(0, primary_travel_km - saved_travel_km)
        effective_km_per_job = round(primary_travel_km / max(1, total_jobs_resolved), 1)

        cols[label] = {
            "Primary jobs dispatched": len(done_primary),
            "Bundled secondary jobs resolved": bundled_count,
            "Total repairs completed": total_jobs_resolved,
            "Safety-critical jobs completed": int((done_primary.safety_prob >= policy.safety_floor).sum()),
            "Remote jobs completed": int((done_primary.distance_km >= policy.remote_km).sum()),
            "Communities visited": int(trips.size),
            "Direct fleet travel (km)": primary_travel_km,
            "Travel avoided via bundling (km)": saved_travel_km,
            "Fleet cost saved via bundling ($)": f"${cost_saved_aud:,}",
            "Effective travel per job (km/job)": effective_km_per_job,
            "Longest wait still in queue (days)": int(left.days_waiting.max()) if len(left) else 0,
            "Remote jobs still waiting": int((left.distance_km >= policy.remote_km).sum()),
        }
    return pd.DataFrame(cols)


def compute_manifest_breakdown(scored: pd.DataFrame, jobs_per_week: int, policy: Policy) -> pd.DataFrame:
    """Generate detailed vehicle manifest showing primary dispatches and piggybacked jobs."""
    primary_done = scored.nsmallest(jobs_per_week, "rank")
    manifest_rows = []
    
    run_num = 1
    for loc, group in primary_done.groupby("location"):
        lead_job = group.iloc[0]
        # Find bundled secondary jobs at this location
        other_in_primary = group.iloc[1:]["ticket_id"].tolist()
        queue_remaining = scored.drop(primary_done.index)
        bundled_at_loc = queue_remaining[queue_remaining["location"] == loc]
        
        piggybacked_ids = other_in_primary + bundled_at_loc["ticket_id"].tolist()
        total_resolved = 1 + len(piggybacked_ids)
        distance = lead_job["distance_km"]
        avoided_km = int(len(piggybacked_ids) * 2 * distance)
        saved_dollars = int(avoided_km * policy.fleet_cost_per_km)
        
        manifest_rows.append({
            "Run #": f"RUN-{run_num:02d}",
            "Destination": loc,
            "Round-Trip Distance": f"{int(2 * distance)} km",
            "Lead Safety/Equity Job": f"{lead_job['ticket_id']} ({lead_job['category'].title()})",
            "Bundled Manifest Repairs": ", ".join(piggybacked_ids) if piggybacked_ids else "None (Single Run)",
            "Total Jobs Resolved": total_resolved,
            "Future Travel Saved": f"{avoided_km} km",
            "Cost Recovered": f"${saved_dollars:,}",
        })
        run_num += 1

    return pd.DataFrame(manifest_rows)


# --------------------------------------------------------------------------
# 6. AUDIT LOG: "let the human own it" means the decision is recorded
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
        "decision": decision,  # e.g. "ENFORCE_EQUITY" / "ACCEPT_EFFICIENCY"
        "justification": justification.strip(),
        "coordinator": coordinator,
        "system_explanation": ticket_row.explanation,
        "policy": asdict(policy),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
