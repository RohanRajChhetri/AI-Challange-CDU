"""EquiTriage engine.

Design rules
------------
1. The AI (Laya) only EXTRACTS signals from free text (slow, run once, cacheable).
2. Everything after that is plain, deterministic maths (instant, auditable),
   so sliders in the UI can re-rank live and every score can be explained.
3. Two rankings are always computed side by side:
     - efficiency-only (what "cheapest trip first" quietly does today)
     - the policy ranking (safety floor + ageing guardrail + equity slider)
   The gap between them IS the equity trade-off the human must own.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. AI EXTRACTION (typed questions, same as your original)
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
}


def extract_signals(tickets: pd.DataFrame, router) -> pd.DataFrame:
    """Slow step (one model call per ticket). Cache this in Streamlit with
    st.cache_data keyed on the ticket text, NOT on the policy sliders."""
    rows = []
    for t in tickets.to_dict("records"):
        ans = router.predict(state=t["text"], questions=LAYA_QUESTIONS)["answers"]
        rows.append(
            {
                **t,
                "category": ans["category"]["choice"],
                "safety_prob": float(ans["safety_hazard"]["noul"]),
                "det_score": int(ans["deterioration_risk"]["score"]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. POLICY: every judgement call is a named, visible, adjustable number
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    safety_weight: float = 6.0            # max points from safety probability
    deterioration_weight: float = 4 / 3   # points per det_score step (max 4)
    equity_weight: float = 0.5            # 0 = efficiency-only, 1 = distance-blind
    travel_cost_per_100km: float = 0.25   # how hard distance penalises a job (efficiency view)
    remote_uplift_per_100km: float = 0.10 # extra boost for remoteness, scaled by equity_weight
    wait_rate_per_day: float = 0.02       # fairness to EVERYONE who waits, any location
    wait_cap: float = 1.0                 # ageing can add at most +100%
    safety_floor: float = 0.70            # >= this probability => always Tier 0
    max_wait_days: int = 45               # guardrail: no job waits longer than this
    guardrail_min_urgency: float = 2.0    # ...but only jobs this urgent can jump the queue
    review_band: tuple = (0.40, 0.60)     # model "unsure" zone => human check
    remote_km: float = 100.0              # what counts as "remote" in summaries


# --------------------------------------------------------------------------
# 3. SCORING
# --------------------------------------------------------------------------
def score_tickets(signals: pd.DataFrame, policy: Policy = Policy()) -> pd.DataFrame:
    d = signals.copy()
    e = policy.equity_weight

    d["base_urgency"] = (
        d.safety_prob * policy.safety_weight
        + d.det_score * policy.deterioration_weight
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
    return "; ".join(reasons)


# --------------------------------------------------------------------------
# 4. EXPLANATIONS (generated from the real score components, never free text,
#    so they cannot hallucinate and always match the ranking)
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
        f"Urgency {r.base_urgency:.1f} (safety {r.safety_prob:.0%}, damage-risk level {int(r.det_score)}/3, {r.category})."
    )
    out.append(f"Waiting {int(r.days_waiting)} days adds +{(r.wait_mult - 1):.0%}.")

    if r.distance_km > 0:
        out.append(
            f"At {r.distance_km:.0f} km, travel would cut its score by "
            f"{1 - 1 / r.travel_penalty:.0%} under efficiency-only; the equity setting "
            f"({policy.equity_weight:.0%}) offsets that and adds +{(r.remote_mult - 1):.0%} for remoteness."
        )
    if r.bundle_with_trip:
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
# 5. THE PRICE OF EQUITY: make the trade-off a number, not a feeling
# --------------------------------------------------------------------------
def price_of_equity(scored: pd.DataFrame, jobs_per_week: int, policy: Policy) -> pd.DataFrame:
    """Illustrative: if the crew can finish `jobs_per_week` jobs, compare what
    each ranking would do. Travel = round trip to each distinct location."""
    cols = {}
    for label, rank_col in (("Efficiency-first", "rank_efficiency"), ("Equity policy", "rank")):
        done = scored.nsmallest(jobs_per_week, rank_col)
        left = scored.drop(done.index)
        trips = done.groupby("location")["distance_km"].first()
        cols[label] = {
            "Jobs completed": len(done),
            "Safety-critical jobs completed": int((done.safety_prob >= policy.safety_floor).sum()),
            "Remote jobs completed": int((done.distance_km >= policy.remote_km).sum()),
            "Communities visited": int(trips.size),
            "Total travel (km)": int(2 * trips.sum()),
            "Longest wait still in queue (days)": int(left.days_waiting.max()) if len(left) else 0,
            "Remote jobs still waiting": int((left.distance_km >= policy.remote_km).sum()),
        }
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------
# 6. AUDIT LOG: "let the human own it" means the decision is recorded
# --------------------------------------------------------------------------
def log_decision(path: str | Path, ticket_row, decision: str, justification: str,
                 coordinator: str, policy: Policy) -> None:
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
