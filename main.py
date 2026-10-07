from pathlib import Path
from typing import IO

import pandas as pd
import math

TICKET_CSV = Path(__file__).with_name("maintenance_tickets.csv")
REQUIRED_TICKET_COLUMNS = {
    "ticket_id",
    "location",
    "distance_km",
    "days_waiting",
    "text",
}


def load_tickets(source: str | Path | IO[bytes]) -> pd.DataFrame:
    tickets = pd.read_csv(
        source,
        dtype={"ticket_id": "string", "location": "string", "text": "string"},
    )
    missing_columns = REQUIRED_TICKET_COLUMNS - set(tickets.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")
    if tickets.empty:
        raise ValueError("CSV must contain at least one ticket.")

    for column in ("ticket_id", "location", "text"):
        tickets[column] = tickets[column].str.strip()
        if tickets[column].isna().any() or tickets[column].eq("").any():
            raise ValueError(f"Every ticket must have a non-empty {column}.")

    if tickets["ticket_id"].duplicated().any():
        raise ValueError("ticket_id values must be unique.")

    for column in ("distance_km", "days_waiting"):
        tickets[column] = pd.to_numeric(tickets[column], errors="coerce")
        if tickets[column].isna().any() or tickets[column].lt(0).any():
            raise ValueError(f"{column} must contain non-negative numbers.")

    return tickets

# ==========================================
# 3. LAYA QUESTIONS DEFINITION
# ==========================================
# Instead of prompting an LLM to generate text, we define strict typed questions.
laya_questions = {
    "category": {
        "type": "choice",
        "instructions": "Categorize the primary issue in this maintenance request.",
        "criteria": {
            "plumbing": "Water, leaks, pipes, or drainage issues",
            "electrical": "Power, wiring, outlets, or appliance issues",
            "structural": "Roof, walls, windows, doors, or physical damage",
            "pest": "Insects, rodents, or animal issues",
            "other": "Everything else"
        }
    },
    "safety_hazard": {
        "type": "noul",
        "instructions": "Does this issue present an immediate physical safety, fire, or severe health hazard to the tenants?"
    },
    "deterioration_risk": {
        "type": "score",
        "instructions": "Rate the risk of this issue causing significant property damage if left unattended.",
        "criteria": [
            "Stable, will not get worse immediately",
            "Minor deterioration over time",
            "Will cause significant damage if left for a month",
            "Imminent failure or catastrophic damage"
        ]
    }
}

# ==========================================
# 4. THE EQUI-TRIAGE ALGORITHM
# ==========================================
def process_tickets(tickets, router):
    results = []
    
    for ticket in tickets:
        # Laya evaluates the ticket text against our strict questions in a single forward pass (~35ms)
        laya_response = router.predict(state=ticket["text"], questions=laya_questions)
        
        answers = laya_response["answers"]
        
        # 1. Extract typed values from Laya
        category = answers["category"]["choice"]
        safety_prob = answers["safety_hazard"]["noul"]  # Returns a probability (0.0 to 1.0)
        # Score returns an index (0 to 3 based on our 4 criteria levels)
        det_score = answers["deterioration_risk"]["score"] 
        
        # 2. Base Urgency (AI extraction of the physical risk)
        # Weight safety heavily (max 6 points) and deterioration (max 4 points)
        base_urgency = (safety_prob * 6) + (det_score * 1.33)
        
        # 3. Equity Multiplier (The Bias Breaker)
        # 1.0 is baseline. Add 2% for every day waiting, and 0.1% for every km away.
        equity_multiplier = 1.0 + (ticket["days_waiting"] * 0.02) + (ticket["distance_km"] * 0.001)
        
        # 4. Final EquiTriage Score
        final_score = base_urgency * equity_multiplier
        
        # 5. TRUST TWIST: Human Exception Flagging
        # We flag ambiguous edge-cases for human review if safety probability sits in the middle (unsure)
        requires_human_review = 0.4 < safety_prob < 0.6 

        results.append({
            "Ticket": ticket["ticket_id"],
            "Location": ticket["location"],
            "Days Wait": ticket["days_waiting"],
            "Issue": category.title(),
            "Safety (Prob)": f"{safety_prob:.2f}",
            "Base Urgency": round(base_urgency, 2),
            "Equity Multiplier": round(equity_multiplier, 2),
            "Final Score": round(final_score, 2),
            "Human Review": "Yes" if requires_human_review else "No"
        })
        
    return pd.DataFrame(results)

# ==========================================
# 5. EXECUTION & OUTPUT
# ==========================================
if __name__ == "__main__":
    print("\nRunning EquiTriage Pipeline using Native Laya...")
    print("Loading Laya decision model into memory...")
    from laya import Router

    router = Router(preload=True)
    df = process_tickets(load_tickets(TICKET_CSV).to_dict("records"), router)
    
    # Sort by Final Score (Descending) to simulate the dashboard queue
    df_sorted = df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
    
    print("\n=== WEEKLY DISPATCH RANKING (EQUITY ADJUSTED) ===")
    print(df_sorted.to_string())
    
    print("\n[!] Notice the Equity Multiplier effect.")
    print("[!] High distance and high wait times artificially inflate remote ticket scores,")
    print("[!] breaking the standard urban efficiency feedback loop.")
