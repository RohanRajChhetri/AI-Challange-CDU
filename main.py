from pathlib import Path
from typing import IO

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TICKET_CSV = (
    DATA_DIR / "nt_housing_operations.csv"
    if (DATA_DIR / "nt_housing_operations.csv").exists()
    else Path(__file__).with_name("nt_housing_operations.csv")
)
if not TICKET_CSV.exists():
    TICKET_CSV = (
        DATA_DIR / "maintenance_tickets.csv"
        if (DATA_DIR / "maintenance_tickets.csv").exists()
        else Path(__file__).with_name("maintenance_tickets.csv")
    )

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


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EQUITRIAGE // Public Housing Logistics & Decision Engine (CLI Demonstration)")
    print("=" * 70)
    print("Loading Laya decision model into memory...")
    from laya import Router
    from equitriage_engine import (
        Policy,
        extract_signals,
        score_tickets,
        price_of_equity,
        compute_manifest_breakdown,
    )

    router = Router(preload=True)
    raw_tickets = load_tickets(TICKET_CSV)
    print(f"Loaded {len(raw_tickets)} authentic NT housing maintenance requests.")
    print("Running context-aware RAG extraction via Laya AI...")

    signals = extract_signals(raw_tickets, router)
    policy = Policy(equity_weight=0.55, safety_floor=0.70, max_wait_days=45)
    scored = score_tickets(signals, policy, jobs_per_week=5)

    print("\n" + "-" * 70)
    print("TOP DISPATCH MANIFEST (EQUITY-ADJUSTED WITH BUNDLED RUNS):")
    print("-" * 70)
    summary_cols = ["rank", "ticket_id", "location", "category", "days_waiting", "tier", "bundle_status", "final_score"]
    print(scored.head(10)[summary_cols].to_string(index=False))

    print("\n" + "-" * 70)
    print("INTELLIGENT TRIP BUNDLING & COST RECOVERY MANIFEST:")
    print("-" * 70)
    manifest = compute_manifest_breakdown(scored, jobs_per_week=5, policy=policy)
    print(manifest.to_string(index=False))

    print("\n" + "-" * 70)
    print("PRICE OF EQUITY (EFFICIENCY-FIRST VS EQUITRIAGE TRADE-OFF):")
    print("-" * 70)
    poe = price_of_equity(scored, jobs_per_week=5, policy=policy)
    print(poe.to_string())

    print("\n" + "=" * 70)
    print("[OK] Pipeline executed successfully.")
    print("Run `streamlit run app.py` for the interactive 3D geospatial dashboard.")
    print("=" * 70 + "\n")
