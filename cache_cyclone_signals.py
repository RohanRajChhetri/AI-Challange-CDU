"""Pre-cache Laya signals for cyclone incident catalog so the slider runs interactively without delay."""
import pandas as pd
import json
from pathlib import Path
from laya import Router
from equitriage_engine import LAYA_QUESTIONS

print("Loading Laya router...")
router = Router(preload=True)

DATA_DIR = Path(__file__).parent / "data"
catalog_file = DATA_DIR / "cyclone_incident_catalog.csv" if (DATA_DIR / "cyclone_incident_catalog.csv").exists() else Path("cyclone_incident_catalog.csv")
out_cache = DATA_DIR / "cyclone_signals_cache.json"

df = pd.read_csv(catalog_file)
records = df.to_dict("records")

# Questions including chronic_failure
q = dict(LAYA_QUESTIONS)
q["chronic_failure"] = {
    "type": "noul",
    "instructions": "Based on the maintenance history and current request, does this issue represent a chronic, repeat, or recurrent failure at this property?",
}

print(f"Running Laya extraction on {len(records)} cyclone tickets from {catalog_file}...")
requests = [{"state": t["text"], "questions": q} for t in records]
batch_results = router.predict_batch(requests)

cached = []
for t, res in zip(records, batch_results):
    ans = res["answers"]
    cached.append({
        **t,
        "category": ans["category"]["choice"],
        "safety_prob": float(ans["safety_hazard"]["noul"]),
        "det_score": float(ans["deterioration_risk"]["score"]),
        "chronic_prob": float(ans["chronic_failure"]["noul"]),
    })

with open(out_cache, "w", encoding="utf-8") as f:
    json.dump(cached, f, indent=2)

print(f"Saved {out_cache} successfully!")
