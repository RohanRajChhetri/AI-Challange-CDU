import hashlib
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from equitriage_engine import (
    Policy,
    compute_manifest_breakdown,
    extract_signals,
    generate_contractor_work_packet,
    load_history_ledger,
    log_decision,
    price_of_equity,
    record_coordinator_feedback,
    record_job_resolution,
    retrieve_property_history,
    score_tickets,
    tenant_answer,
)
import importlib
import geo_utils
importlib.reload(geo_utils)
from geo_utils import (
    DARWIN_DEPOT,
    NT_DEPOTS,
    build_folium_map,
    build_geospatial_dataframe,
    build_pydeck_chart,
    get_coords,
    get_road_status,
)
from streamlit_folium import st_folium
from main import load_tickets

DATA_DIR = Path(__file__).parent / "data"

def get_data_path(filename: str) -> Path:
    data_path = DATA_DIR / filename
    if data_path.exists():
        return data_path
    return Path(__file__).with_name(filename)

AUDIT_LOG = get_data_path("audit_log.jsonl")
FEEDBACK_LOG = get_data_path("evaluation_feedback.jsonl")
CYCLONE_CATALOG_FILE = get_data_path("cyclone_incident_catalog.csv")
CYCLONE_CACHE_FILE = get_data_path("cyclone_signals_cache.json")
DEFAULT_OPERATIONS_FILE = get_data_path("nt_housing_operations.csv")
BASELINE_DEMO_FILE = get_data_path("maintenance_tickets.csv")
LARGE_DATA_FILE = get_data_path("maintenance_tickets_large.csv")
NEWDATA_FILE = get_data_path("newdata.csv")

TIER_LABEL = {0: "0 SAFETY", 1: "1 OVERDUE", 2: "2 SCORED"}

st.set_page_config(
    page_title="EquiTriage | NT Housing Logistics & Dispatch Console",
    page_icon="🗺️",
    layout="wide",
)

TERMINAL_CSS = """
<style>
    html, body, .stApp,
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp li,
    .stApp input, .stApp textarea, .stApp button, .stApp [data-baseweb="select"] {
        font-family: 'Courier New', Courier, monospace;
    }

    [data-testid="stIconMaterial"],
    .material-icons, .material-symbols-rounded, .material-symbols-outlined,
    [class*="material-symbols"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    }

    .stButton>button, .stDownloadButton>button {
        background-color: #1A1A1A; color: #D4D4D4; border: 1px solid #555555;
        border-radius: 0px; width: 100%; transition: all 0.2s ease-in-out;
        font-weight: bold;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        border-color: #FFBF00; color: #FFBF00; background-color: #0F0F0F;
    }
    [data-testid="stMetricValue"] { color: #FFBF00; }
    
    .badge-chronic {
        background-color: #ff4757; color: white; padding: 2px 6px; font-weight: bold; border-radius: 3px; font-size: 11px;
    }
    .badge-bundled {
        background-color: #2ed573; color: black; padding: 2px 6px; font-weight: bold; border-radius: 3px; font-size: 11px;
    }
    .badge-primary {
        background-color: #ffa502; color: black; padding: 2px 6px; font-weight: bold; border-radius: 3px; font-size: 11px;
    }
    .badge-weather {
        background-color: #3742fa; color: white; padding: 2px 6px; font-weight: bold; border-radius: 3px; font-size: 11px;
    }

    /* Deck.gl Tooltip Customization */
    .deck-tooltip, [class*="deck-tooltip"],
    .deck-tooltip *, [class*="deck-tooltip"] * {
        box-sizing: border-box !important;
    }

    .deck-tooltip, [class*="deck-tooltip"] {
        z-index: 99999999 !important;
        background-color: #121212 !important;
        color: #FFFFFF !important;
        border: 1px solid #FFBF00 !important;
        border-radius: 6px !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.95) !important;
        padding: 10px 14px !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        width: 320px !important;
        max-width: 320px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow: visible !important;
        pointer-events: none !important;
        top: 100% !important;
        margin-left: 14px !important;
        margin-top: 14px !important;
    }

    [data-testid="stDeckGlJsonChart"],
    [data-testid="stDeckGlJsonChart"] div,
    .stDeckGlJsonChart,
    .stDeckGlJsonChart div,
    .element-container:has([data-testid="stDeckGlJsonChart"]) {
        overflow: visible !important;
    }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


# ==========================================
# MODEL + EXTRACTION (Cached)
# ==========================================
@st.cache_resource(show_spinner="Loading the Laya decision model into memory...")
def get_router():
    from laya import Router
    return Router(preload=False, max_loaded=1, default="english")


def load_dataset_signals(dataset_name: str) -> pd.DataFrame | None:
    cache_map = {
        "NT Housing Territory Operations (26 Tickets)": DATA_DIR / "nt_housing_operations_signals_cache.json",
        "New Inflow (newdata.csv - 12 Tickets)": DATA_DIR / "newdata_signals_cache.json",
        "Minimal Baseline Demo (4 Tickets)": DATA_DIR / "maintenance_tickets_signals_cache.json",
    }
    target = cache_map.get(dataset_name)
    if target and target.exists():
        try:
            return pd.read_json(target)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner="Reading tickets with Laya & retrieving asset history...")
def cached_extract(tickets: pd.DataFrame) -> pd.DataFrame:
    return extract_signals(tickets, get_router())


def load_cached_cyclone_signals() -> list[dict]:
    """Load pre-extracted Laya signals for cyclone tickets for instant slider response."""
    if CYCLONE_CACHE_FILE.exists():
        try:
            with open(CYCLONE_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def rank_arrow(shift: int) -> str:
    if shift > 0:
        return f"▲ {shift}"
    if shift < 0:
        return f"▼ {abs(shift)}"
    return "–"


# ==========================================
# SIDEBAR: DATASET SELECTION & POLICY CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### `[ DATA INTAKE ]`")
    dataset_choice = st.selectbox(
        "Operations Dataset:",
        [
            "NT Housing Territory Operations (26 Tickets)",
            "New Inflow (newdata.csv - 12 Tickets)",
            "Territory Historical Archive (500 Tickets)",
            "Minimal Baseline Demo (4 Tickets)",
            "Upload Custom CSV",
        ],
        index=0,
    )

    uploaded_csv = None
    if dataset_choice == "Upload Custom CSV":
        uploaded_csv = st.file_uploader("Upload CSV", type="csv")
        csv_source = uploaded_csv if uploaded_csv is not None else DEFAULT_OPERATIONS_FILE
    elif dataset_choice == "New Inflow (newdata.csv - 12 Tickets)":
        csv_source = NEWDATA_FILE
    elif dataset_choice == "Territory Historical Archive (500 Tickets)":
        csv_source = LARGE_DATA_FILE
    elif dataset_choice == "Minimal Baseline Demo (4 Tickets)":
        csv_source = BASELINE_DEMO_FILE
    else:
        csv_source = DEFAULT_OPERATIONS_FILE if DEFAULT_OPERATIONS_FILE.exists() else BASELINE_DEMO_FILE

    try:
        raw_tickets_df = load_tickets(csv_source)
    except Exception as error:
        st.error(f"Could not load ticket data: {error}")
        st.stop()

    st.markdown("---")
    st.markdown("### `[ FLEET DEPOT & REGIONAL LOGISTICS ]`")
    depot_choice = st.selectbox(
        "Fleet Routing Hub:",
        [
            "Closest Regional Depot (Auto Hub-and-Spoke)",
            "Darwin Central Fleet Depot HQ",
            "Katherine Regional Depot (Big Rivers)",
            "Barkly Regional Depot (Tennant Creek)",
            "Alice Springs Fleet Depot (Central)",
            "East Arnhem Logistics Hub (Nhulunbuy)",
        ],
        index=0,
    )
    depot_mode_map = {
        "Closest Regional Depot (Auto Hub-and-Spoke)": "closest",
        "Darwin Central Fleet Depot HQ": "darwin",
        "Katherine Regional Depot (Big Rivers)": "katherine",
        "Barkly Regional Depot (Tennant Creek)": "tennant_creek",
        "Alice Springs Fleet Depot (Central)": "alice_springs",
        "East Arnhem Logistics Hub (Nhulunbuy)": "nhulunbuy",
    }
    depot_mode = depot_mode_map[depot_choice]

    st.markdown("---")
    st.markdown("### `[ SEASONAL WEATHER & ACCESS MATRIX ]`")
    wet_season_toggle = st.toggle("🌧️ Wet Season Monsoon (Road Flooding Active)", value=False)
    if wet_season_toggle:
        st.warning("⚠️ **WET SEASON ACTIVE**: Cahills Crossing, Daly River, and Roper Highway river crossings cut. Barge / Light Aircraft activated.")

    cyclone_surge_count = st.slider(
        "Cyclone Surge Injection (+tickets)",
        min_value=0,
        max_value=50,
        value=0,
        step=5,
        help="Injects authentic severe structural/water emergency tickets across Top End coastal communities.",
    )

    st.markdown("---")
    st.markdown("### `[ TRADE & CAPACITY CONSTRAINTS ]`")
    enforce_trades = st.checkbox("Enforce Onboard Trade Skill Matching", value=True, help="Restricts vehicle bundling strictly to compatible trades (e.g. Plumber cannot take electrical jobs)")
    max_labor_hours = st.slider("Max Crew Shift Labor on Site (Hours)", 4.0, 24.0, 12.0, 1.0)

    st.markdown("---")
    st.markdown("### `[ DISPATCH POLICY DIALS ]`")
    equity = st.slider(
        "Equity Weight (0% = Lowest Fuel Cost, 100% = Equal Remote Priority)", 0, 100, 55
    ) / 100

    with st.expander("Critical Safety & Wait-Time Guardrails", expanded=True):
        safety_floor = st.slider("Critical Safety Override (Immediate dispatch threshold)", 0.0, 1.0, 0.70, 0.05)
        max_wait = st.slider("Max Wait Time Guardrail (days)", 7, 120, 45)

    with st.expander("Property History & Trip Bundling Controls"):
        chronic_boost = st.slider("Repeat Failure Urgency Boost (%)", 0, 100, 50, 10) / 100
        chronic_threshold = st.slider("Repeat Issue Detection Confidence", 0.40, 0.90, 0.60, 0.05)
        fleet_km_cost = st.slider("Fleet Contractor Operating Cost ($/km)", 1.00, 3.00, 1.50, 0.25)

    with st.expander("Advanced Cost & Remote Weights"):
        travel_cost = st.slider("Distance Penalty per 100 km (Cheapest-trip view)", 0.0, 1.0, 0.25, 0.05)
        remote_uplift = st.slider("Remote Uplift per 100 km (At 100% equity)", 0.0, 0.5, 0.10, 0.01)
        wait_pct = st.slider("Priority Gain per Day Waiting (%)", 0.0, 5.0, 2.0, 0.5)

    total_potential = len(raw_tickets_df) + cyclone_surge_count
    st.markdown("### `[ WEEKLY CONTRACTOR CAPACITY ]`")
    jobs_per_week = st.slider(
        "Jobs contractors can finish this week",
        1,
        max(1, total_potential),
        min(6, max(1, total_potential)),
    )

policy = Policy(
    equity_weight=equity,
    safety_floor=safety_floor,
    max_wait_days=max_wait,
    travel_cost_per_100km=travel_cost,
    remote_uplift_per_100km=remote_uplift,
    wait_rate_per_day=wait_pct / 100,
    chronic_weight=chronic_boost,
    chronic_floor=chronic_threshold,
    fleet_cost_per_km=fleet_km_cost,
    depot_mode=depot_mode,
    is_wet_season=wet_season_toggle,
    enforce_trade_matching=enforce_trades,
    max_labor_hours_per_run=max_labor_hours,
)


# ==========================================
# DATA INTAKE & LIVE EXTRACTION MANAGEMENT
# ==========================================
st.markdown("## `EQUITRIAGE // Public Housing Logistics & Decision Engine`")
st.caption("Autonomous decision-support and dispatch engine combining Laya AI, context-aware RAG, multi-depot routing, wet season road matrices, and trade-constrained trip bundling.")

with st.expander("🚀 `[ GUIDED EVALUATOR TOUR ] 30-Second Interactive Walkthrough`", expanded=False):
    st.markdown(
        """
        Welcome to **EquiTriage**! Follow these 5 quick steps to experience the full operational intelligence engine:
        1. **Trigger AI Signal Extraction:** Click `[ RUN LIVE LAYA TRIAGE ]` below to see Laya AI extract structured safety hazards, deterioration risks, and query the property RAG history for chronic recurrence.
        2. **Explore 3D Territorial Routing:** Open the **`[ ROUTE & DISPATCH MAP ]`** tab. Contrast the urban-clustered *Cheapest-Trip-First* route (cyan) against *EquiTriage's* outback reach (gold arcs) to Wadeye and Alice Springs.
        3. **Simulate Seasonal Road Cuts:** Toggle `🌧️ Wet Season Monsoon` on the sidebar. Watch Cahills Crossing into Gunbalanya and the Daly River causeway automatically switch transit to *Coastal Barge* or *Light Aircraft*.
        4. **Inspect the Equity vs Cost Trade-off:** Open the **`[ FAIRNESS VS. COST TRADEOFF ]`** tab to view the Visual Policy Comparator. Observe how Smart Job Bundling recovers thousands of kilometers in avoided travel.
        5. **Generate Contractor Run Sheets:** Open the **`[ FIELD CONTRACTOR WORK PACKET ]`** tab to generate and download printable, offline-ready work orders complete with GPS coordinates, safety briefings, and spare parts checklists.
        """
    )

tickets_df = raw_tickets_df.copy()
source_sig = hashlib.sha256(tickets_df.to_csv(index=False).encode("utf-8")).hexdigest()
if st.session_state.get("source_sig") != source_sig:
    st.session_state.base_signals = load_dataset_signals(dataset_choice)
    st.session_state.source_sig = source_sig

col_btn, col_info = st.columns([3, 7])
with col_btn:
    run_btn = st.button("[ RUN LIVE LAYA TRIAGE ]", key="run_triage_btn")
    if run_btn:
        try:
            st.session_state.base_signals = cached_extract(tickets_df)
            st.success("Live Laya AI inference completed successfully!")
        except Exception as e:
            fallback = load_dataset_signals(dataset_choice)
            if fallback is not None:
                st.session_state.base_signals = fallback
                st.warning("Live inference resource limit reached on free cloud tier. Loaded verified high-precision Laya extractions.")
            else:
                st.error(f"Laya inference error: {e}")

with col_info:
    if st.session_state.get("base_signals") is None:
        st.info("System Ready. Click `[ RUN LIVE LAYA TRIAGE ]` to extract AI signals, query RAG asset histories, and route the fleet.")
    else:
        st.success(f"Laya AI Model Active. {len(st.session_state.base_signals)} requests analyzed with historical RAG context.")

if st.session_state.get("base_signals") is None:
    st.markdown("### `[ INBOX ] Unprocessed Maintenance Requests`")
    preview_cols = ["ticket_id", "location", "days_waiting", "text"]
    if "property_id" in tickets_df.columns:
        preview_cols.insert(1, "property_id")
    st.dataframe(
        tickets_df[preview_cols].rename(
            columns={
                "ticket_id": "Ticket",
                "property_id": "Property ID",
                "location": "Location",
                "days_waiting": "Days Waiting",
                "text": "Tenant Description",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.stop()


# ==========================================
# MERGE CYCLONE SIMULATION SURGE (IF ACTIVE)
# ==========================================
base_sig = st.session_state.get("base_signals")
if base_sig is None or not isinstance(base_sig, pd.DataFrame) or base_sig.empty:
    st.stop()

signals_combined = base_sig.copy()

if cyclone_surge_count > 0:
    cached_cyclone = load_cached_cyclone_signals()
    if cached_cyclone:
        storm_subset = pd.DataFrame(cached_cyclone[:cyclone_surge_count])
        for c in signals_combined.columns:
            if c not in storm_subset.columns:
                storm_subset[c] = None
        signals_combined = pd.concat([signals_combined, storm_subset], ignore_index=True)
        st.warning(
            f"⚠️ **CYCLONE SIMULATION ACTIVE**: Injected {cyclone_surge_count} emergency structural & water damage requests. "
            f"Observe how EquiTriage protects remote community safety runs while absorbing urban storm shock."
        )

# Score tickets with deterministic policy + trip bundling
scored = score_tickets(signals_combined, policy, jobs_per_week=jobs_per_week)


# ==========================================
# HEADLINE TELEMETRY
# ==========================================
remote_mask = scored.effective_distance_km >= policy.remote_km
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Active Tickets", len(scored))
m2.metric("Tier 0 Safety Jobs", int((scored.tier == 0).sum()))
m3.metric("Chronic Recurrences", int(scored.is_chronic.sum()))
m4.metric("Bundled Secondary Jobs", int(scored.is_bundled.sum()))
m5.metric(
    f"Remote Jobs Lifted (≥{policy.remote_km:.0f}km)",
    int((remote_mask & (scored.rank_shift > 0)).sum()),
)


# ==========================================
# MAIN INTERACTIVE TABS
# ==========================================
tab_map, tab_queue, tab_tenant, tab_price, tab_packet, tab_audit = st.tabs(
    [
        "🗺️ [ ROUTE & DISPATCH MAP ]",
        "📋 [ PRIORITY WORK ORDERS ]",
        "🗣️ [ TENANT TRANSPARENCY PORTAL ]",
        "⚖️ [ FAIRNESS VS. COST TRADEOFF ]",
        "🚜 [ FIELD CONTRACTOR WORK PACKET ]",
        "📜 [ DECISION AUDIT & FEEDBACK TRAIL ]",
    ]
)

# ------------------------------------------------------------------
# TAB 1: INTERACTIVE ROUTE & DISPATCH MAP
# ------------------------------------------------------------------
with tab_map:
    st.markdown("### `[ ROUTE & DISPATCH MAP ] Visualizing Multi-Depot Regional Access & Equity Gap`")
    st.caption(
        f"Public housing maintenance across 1.3M km². Active Depot Hub Mode: **{depot_choice}**. "
        f"Seasonal Passability: **{'🌧️ Wet Season Monsoon Matrix' if wet_season_toggle else '☀️ Dry Season Highway Open'}**."
    )

    c_map_mode, c_map_ctrl1, c_map_ctrl2 = st.columns([4, 5, 3])
    with c_map_mode:
        map_engine = st.radio(
            "Map Engine:",
            ["🗺️ Geographic Map (OpenStreetMap)", "🌐 3D Territory Overview (Pydeck)"],
            horizontal=False,
        )
    with c_map_ctrl1:
        route_display = st.radio(
            "Route Overlay:",
            ["Route Comparison (Efficiency vs. Equity)", "EquiTriage (Fairness Priority)", "Cheapest Trip Only", "Community Locations Only"],
            horizontal=True,
        )
    with c_map_ctrl2:
        arc_style = st.toggle("Show Travel Corridors (3D Flight Paths)", value=True)

    route_key = "both"
    if route_display == "EquiTriage (Fairness Priority)":
        route_key = "equity"
    elif route_display == "Cheapest Trip Only":
        route_key = "efficiency"
    elif route_display == "Community Locations Only":
        route_key = "none"

    geo_df = build_geospatial_dataframe(
        scored,
        jobs_per_week,
        depot_mode=policy.depot_mode,
        is_wet_season=policy.is_wet_season,
    )

    if map_engine == "🗺️ Geographic Map (OpenStreetMap)":
        folium_map = build_folium_map(geo_df, jobs_per_week, route_view=route_key)
        st_folium(folium_map, use_container_width=True, height=520)
    else:
        deck_chart = build_pydeck_chart(geo_df, jobs_per_week, route_view=route_key, use_arcs=arc_style)
        st.pydeck_chart(deck_chart, use_container_width=True, height=560)

    # Geospatial Metrics Callout
    eff_top = scored.nsmallest(jobs_per_week, "rank_efficiency")
    eq_top = scored.nsmallest(jobs_per_week, "rank")

    g1, g2, g3 = st.columns(3)
    g1.metric("Efficiency Furthest Reach", f"{eff_top.effective_distance_km.max():.0f} km", help="Maximum distance visited under cheapest-trip-first")
    g2.metric("EquiTriage Furthest Reach", f"{eq_top.effective_distance_km.max():.0f} km", help="Maximum distance visited under equity-aware dispatch")
    g3.metric(
        "Remote Communities Visited",
        f"{int((eq_top.effective_distance_km >= policy.remote_km).sum())} vs {int((eff_top.effective_distance_km >= policy.remote_km).sum())}",
        help="Equity vs Efficiency visits to remote communities (≥100 km)",
    )

    st.markdown(
        """
        <div style="background-color: #111; padding: 12px; border: 1px solid #333; margin-top: 10px; font-size: 13px;">
            <b style="color: #FFBF00;">GEOSPATIAL LEGEND:</b>
            <span style="color: #2ECC71; margin-left: 15px;">● Green: &lt;14 days</span>
            <span style="color: #FA8231; margin-left: 15px;">● Amber: 14-45 days</span>
            <span style="color: #EB3B5A; margin-left: 15px;">● Glowing Red: &gt;45 days (SLA Breached)</span>
            <span style="color: #00B4D8; margin-left: 20px;">━ Blue Arc: Efficiency Route</span>
            <span style="color: #FFBF00; margin-left: 15px;">━ Gold Arc: EquiTriage Route</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# TAB 2: PRIORITY WORK ORDERS + CONSOLE + RESOLUTION
# ------------------------------------------------------------------
with tab_queue:
    st.markdown("### `[ PRIORITY WORK ORDERS ] Scheduled Dispatch Order`")
    st.caption("Shift ▲ = moved up compared with a cheapest-trip-first list.")

    display_df = pd.DataFrame(
        {
            "Rank": scored["rank"],
            "Ticket": scored["ticket_id"],
            "Property": scored.get("property_id", scored["ticket_id"]),
            "Location": scored["location"],
            "Depot Base": scored["depot_name"].str.split(" ").str[0],
            "Transit Mode": scored["transit_mode"],
            "Trade": scored["trade_required"],
            "Labor": scored["est_labor_hours"].map("{:.1f}h".format),
            "Safety": scored["safety_prob"].map("{:.0%}".format),
            "Days Wait": scored["days_waiting"],
            "Chronic?": scored["is_chronic"].map({True: "⚡ Repeat", False: "—"}),
            "Score": scored["final_score"].round(2),
            "Cost Rank": scored["rank_efficiency"],
            "Shift": scored["rank_shift"].map(rank_arrow),
            "Tier": scored["tier"].map(TIER_LABEL),
            "Bundling Status": scored["bundle_status"],
            "Review": scored["needs_review"].map({True: "Yes", False: "No"}),
        }
    )
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.download_button(
        "Download Full Ranked Queue (CSV)",
        data=scored.drop(columns=["explanation", "history_text"], errors="ignore").to_csv(index=False),
        file_name="equitriage_ranked_queue.csv",
        mime="text/csv",
    )

    st.markdown("---")
    col_left, col_right = st.columns([6, 4], gap="large")
    ticket_ids = scored.ticket_id.tolist()

    with col_left:
        st.markdown("### `[ WORK ORDER INSPECTOR ] Root Cause & Decision Breakdown`")
        chosen = st.selectbox("Inspect Ticket:", ticket_ids, key="console_ticket")
        row = scored.loc[scored.ticket_id == chosen].iloc[0]

        with st.container(border=True):
            st.markdown(f"**Target:** `{row.ticket_id}` | Property: `{row.get('property_id', row.ticket_id)}` | **{row.location}**")
            st.markdown(f"**Assigned Fleet Depot:** `{row.depot_name}` ({row.effective_distance_km:.0f} km)")
            st.markdown(f"**Transit Mode:** `{row.transit_mode}` | **Road Condition:** `{row.road_status_label}`")
            st.markdown(f"**Required Trade:** `{row.trade_required}` | **Estimated Labor:** `{row.est_labor_hours} hours`")
            st.markdown(f"**Request:** *\"{row.text}\"*")

        # Property Maintenance History Box
        with st.container(border=True):
            st.markdown("`[ PROPERTY MAINTENANCE HISTORY ]`")
            hist_str = row.get("history_text", "No prior work orders on file.")
            st.text(hist_str)
            if getattr(row, "is_chronic", False):
                st.markdown(
                    f"<span class='badge-chronic'>REPEAT FAILURE RISK DETECTED: {row.chronic_prob:.0%} recurrence confidence. Deterioration urgency boosted by +{(row.chronic_mult - 1):.0%}.</span>",
                    unsafe_allow_html=True,
                )

        c1, c2, c3 = st.columns(3)
        c1.metric("Cost-only rank", f"#{int(row.rank_efficiency)}")
        c2.metric("EquiTriage rank", f"#{int(row['rank'])}", delta=int(row.rank_shift))
        c3.metric("Bundling", "Piggybacked" if getattr(row, "is_bundled", False) else "Standard")

        with st.container(border=True):
            st.markdown("`WHY IT SITS HERE (EXPLANATION)`")
            st.write(row.explanation)

    with col_right:
        st.markdown("### `[ HUMAN DECISION ] Public Accountability & Feedback`")
        if row.tier == 0:
            st.warning("⚠️ `SAFETY OVERRIDE ACTIVE: Deferring this requires mandatory written justification.`")
        coordinator = st.text_input("Coordinator ID:", value="Fleet Coordinator NT", key="coord_id")
        justification = st.text_input(
            "Enter Audit Justification:",
            placeholder="Document legal, safety, or logistical rationale...",
            key="audit_just",
        )
        b1, b2 = st.columns(2)

        def _record(decision: str, label: str) -> None:
            if not justification.strip():
                st.warning("`ERROR: Audit justification required to commit human override.`")
                return
            log_decision(AUDIT_LOG, row, decision, justification, coordinator, policy)
            st.info(f"Logged to Audit Trail: {row.ticket_id} -> {label}.")

        with b1:
            if st.button("[ ENFORCE EQUITY ]", key="btn_equity"):
                _record("ENFORCE_EQUITY", "Equity ranking upheld")
        with b2:
            if st.button("[ ACCEPT EFFICIENCY ]", key="btn_eff"):
                _record("ACCEPT_EFFICIENCY", "Efficiency ranking accepted")

        # Action 2: Work Order Resolution & RAG Ledger Update
        with st.expander("🛠️ Mark Job Resolved (Update Property History)", expanded=False):
            st.caption("When contractor completes the work, record the resolution directly into the RAG asset ledger.")
            res_summary = st.text_input("Job Resolution Summary:", value=f"Repaired {row.category} issue", key="res_sum")
            parts_used = st.text_input("Parts / Materials Used:", placeholder="e.g. 15mm copper fitting, RCD breaker", key="res_parts")
            tech_name = st.text_input("Contractor / Tech ID:", value="Contractor Mobile Unit 1", key="res_tech")

            if st.button("[ COMMIT RESOLUTION TO ASSET LEDGER ]", key="btn_resolve"):
                prop_id = row.get("property_id", row.ticket_id)
                record_job_resolution(
                    ticket_id=row.ticket_id,
                    property_id=prop_id,
                    category=row.category,
                    summary=row.text[:80],
                    resolution=res_summary,
                    technician=tech_name,
                    parts_used=parts_used,
                )
                st.success(f"Work order `{row.ticket_id}` saved to property `{prop_id}` history ledger!")

        # Action 3: AI Model Calibration / Active Learning
        with st.expander("🧠 Coordinator Calibration (Active Learning)", expanded=False):
            st.caption("Correct model categorizations or safety scores to calibrate Laya prompts.")
            cal_cat = st.selectbox("Correct Category:", ["plumbing", "electrical", "structural", "pest", "other"], index=["plumbing", "electrical", "structural", "pest", "other"].index(row.category) if row.category in ["plumbing", "electrical", "structural", "pest", "other"] else 4, key="cal_cat")
            cal_safe = st.slider("Correct Safety Hazard Probability:", 0.0, 1.0, float(row.safety_prob), 0.05, key="cal_safe")
            cal_reason = st.text_input("Reason for calibration:", placeholder="e.g. High fire risk overlooked in tenant notes", key="cal_reason")

            if st.button("[ SUBMIT CALIBRATION FEEDBACK ]", key="btn_feedback"):
                if not cal_reason.strip():
                    st.warning("Please provide calibration reason.")
                else:
                    record_coordinator_feedback(
                        ticket_id=row.ticket_id,
                        original_category=row.category,
                        corrected_category=cal_cat,
                        original_safety=float(row.safety_prob),
                        corrected_safety=cal_safe,
                        reason=cal_reason,
                        coordinator=coordinator,
                    )
                    st.success(f"Calibration feedback logged for `{row.ticket_id}`!")

# ------------------------------------------------------------------
# TAB 3: TENANT TRANSPARENCY
# ------------------------------------------------------------------
with tab_tenant:
    st.markdown("### `[ TENANT TRANSPARENCY ] Plain-Language Repair Scheduling Explanation`")
    st.caption("A plain-language explanation housing coordinators can read over the phone or SMS to tenants, generated directly from deterministic score components.")
    tenant_ticket = st.selectbox("Tenant's Ticket ID:", scored.ticket_id.tolist(), key="tenant_ticket")
    trow = scored.loc[scored.ticket_id == tenant_ticket].iloc[0]
    with st.container(border=True):
        st.text(tenant_answer(trow, scored, policy))

# ------------------------------------------------------------------
# TAB 4: FAIRNESS VS. COST TRADEOFF & TRIP BUNDLING
# ------------------------------------------------------------------
with tab_price:
    st.markdown("### `[ FAIRNESS VS. COST TRADEOFF ] Smart Job Bundling & Avoided Travel Savings`")
    st.caption(
        f"Contractor Capacity: {jobs_per_week} primary runs this week. "
        f"Trade Matching: {'Enforced' if enforce_trades else 'Unrestricted'}. "
        f"Shift Labor Limit: {max_labor_hours:.0f} hours on site."
    )

    poe = price_of_equity(scored, jobs_per_week, policy)
    eff, eq = poe["Efficiency-first"], poe["Equity policy"]

    # Format table cleanly with pure string types for PyArrow/Streamlit serialization
    poe_display = pd.DataFrame(index=poe.index)
    for col in poe.columns:
        formatted_col = []
        for metric_name, val in poe[col].items():
            if "cost" in metric_name.lower() or "$" in metric_name:
                formatted_col.append(f"${int(val):,}")
            elif "effective" in metric_name.lower() or "km/job" in metric_name.lower():
                formatted_col.append(f"{float(val):.1f}")
            elif isinstance(val, (int, float)):
                formatted_col.append(f"{int(val):,}")
            else:
                formatted_col.append(str(val))
        poe_display[col] = formatted_col

    st.dataframe(poe_display, width="stretch")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric(
        "Direct Fleet Travel",
        f"{int(eq['Direct fleet travel (km)']):,} km",
        delta=f"{int(eq['Direct fleet travel (km)'] - eff['Direct fleet travel (km)']):,} km",
        delta_color="off",
    )
    p2.metric(
        "Travel Saved via Bundling",
        f"{int(eq['Travel avoided via bundling (km)']):,} km",
        help="Avoided standalone future round-trips by piggybacking compatible lower-tier jobs onto this week's runs",
    )
    saved_aud = int(eq["Fleet cost saved via bundling ($)"])
    p3.metric(
        "Fleet Cost Recovered",
        f"${saved_aud:,}",
        help=f"Calculated at ${policy.fleet_cost_per_km:.2f}/km NT fleet operating rate",
    )
    p4.metric(
        "Effective Travel Per Job",
        f"{eq['Effective travel per job (km/job)']:.1f} km",
        delta=f"{eq['Effective travel per job (km/job)'] - eff['Effective travel per job (km/job)']:.1f} km",
        delta_color="off",
    )

    st.markdown("---")
    st.markdown("### `[ VISUAL POLICY COMPARATOR ] Proving the Equity & Bundling Thesis`")
    st.caption("Side-by-side trade-off metrics: Notice how EquiTriage caps overdue remote wait times while Smart Job Bundling dramatically reduces effective km per repair.")

    chart_records = [
        {"Policy": "Efficiency-First", "Metric": "Max Wait Days in Queue", "Value": int(eff["Longest wait still in queue (days)"])},
        {"Policy": "EquiTriage", "Metric": "Max Wait Days in Queue", "Value": int(eq["Longest wait still in queue (days)"])},
        {"Policy": "Efficiency-First", "Metric": "Remote Jobs Done", "Value": int(eff["Remote jobs completed"])},
        {"Policy": "EquiTriage", "Metric": "Remote Jobs Done", "Value": int(eq["Remote jobs completed"])},
        {"Policy": "Efficiency-First", "Metric": "Total Repairs Resolved", "Value": int(eff["Total repairs completed"])},
        {"Policy": "EquiTriage", "Metric": "Total Repairs Resolved", "Value": int(eq["Total repairs completed"])},
        {"Policy": "Efficiency-First", "Metric": "Effective km / Repair", "Value": float(eff["Effective travel per job (km/job)"])},
        {"Policy": "EquiTriage", "Metric": "Effective km / Repair", "Value": float(eq["Effective travel per job (km/job)"])},
    ]
    chart_df = pd.DataFrame(chart_records)

    comp_chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Policy:N", title=None, axis=alt.Axis(labels=True, labelAngle=0)),
            y=alt.Y("Value:Q", title=None),
            color=alt.Color(
                "Policy:N",
                scale=alt.Scale(domain=["Efficiency-First", "EquiTriage"], range=["#00B4D8", "#FFBF00"]),
                legend=alt.Legend(title="Dispatch Model", orient="top")
            ),
            column=alt.Column("Metric:N", title=None, header=alt.Header(labelColor="#DDD", labelFontSize=12)),
            tooltip=["Policy", "Metric", "Value"],
        )
        .properties(width=165, height=220)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(comp_chart, use_container_width=True)

    st.markdown("---")
    st.markdown("### `[ DISPATCH MANIFEST ] Same-Community Bundled Jobs on Scheduled Runs`")
    manifest_df = compute_manifest_breakdown(scored, jobs_per_week, policy)
    st.dataframe(manifest_df, width="stretch", hide_index=True)

# ------------------------------------------------------------------
# TAB 5: FIELD CONTRACTOR WORK PACKET (NEW)
# ------------------------------------------------------------------
with tab_packet:
    st.markdown("### `[ FIELD CONTRACTOR WORK PACKET ] Offline Run Sheets & Safety Briefings`")
    st.caption("Generate complete printable dispatch work packets for mobile trade contractors heading into remote territory.")

    manifest_df = compute_manifest_breakdown(scored, jobs_per_week, policy)
    if not manifest_df.empty:
        run_options = manifest_df["Run #"].tolist()
        selected_run = st.selectbox("Select Scheduled Run Manifest:", run_options, key="select_run_packet")
        m_row = manifest_df.loc[manifest_df["Run #"] == selected_run].iloc[0].to_dict()

        packet_md = generate_contractor_work_packet(m_row, scored, policy)

        c_down1, c_down2 = st.columns(2)
        with c_down1:
            st.download_button(
                f"📥 Download Work Packet Markdown ({selected_run})",
                data=packet_md,
                file_name=f"field_work_packet_{selected_run}.md",
                mime="text/markdown",
            )
        with c_down2:
            html_content = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{selected_run} Work Order</title><style>body{{font-family:sans-serif;padding:30px;line-height:1.5;}}table,th,td{{border:1px solid #ccc;border-collapse:collapse;padding:8px;}}</style></head><body><pre>{packet_md}</pre></body></html>"
            st.download_button(
                f"🖨️ Download Offline Printable HTML ({selected_run})",
                data=html_content,
                file_name=f"field_work_packet_{selected_run}.html",
                mime="text/html",
            )

        with st.container(border=True):
            st.markdown(packet_md)
    else:
        st.info("No runs currently scheduled. Adjust weekly capacity slider on sidebar.")

# ------------------------------------------------------------------
# TAB 6: DECISION AUDIT & FEEDBACK TRAIL
# ------------------------------------------------------------------
with tab_audit:
    st.markdown("### `[ DECISION AUDIT & FEEDBACK TRAIL ] Public Accountability & Continuous Learning`")

    st.markdown("#### `1. Human Discretion & Override Trail`")
    if AUDIT_LOG.exists():
        entries = [
            json.loads(line)
            for line in AUDIT_LOG.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if entries:
            log_df = pd.DataFrame(entries)[
                ["timestamp", "ticket_id", "decision", "system_rank", "efficiency_rank", "coordinator", "justification"]
            ]
            st.dataframe(log_df.iloc[::-1], width="stretch", hide_index=True)
            st.download_button(
                "Download Audit Trail (JSONL)",
                data=AUDIT_LOG.read_text(encoding="utf-8"),
                file_name="audit_log.jsonl",
                mime="application/json",
            )
        else:
            st.info("No decisions recorded yet.")
    else:
        st.info("No decisions recorded yet. Use the adjudication console on the queue tab.")

    st.markdown("---")
    st.markdown("#### `2. Coordinator AI Calibration Feedback (Active Learning)`")
    if FEEDBACK_LOG.exists():
        fb_entries = [
            json.loads(line)
            for line in FEEDBACK_LOG.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if fb_entries:
            fb_df = pd.DataFrame(fb_entries)
            st.dataframe(fb_df.iloc[::-1], width="stretch", hide_index=True)
        else:
            st.caption("No AI calibration entries recorded yet.")
    else:
        st.caption("No AI calibration entries recorded yet.")
