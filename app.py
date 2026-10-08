import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from equitriage_engine import (
    Policy,
    extract_signals,
    log_decision,
    price_of_equity,
    compute_manifest_breakdown,
    score_tickets,
    tenant_answer,
    load_history_ledger,
    retrieve_property_history,
)
import importlib
import geo_utils
importlib.reload(geo_utils)
from geo_utils import (
    build_geospatial_dataframe,
    build_pydeck_chart,
    build_folium_map,
    DARWIN_DEPOT,
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

    /* Fix Pydeck 3D Deck Hover Tooltip Layout, Typography & Box Model */
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
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
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
    return Router(preload=True)


@st.cache_data(show_spinner="Reading tickets with Laya & retrieving asset history...")
def cached_extract(tickets: pd.DataFrame) -> pd.DataFrame:
    return extract_signals(tickets, get_router())


def load_cached_cyclone_signals() -> list[dict]:
    """Load pre-extracted Laya signals for cyclone tickets to ensure instantaneous slider response."""
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
    st.markdown("### `[ EXTREME WEATHER STRESS TEST ]`")
    st.caption("Stress-test system resilience against extreme monsoonal & cyclone surges.")
    cyclone_surge_count = st.slider(
        "Cyclone & Monsoon Surge Simulation (+tickets)",
        min_value=0,
        max_value=50,
        value=0,
        step=5,
        help="Programmatically injects authentic structural/water storm damage tickets across coastal & Top End NT.",
    )

    st.markdown("---")
    st.markdown("### `[ DISPATCH POLICY DIALS ]`")
    st.caption("Human governance dials. Adjusting any dial recalculates rankings instantly.")
    equity = st.slider(
        "Equity Weight (0% = Lowest Fuel Cost, 100% = Equal Remote Priority)", 0, 100, 55
    ) / 100

    with st.expander("Critical Safety & Wait-Time Guardrails", expanded=True):
        safety_floor = st.slider("Critical Safety Override (Immediate dispatch threshold)", 0.0, 1.0, 0.70, 0.05)
        max_wait = st.slider("Max Wait Time Guardrail (days)", 7, 120, 45)

    with st.expander("Property History & Trip Bundling Controls", expanded=True):
        chronic_boost = st.slider("Repeat Failure Urgency Boost (%)", 0, 100, 50, 10) / 100
        chronic_threshold = st.slider("Repeat Issue Detection Confidence", 0.40, 0.90, 0.60, 0.05)
        fleet_km_cost = st.slider("Fleet Contractor Operating Cost ($/km)", 1.00, 3.00, 1.50, 0.25)

    with st.expander("Advanced Cost & Remote Weights"):
        travel_cost = st.slider("Distance Penalty per 100 km (Cheapest-trip view)", 0.0, 1.0, 0.25, 0.05)
        remote_uplift = st.slider("Remote Uplift per 100 km (At 100% equity)", 0.0, 0.5, 0.10, 0.01)
        wait_pct = st.slider("Priority Gain per Day Waiting (%)", 0.0, 5.0, 2.0, 0.5)

    # Dynamic capacity based on total available tickets
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
)

# ==========================================
# DATA INTAKE & LIVE EXTRACTION MANAGEMENT
# ==========================================
st.markdown("## `EQUITRIAGE // Public Housing Logistics & Decision Engine`")
st.caption("Autonomous triage engine combining Laya AI, context-aware RAG, geospatial routing, and intelligent trip bundling.")

# Track data signature to avoid unnecessary re-extraction
tickets_df = raw_tickets_df.copy()
source_sig = hashlib.sha256(tickets_df.to_csv(index=False).encode("utf-8")).hexdigest()
if st.session_state.get("source_sig") != source_sig:
    st.session_state.base_signals = None
    st.session_state.source_sig = source_sig

col_btn, col_info = st.columns([3, 7])
with col_btn:
    run_btn = st.button("[ RUN LIVE LAYA TRIAGE ]", key="run_triage_btn")
    if run_btn:
        try:
            st.session_state.base_signals = cached_extract(tickets_df)
        except ImportError:
            st.error("The `laya` package is not installed in this environment.")
        except Exception as e:
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
base_sig = st.session_state.get("base_signals")
if base_sig is None or not isinstance(base_sig, pd.DataFrame) or base_sig.empty:
    st.stop()

signals_combined = base_sig.copy()

if cyclone_surge_count > 0:
    cached_cyclone = load_cached_cyclone_signals()
    if cached_cyclone:
        storm_subset = pd.DataFrame(cached_cyclone[:cyclone_surge_count])
        # Ensure compatible columns
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
remote_mask = scored.distance_km >= policy.remote_km
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
tab_map, tab_queue, tab_tenant, tab_price, tab_audit = st.tabs(
    [
        "🗺️ [ ROUTE & DISPATCH MAP ]",
        "📋 [ PRIORITY WORK ORDERS ]",
        "🗣️ [ TENANT TRANSPARENCY PORTAL ]",
        "⚖️ [ FAIRNESS VS. COST TRADEOFF ]",
        "📜 [ DECISION AUDIT TRAIL ]",
    ]
)

# ------------------------------------------------------------------
# TAB 1: INTERACTIVE ROUTE & DISPATCH MAP
# ------------------------------------------------------------------
with tab_map:
    st.markdown("### `[ ROUTE & DISPATCH MAP ] Visualizing the Regional Access & Equity Gap`")
    st.caption(
        "Public housing maintenance is inherently spatial. Compare the cheapest-trip-first route "
        "(clustering around Darwin) against the EquiTriage route (reaching remote outback communities like Wadeye and Tennant Creek)."
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

    geo_df = build_geospatial_dataframe(scored, jobs_per_week)

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
    g1.metric("Efficiency Furthest Reach", f"{eff_top.distance_km.max():.0f} km", help="Maximum distance visited under cheapest-trip-first")
    g2.metric("EquiTriage Furthest Reach", f"{eq_top.distance_km.max():.0f} km", help="Maximum distance visited under equity-aware dispatch")
    g3.metric(
        "Remote Communities Visited",
        f"{int((eq_top.distance_km >= policy.remote_km).sum())} vs {int((eff_top.distance_km >= policy.remote_km).sum())}",
        help="Equity vs Efficiency visits to remote communities (≥100 km)",
    )

    st.markdown(
        """
        <div style="background-color: #111; padding: 12px; border: 1px solid #333; margin-top: 10px; font-size: 13px;">
            <b style="color: #FFBF00;">GEOSPATIAL LEGEND:</b>
            <span style="color: #2ECC71; margin-left: 15px;">● Green: &lt;14 days</span>
            <span style="color: #FA8231; margin-left: 15px;">● Amber: 14-45 days</span>
            <span style="color: #EB3B5A; margin-left: 15px;">● Glowing Red: &gt;45 days (SLA Breached)</span>
            <span style="color: #00B4D8; margin-left: 20px;">━ Blue Arc: Efficiency Route (Urban Clustered)</span>
            <span style="color: #FFBF00; margin-left: 15px;">━ Gold Arc: EquiTriage Route (Territory-Wide)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# TAB 2: PRIORITY WORK ORDERS + CONSOLE
# ------------------------------------------------------------------
with tab_queue:
    st.markdown("### `[ PRIORITY WORK ORDERS ] Scheduled Dispatch Order`")
    st.caption("Shift ▲ = moved up compared with a cheapest-trip-first list.")

    # Format queue view
    display_df = pd.DataFrame(
        {
            "Rank": scored["rank"],
            "Ticket": scored["ticket_id"],
            "Property": scored.get("property_id", scored["ticket_id"]),
            "Location": scored["location"],
            "Issue": scored["category"].str.title(),
            "Safety": scored["safety_prob"].map("{:.0%}".format),
            "Days Wait": scored["days_waiting"],
            "Chronic?": scored["is_chronic"].map({True: "⚡ Repeat Risk", False: "—"}),
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
            st.markdown(f"**Target:** `{row.ticket_id}` | Property: `{row.get('property_id', row.ticket_id)}` | **{row.location}** ({row.distance_km:.0f} km)")
            st.markdown(f"**Status:** Waiting {int(row.days_waiting)} days | **Category:** {row.category.title()}")
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
        st.markdown("### `[ HUMAN DECISION ] Public Accountability`")
        if row.tier == 0:
            st.warning("⚠️ `SAFETY OVERRIDE ACTIVE: Deferring this requires mandatory written justification.`")
        coordinator = st.text_input("Coordinator ID:", value="Fleet Coordinator NT")
        justification = st.text_input(
            "Enter Audit Justification:",
            placeholder="Document legal, safety, or logistical rationale...",
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
        "Observe how Smart Job Bundling recovers travel efficiency when remote runs are scheduled."
    )

    poe = price_of_equity(scored, jobs_per_week, policy)
    st.dataframe(poe, width="stretch")

    eff, eq = poe["Efficiency-first"], poe["Equity policy"]

    p1, p2, p3, p4 = st.columns(4)
    p1.metric(
        "Direct Fleet Travel",
        f"{eq['Direct fleet travel (km)']} km",
        delta=f"{eq['Direct fleet travel (km)'] - eff['Direct fleet travel (km)']} km",
        delta_color="off",
    )
    p2.metric(
        "Travel Saved via Bundling",
        f"{eq['Travel avoided via bundling (km)']} km",
        help="Avoided standalone future round-trips by piggybacking lower-tier jobs onto this week's runs",
    )
    p3.metric(
        "Fleet Cost Recovered",
        f"{eq['Fleet cost saved via bundling ($)']}",
        help=f"Calculated at ${policy.fleet_cost_per_km:.2f}/km NT fleet operating rate",
    )
    p4.metric(
        "Effective Travel Per Job",
        f"{eq['Effective travel per job (km/job)']} km",
        delta=f"{eq['Effective travel per job (km/job)'] - eff['Effective travel per job (km/job)']:.1f} km",
        delta_color="off",
    )

    st.markdown("---")
    st.markdown("### `[ DISPATCH MANIFEST ] Same-Community Bundled Jobs on Scheduled Runs`")
    st.caption(
        "When an urgent Tier 0 or high-equity job triggers a remote trip (e.g., 420 km to Wadeye), "
        "the algorithm appends all overdue and lower-tier jobs at that exact community. Marginal travel = 0 km."
    )
    manifest_df = compute_manifest_breakdown(scored, jobs_per_week, policy)
    st.dataframe(manifest_df, width="stretch", hide_index=True)

# ------------------------------------------------------------------
# TAB 5: DECISION AUDIT TRAIL
# ------------------------------------------------------------------
with tab_audit:
    st.markdown("### `[ DECISION AUDIT TRAIL ] Immutable Record of Human Discretion & Policy Compliance`")
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
