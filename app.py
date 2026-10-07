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
    score_tickets,
    tenant_answer,
)
from main import TICKET_CSV, load_tickets

AUDIT_LOG = Path(__file__).with_name("audit_log.jsonl")
TIER_LABEL = {0: "0 SAFETY", 1: "1 OVERDUE", 2: "2 SCORED"}

st.set_page_config(page_title="EquiTriage | Dialectic Console", layout="wide")

TERMINAL_CSS = """
<style>
    /* Monospace look on TEXT elements only.
       Never target bare `span` or `div` with a font-family: Streamlit draws its
       icons (upload, expander arrows, sidebar collapse) as text in an icon font,
       and overriding the font turns them into words like "keyboard_double_arrow_right". */
    html, body, .stApp,
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp li,
    .stApp input, .stApp textarea, .stApp button, .stApp [data-baseweb="select"] {
        font-family: 'Courier New', Courier, monospace;
    }

    /* Safety net: always keep icons on their own font */
    [data-testid="stIconMaterial"],
    .material-icons, .material-symbols-rounded, .material-symbols-outlined,
    [class*="material-symbols"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    }

    .stButton>button, .stDownloadButton>button {
        background-color: #1A1A1A; color: #D4D4D4; border: 1px solid #555555;
        border-radius: 0px; width: 100%; transition: all 0.2s ease-in-out; }
    .stButton>button:hover, .stDownloadButton>button:hover {
        border-color: #FFBF00; color: #FFBF00; background-color: #0F0F0F; }
    [data-testid="stMetricValue"] { color: #FFBF00; }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


# ==========================================
# MODEL (slow, loaded once) + EXTRACTION (slow, cached per ticket set)
# ==========================================
@st.cache_resource(show_spinner="Loading the Laya decision model...")
def get_router():
    from laya import Router

    return Router(preload=True)


@st.cache_data(show_spinner="Reading tickets with Laya...")
def cached_extract(tickets: pd.DataFrame) -> pd.DataFrame:
    # Only depends on the ticket data, NOT on the policy sliders,
    # so moving a slider never triggers another model call.
    return extract_signals(tickets, get_router())


def rank_arrow(shift: int) -> str:
    if shift > 0:
        return f"▲ {shift}"
    if shift < 0:
        return f"▼ {abs(shift)}"
    return "–"


# ==========================================
# DATA LOADING
# ==========================================
st.markdown("## `EQUITRIAGE // Housing Maintenance Dispatch`")
st.caption(
    "Use the included maintenance_tickets.csv or upload another CSV. "
    "Required columns: ticket_id, location, distance_km, days_waiting, text."
)
uploaded_csv = st.file_uploader("Ticket data CSV", type="csv")
csv_source = uploaded_csv if uploaded_csv is not None else TICKET_CSV
try:
    tickets_df = load_tickets(csv_source)
except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Could not load ticket data: {error}")
    st.stop()

source_signature = hashlib.sha256(tickets_df.to_csv(index=False).encode("utf-8")).hexdigest()
if st.session_state.get("source_signature") != source_signature:
    st.session_state.signals = None
    st.session_state.source_signature = source_signature
if "signals" not in st.session_state:
    st.session_state.signals = None

n_tickets = len(tickets_df)

# ==========================================
# SIDEBAR: THE POLICY THE HUMAN OWNS
# ==========================================
with st.sidebar:
    st.markdown("### `[ POLICY ]`")
    st.caption("Every judgement call is a visible dial. Move one and the whole queue re-ranks.")
    equity = st.slider(
        "Equity weight  (0 = cheapest trip first, 100 = distance-blind)", 0, 100, 50
    ) / 100
    with st.expander("Safety & guardrails", expanded=True):
        safety_floor = st.slider("Safety floor: always dispatch above this risk", 0.0, 1.0, 0.70, 0.05)
        max_wait = st.slider("Max wait guardrail (days)", 7, 120, 45)
    with st.expander("Advanced weights"):
        travel_cost = st.slider("Travel penalty per 100 km (efficiency view)", 0.0, 1.0, 0.25, 0.05)
        remote_uplift = st.slider("Remote uplift per 100 km (at 100% equity)", 0.0, 0.5, 0.10, 0.01)
        wait_pct = st.slider("Priority gain per day waiting (%)", 0.0, 5.0, 2.0, 0.5)
    st.markdown("### `[ CAPACITY ]`")
    if n_tickets > 1:
        jobs_per_week = st.slider("Jobs the crews can finish this week", 1, n_tickets, min(5, n_tickets))
    else:
        jobs_per_week = 1

policy = Policy(
    equity_weight=equity,
    safety_floor=safety_floor,
    max_wait_days=max_wait,
    travel_cost_per_100km=travel_cost,
    remote_uplift_per_100km=remote_uplift,
    wait_rate_per_day=wait_pct / 100,
)

# ==========================================
# RUN AI EXTRACTION
# ==========================================
if st.button("[ RUN LIVE LAYA TRIAGE ]", key="run_laya_triage"):
    try:
        st.session_state.signals = cached_extract(tickets_df)
    except ImportError:
        st.error("The `laya` package is not installed in this environment.")

if st.session_state.signals is None:
    st.markdown("### `[ INBOX ] Pending Dispatches`")
    st.dataframe(
        tickets_df.rename(
            columns={"ticket_id": "Ticket", "location": "Location",
                     "days_waiting": "Days Wait", "text": "Request"}
        )[["Ticket", "Location", "Days Wait", "Request"]],
        width="stretch",
        hide_index=True,
    )
    st.info("Press RUN LIVE LAYA TRIAGE to read the reports, then use the sidebar to set policy.")
    st.stop()

scored = score_tickets(st.session_state.signals, policy)

# ==========================================
# HEADLINE NUMBERS
# ==========================================
remote_mask = scored.distance_km >= policy.remote_km
m1, m2, m3, m4 = st.columns(4)
m1.metric("Tickets", len(scored))
m2.metric("Safety-tier jobs", int((scored.tier == 0).sum()))
m3.metric("Needs human review", int(scored.needs_review.sum()))
m4.metric(
    f"Remote jobs lifted by equity (≥{policy.remote_km:.0f} km)",
    int((remote_mask & (scored.rank_shift > 0)).sum()),
)

tab_queue, tab_tenant, tab_price, tab_audit = st.tabs(
    ["[ DISPATCH QUEUE ]", "[ WHY IS MY REPAIR HERE? ]", "[ PRICE OF EQUITY ]", "[ AUDIT LOG ]"]
)

# ------------------------------------------------------------------
# TAB 1: QUEUE + ADJUDICATION CONSOLE
# ------------------------------------------------------------------
with tab_queue:
    view = pd.DataFrame(
        {
            "Rank": scored["rank"],
            "Ticket": scored.ticket_id,
            "Location": scored.location,
            "Issue": scored.category.str.title(),
            "Safety": scored.safety_prob.map("{:.0%}".format),
            "Days Wait": scored.days_waiting,
            "Score": scored.final_score.round(2),
            "Cost-only Rank": scored.rank_efficiency,
            "Shift": scored.rank_shift.map(rank_arrow),
            "Tier": scored.tier.map(TIER_LABEL),
            "Review": scored.needs_review.map({True: "Yes", False: "No"}),
        }
    )
    st.markdown("### `[ QUEUE ] Equity-adjusted dispatch order`")
    st.caption("Shift ▲ = moved up compared with a cheapest-trip-first list.")
    st.dataframe(view, width="stretch", hide_index=True)
    st.download_button(
        "Download ranked queue as CSV",
        data=scored.drop(columns=["explanation"]).to_csv(index=False),
        file_name="equitriage_ranked_queue.csv",
        mime="text/csv",
    )

    st.markdown("---")
    col_left, col_right = st.columns([6, 4], gap="large")
    ticket_ids = scored.ticket_id.tolist()  # ordered by current rank

    with col_left:
        st.markdown("### `[ META-SYNTHESIZER ] Console`")
        # Selecting by ticket ID (not table row) keeps the selection correct
        # when a slider reorders the queue.
        chosen = st.selectbox("Target Ticket:", ticket_ids, key="console_ticket")
        row = scored.loc[scored.ticket_id == chosen].iloc[0]
        with st.container(border=True):
            st.text(f"Target: {row.ticket_id} - {row.location} ({row.distance_km:.0f} km)")
            st.text(f"Status: WAITING {int(row.days_waiting)} DAYS")
            st.text(f"Request: {row.text}")
        c1, c2 = st.columns(2)
        c1.metric("Cost-only rank", f"#{int(row.rank_efficiency)}")
        c2.metric("Policy rank", f"#{int(row['rank'])}", delta=int(row.rank_shift))
        with st.container(border=True):
            st.markdown("`WHY IT SITS HERE`")
            st.write(row.explanation)

    with col_right:
        st.markdown("### `[ DECISION ] Own the trade-off`")
        if row.tier == 0:
            st.warning("`SAFETY-TIER JOB: deferring this needs a strong written reason.`")
        coordinator = st.text_input("Coordinator name:", value="Coordinator")
        justification = st.text_input(
            "Enter Audit Justification:",
            placeholder="Type justification for the public record...",
        )
        b1, b2 = st.columns(2)

        def _record(decision: str, label: str) -> None:
            if not justification.strip():
                st.warning("`ERROR: Audit justification required to proceed.`")
                return
            log_decision(AUDIT_LOG, row, decision, justification, coordinator, policy)
            st.info(f"Logged: {row.ticket_id} -> {label}. Demo only: no dispatch or SMS was sent.")

        with b1:
            if st.button("[ ENFORCE EQUITY ]", key="btn_equity"):
                _record("ENFORCE_EQUITY", "equity ranking upheld")
        with b2:
            if st.button("[ ACCEPT EFFICIENCY ]", key="btn_eff"):
                _record("ACCEPT_EFFICIENCY", "efficiency ranking accepted")

# ------------------------------------------------------------------
# TAB 2: TENANT-FACING "WHY"
# ------------------------------------------------------------------
with tab_tenant:
    st.markdown("### `[ TENANT ] Why is my repair where it is?`")
    st.caption(
        "A plain-language answer a coordinator can read out or send. "
        "It is built from the real score components, so it always matches the queue."
    )
    tenant_ticket = st.selectbox("Ticket ID:", scored.ticket_id.tolist(), key="tenant_ticket")
    trow = scored.loc[scored.ticket_id == tenant_ticket].iloc[0]
    with st.container(border=True):
        st.text(tenant_answer(trow, scored, policy))

# ------------------------------------------------------------------
# TAB 3: PRICE OF EQUITY
# ------------------------------------------------------------------
with tab_price:
    st.markdown("### `[ TRADE-OFF ] What does equity cost, and who does it help?`")
    st.caption(
        f"Illustrative: if crews finish {jobs_per_week} jobs this week, "
        "what changes when we follow the equity policy instead of cheapest-trip-first?"
    )
    poe = price_of_equity(scored, jobs_per_week, policy)
    st.dataframe(poe, width="stretch")
    eff, eq = poe["Efficiency-first"], poe["Equity policy"]
    p1, p2, p3 = st.columns(3)
    p1.metric("Total travel (km)", int(eq["Total travel (km)"]),
              delta=int(eq["Total travel (km)"] - eff["Total travel (km)"]), delta_color="off")
    p2.metric("Remote jobs completed", int(eq["Remote jobs completed"]),
              delta=int(eq["Remote jobs completed"] - eff["Remote jobs completed"]), delta_color="off")
    p3.metric("Longest wait left (days)", int(eq["Longest wait still in queue (days)"]),
              delta=int(eq["Longest wait still in queue (days)"] - eff["Longest wait still in queue (days)"]),
              delta_color="off")
    st.caption("Deltas compare the equity policy against cheapest-trip-first. Travel assumes a round trip to each community visited.")

# ------------------------------------------------------------------
# TAB 4: AUDIT LOG
# ------------------------------------------------------------------
with tab_audit:
    st.markdown("### `[ AUDIT ] Human decisions on the public record`")
    if AUDIT_LOG.exists():
        entries = [json.loads(line) for line in AUDIT_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
        log_df = pd.DataFrame(entries)[
            ["timestamp", "ticket_id", "decision", "system_rank", "efficiency_rank", "coordinator", "justification"]
        ]
        st.dataframe(log_df.iloc[::-1], width="stretch", hide_index=True)
        st.download_button(
            "Download audit log (JSONL)",
            data=AUDIT_LOG.read_text(encoding="utf-8"),
            file_name="audit_log.jsonl",
            mime="application/json",
        )
    else:
        st.info("No decisions recorded yet. Use the console on the queue tab.")
