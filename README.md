# EquiTriage: Equity-Aware Housing Maintenance Dispatch System

## Overview

EquiTriage is a decision-support system for housing maintenance dispatch that combines AI-powered signal extraction with explicit equity-aware policy controls. The system helps maintenance coordinators prioritize repair tickets while making equity trade-offs visible and controllable.

## Key Features

- **AI Signal Extraction**: Uses the Laya model to automatically categorize maintenance requests (plumbing, electrical, structural, pest, other) and assess safety hazards and deterioration risks from free-text descriptions
- **Equity-Aware Prioritization**: Balances efficiency (cheapest trip first) with equity considerations (distance, wait times) through adjustable policy sliders
- **Transparent Decision Making**: Every ranking decision is explainable, showing exactly how equity adjustments affect the queue
- **Human-in-the-Loop Review**: Flags ambiguous cases for human review when the AI is uncertain
- **Audit Trail**: Logs all human decisions with justifications for accountability and transparency
- **Price of Equity Analysis**: Quantifies the trade-off between efficiency and equity in concrete terms (travel distance, jobs completed, etc.)

## How It Works

1. **Signal Extraction (Slow, Cached)**:
   - The Laya AI model processes each ticket text to extract:
     - Category (plumbing, electrical, structural, pest, other)
     - Safety hazard probability (0.0-1.0)
     - Deterioration risk score (0-3)

2. **Deterministic Scoring (Instant, Auditable)**:
   - Base urgency combines safety and deterioration signals
   - Equity multiplier adds priority for wait times and distance
   - Final score = base urgency × equity multiplier
   - Two rankings always computed:
     - Efficiency-only (what "cheapest trip first" would do)
     - Policy ranking (with equity adjustments applied)

3. **Policy Controls (Visible Sliders)**:
   - Equity weight (0% = efficiency-only, 100% = distance-blind)
   - Safety floor (always dispatch above this risk level)
   - Maximum wait guardrail
   - Travel cost and remote uplift adjustments
   - Waiting priority gain per day

## Files in This Repository

- app.py - Streamlit web interface for interactive triage
- equitriage_engine.py - Core logic for signal extraction, scoring, and policy application
- main.py - Command-line version that demonstrates the core algorithm
- maintenance_tickets.csv - Sample maintenance ticket data
- maintenance_tickets_large.csv - Larger sample dataset
- audit_log.jsonl - Generated log of human decisions (created when used)

## Installation

1. Clone this repository
2. Create a virtual environment (optional but recommended):
   `python -m venv venv
.\venv\Scripts\activate`
3. Install dependencies:

   pip install streamlit pandas numpy
   `

4. Install the Laya model (this appears to be a proprietary model - check with your provider for installation instructions)

## Usage

### Web Interface (Recommended)

`streamlit run app.py`

### Command Line

`python main.py`

## Interpreting the Results

The system shows:

- **Rank**: Current position in the equity-adjusted queue
- **Ticket**: Ticket ID
- **Location**: Community/location
- **Issue**: Categorized problem type
- **Safety**: Safety probability as percentage
- **Days Wait**: How long the ticket has been waiting
- **Score**: Final EquiTriage score
- **Cost-only Rank**: Where efficiency-only would place it
- **Shift**: How much equity rules moved the ticket (▲ = moved up, ▼ = moved down)
- **Tier**: Safety tier (0 = safety override, 1 = ageing guardrail, 2 = normal scoring)
- **Review**: Whether flagged for human review

## Equity Trade-Off Visualization

The "Price of Equity" tab shows concretely what choosing equity over pure efficiency means:

- How much extra travel distance is incurred
- How many more remote jobs get completed
- How communities visited change
- Waiting times for remaining jobs

## Audit System

All human decisions are logged to udit_log.jsonl with:

- Timestamp
- Ticket ID
- System rank vs efficiency rank
- Decision made (ENFORCE_EQUITY or ACCEPT_EFFICIENCY)
- Coordinator name
- Written justification
- System explanation
- Policy settings used

## Customization

Adjust the policy sliders in the sidebar to see how different equity preferences affect:

- The dispatch queue
- Which tickets move up/down
- The price of equity metrics
- Explanations for each ticket's position

## Data Requirements

The system expects CSV files with these columns:

-     Ticket_id: Unique identifier
- location: Community/location name
- distance_km: Distance from depot (kilometers)
- days_waiting: How many days the ticket has been waiting
-     Text: Free-text description of the maintenance issue

## Design Principles

1. **AI Only Extracts Signals**: The AI model is used solely for extracting structured data from free text, not for making final prioritization decisions
2. **Deterministic After Extraction**: All scoring and ranking after signal extraction is deterministic math, making it auditable and controllable
3. **Visible Trade-Offs**: Equity considerations are implemented as visible, adjustable parameters rather than hidden biases
4. **Explainable Decisions**: Every ranking position can be explained in terms of the underlying components
5. **Human Ownership**: The system forces humans to own equity trade-offs through explicit controls and required justifications

## Future Enhancements

- Integration with actual dispatch systems
- Multilingual support for tenant communications
- Predictive modeling for emergent issues
- Mobile app for field workers
- Advanced analytics dashboard for housing managers

---

\*EquiTriage makes equity in public service delivery visible, controllable, and accountable."
