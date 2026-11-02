"""Tunable constants for the live engine.

Everything a demo operator or a clinician might want to argue about lives here
as a named constant. Nothing downstream should contain a bare number.
"""

from __future__ import annotations

import os
import pathlib

# --------------------------------------------------------------------------
# Replay clock
# --------------------------------------------------------------------------

# Wall-clock seconds between ticks.
TICK_SECONDS = 1.0

# Simulated minutes per wall-clock minute. At 60x, one wall second is one
# simulated minute, so the 5-minute charting grid advances every 5 ticks and an
# hour of ED time passes in a minute.
DEFAULT_SPEED = 60.0
MIN_SPEED = 1.0
MAX_SPEED = 3600.0

# A stay leaves the census this many simulated minutes after its last charted
# observation. Real extracts have ragged tails; this stops a patient lingering
# on the board forever.
DEPART_GRACE_MIN = 15.0

# --------------------------------------------------------------------------
# Default cohort (simulator). Ignored when a MIMIC source is wired in.
# --------------------------------------------------------------------------

DEMO_PATIENTS = int(os.environ.get("WATCHLIST_PATIENTS", 400))
DEMO_SEED = int(os.environ.get("WATCHLIST_SEED", 11))

# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

# Browser origins allowed to call the API. Defaults to the usual local dev
# servers; set WATCHLIST_CORS_ORIGINS to a comma-separated list to override,
# or to "*" if you knowingly want it open. Wide-open CORS was the old default
# and is not one you want to leave on anything reachable.
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "WATCHLIST_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

MODEL_PATH = pathlib.Path(os.environ.get("WATCHLIST_MODEL", "ml/deterioration.json"))
META_PATH = pathlib.Path(os.environ.get("WATCHLIST_MODEL_META", "ml/deterioration_meta.json"))

# Admission model ("who will need a bed"). Optional: if it has not been
# trained the board simply omits the bed columns.
ADMISSION_MODEL_PATH = pathlib.Path(
    os.environ.get("WATCHLIST_ADMISSION_MODEL", "ml/admission.json")
)
ADMISSION_META_PATH = pathlib.Path(
    os.environ.get("WATCHLIST_ADMISSION_META", "ml/admission_meta.json")
)

# Points of score history kept per patient for the detail sparkline.
RISK_HISTORY_POINTS = 288

# Lookback for the board's trend arrow, in simulated minutes.
TREND_LOOKBACK_MIN = 30.0

# How many feature contributions the explanation panel returns.
EXPLAIN_TOP_K = 8

# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------
#
# How many plain-language reasons accompany every score. These are ranked by
# CHANNEL, not by column: the three largest raw contributions to an alert are
# routinely rr_d15, rr_d30 and rr_slope30, which is one clinical fact printed
# three times. api.reasons sums contributions within a channel first, so three
# reasons mean three different things about the patient.
#
# Three is a deliberate ceiling, not a default. A nurse walking a board reads
# the first line and maybe the second.
REASON_TOP_K = 3

# TreeSHAP costs ~60x what a prediction does (85 ms for a 400-patient census
# against 1.4 ms to score it), so reasons are computed in the tick loop and
# memoised -- never on the request path. See ReplayEngine._explain_active.
#
# The memo key is the patient's grid index, because that is exactly what the
# feature vector depends on: vitals arrive on the GRID_MIN charting grid, so
# between grid advances the feature vector is byte-identical and so is its
# explanation. Recomputing it every tick would be recomputing the same answer.
# Set false to explain only on request (saves the tick cost, adds ~3 ms and a
# round trip to every panel open).
EXPLAIN_IN_TICK = os.environ.get("WATCHLIST_EXPLAIN_IN_TICK", "1") != "0"

# Ceiling on how many patients get explained in one tick. The census is a few
# hundred; this only bites if someone points the engine at a much larger
# cohort, where it degrades by explaining the sickest patients first rather
# than by blowing the tick budget.
EXPLAIN_MAX_PER_TICK = int(os.environ.get("WATCHLIST_EXPLAIN_MAX_PER_TICK", 600))

# Nominal treatment-space capacity, for the bed board's occupancy figure.
# The data carries no bed inventory, so this is a declared denominator, not
# something observed.
ED_CAPACITY = int(os.environ.get("WATCHLIST_ED_CAPACITY", 30))

# --------------------------------------------------------------------------
# Current instability (NEWS2)
# --------------------------------------------------------------------------
#
# Standard NEWS2 escalation bands. These are not ours to invent -- 5 is the
# usual "urgent review" trigger and 7 the "emergency response" trigger.
#
# They do real work on this board beyond display. The deterioration model is
# trained only on pre-onset rows, so a patient who has already deteriorated is
# outside its training population and its score there is extrapolation. We
# cannot see onset live, so NEWS2 >= NEWS2_HIGH stands in for "already
# deteriorated": above it the board reports state from the vitals rather than
# leaning on a forward-looking score that no longer applies.
NEWS2_MEDIUM = 5
NEWS2_HIGH = 7

# --------------------------------------------------------------------------
# Manual control: overrides, acknowledgements, scenarios
# --------------------------------------------------------------------------
#
# Everything here is a human acting on the board. Three rules shape it:
#
# 1. A human decision outranks the model and is never silently dropped. The
#    score keeps updating underneath and stays visible, but it does not erase
#    what a person decided.
# 2. Every manual act is recorded -- who, what, when, why -- and the record
#    outlives the flag it was about.
# 3. Manual input reaches the board through the same scoring code as the
#    replay clock. There is no demo mode.

# How far the risk score must climb ABOVE its value at acknowledgement before
# an acknowledged re-triage flag escalates again.
#
# This is the alert-fatigue knob and it cuts both ways. Too low and the
# acknowledge button is theatre -- the flag comes straight back and the nurse
# learns to ignore it. Too high and a patient who is genuinely falling apart
# stays quietly acknowledged. 0.15 on a 0-1 risk scale is roughly one band on
# IMPLIED_ESI_BANDS: enough that the situation has materially changed, not so
# much that it needs a catastrophe.
ACK_REESCALATE_RISK_DELTA = 0.15

# ...and the same rule expressed as a share of the headroom left to 1.0.
#
# The absolute delta alone has a hole in it: acknowledge a patient at 0.95 and
# it asks for 1.10, which cannot happen, so the flag could never re-raise no
# matter how far the patient fell. That is exactly the failure acknowledgement
# is supposed to prevent, and it bites hardest on the sickest patients.
#
# So the trigger is whichever of the two comes FIRST: climb by the absolute
# delta, or cover this fraction of the distance from here to certainty. At 0.50
# the absolute rule governs (0.65 before 0.75); at 0.95 the headroom rule does
# (0.975 before 1.10). Both are explainable at a whiteboard, and the threshold
# is always strictly below 1.0, so it is always reachable.
ACK_REESCALATE_HEADROOM_FRAC = 0.5

# Acknowledgement is per-flag, not per-patient: if a flag clears and later
# raises again, that is a new clinical event and needs its own look.
ACK_CLEARS_WITH_FLAG = True

# Free-text reason required to override. A priority change nobody can explain
# is indistinguishable from a misclick when it is reviewed later.
OVERRIDE_REQUIRES_REASON = True
OVERRIDE_REASON_MIN_CHARS = 3
OVERRIDE_REASON_MAX_CHARS = 500

# Who the board attributes a manual action to when the client does not say.
#
# There is NO AUTHENTICATION here. The actor is a string the client supplies,
# so this records a claim, not an identity. It is honest demo provenance and
# it is not an audit trail in the regulatory sense -- wire it to real sessions
# before anyone relies on it.
DEFAULT_ACTOR = "charge nurse"

# Manual actions retained in memory for the activity log.
AUDIT_LOG_SIZE = 500

# How long a manually-charted patient is held on the board after their last
# entry, in simulated minutes.
#
# Hand-charting takes the patient off their replay trajectory (see
# ReplayEngine.append_observation), so without this they would hit
# DEPART_GRACE_MIN and vanish seconds after the first manual entry. Each new
# entry extends the hold again.
MANUAL_CONTROL_HOLD_MIN = 60.0

# Ceiling on manually injected scenario patients, so a demo cannot fill the
# board by leaning on a button.
MAX_INJECTED_PATIENTS = int(os.environ.get("WATCHLIST_MAX_INJECTED", 40))

# --------------------------------------------------------------------------
# Re-triage logic
# --------------------------------------------------------------------------
#
# The board answers "does this patient's risk still match the acuity they were
# assigned at triage?". To do that we map the model's risk score onto an
# *implied* ESI level and compare it with the assigned one.
#
# These band edges are the calibration knob. They are NOT derived from the
# model -- they are a policy choice about how much risk corresponds to how much
# acuity, and they should be re-fit against the score distribution of whatever
# cohort you deploy on (see ml/deterioration_meta.json for the training-time
# operating point). Lower the edges to flag more aggressively.
#
# Read as: risk >= edge implies that acuity level.
IMPLIED_ESI_BANDS: tuple[tuple[float, int], ...] = (
    (0.80, 1),
    (0.55, 2),
    (0.30, 3),
    (0.12, 4),
    (0.00, 5),
)

# How many ESI levels sicker the implied acuity must be before we raise a flag.
#
# 2, not 1. Inter-rater agreement on ESI is only moderate, so a one-level
# disagreement between the model and the triage nurse is inside the noise of
# triage itself -- flagging it put a third of the board in red, and a board
# where everything is flagged says nothing. Two levels is a claim worth
# interrupting someone for: an ESI 4 behaving like an ESI 2.
# Drop to 1 to flag aggressively; raise to 3 for egregious cases only.
RETRIAGE_ESI_GAP = 2

# Absolute floor. Never raise a re-triage flag below this risk, however large
# the gap -- an ESI 5 at 0.13 risk is implied-ESI-4, but nobody wants that alert.
RETRIAGE_MIN_RISK = 0.30

# The flag must hold for this long before it is raised, and must stay clear
# for this long before it drops. Stops a single noisy blood pressure from
# flapping the board.
#
# Measured in SIMULATED minutes, not ticks: a tick is a different amount of
# patient time at 60x than at 600x, and debounce that changes meaning with
# replay speed would make the demo behave differently from production.
RETRIAGE_SUSTAIN_MIN = 5.0
RETRIAGE_CLEAR_MIN = 10.0

# Over-triage (assigned sicker than the model implies) is real but is not what
# this product is for, and surfacing it invites the nurse to de-escalate on a
# model's say-so. Off by default; the direction is still reported per patient.
FLAG_OVER_TRIAGE = False
