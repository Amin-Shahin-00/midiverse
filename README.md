# Watchlist — ED triage support

Continuously re-scores waiting ED patients and tells the charge nurse who is
deteriorating, who should be re-triaged, and who will need a bed.

Runs on either a synthetic simulator or a real MIMIC-IV extract. The feature
and model code is identical on both paths — the loader re-emits real
observations under the canonical itemids `ml/features.py` already understands,
so nothing downstream knows the data changed.

See **[docs/MIMIC.md](docs/MIMIC.md)** for the itemid catalogue, the unit and
range handling, and the deterioration label definition.

## Setup

```bash
cd watchlist
python -m pip install -r requirements.txt
```

Python 3.11+. Developed on 3.14 with numpy 2.4, pandas 3.0, xgboost 3.2,
scikit-learn 1.8.

## Run

Trained models are included (`ml/deterioration.json`, `ml/admission.json`), so
the API runs without training first.

```bash
python -m uvicorn api.main:app --reload     # uvicorn.exe may not be on PATH
```

Then, in a second terminal, the dashboard:

```bash
cd web
npm install
npm run dev          # http://localhost:5173
```

Vite proxies `/api` and `/ws` to the backend on :8000, so the browser stays on
one origin and the CORS allowlist is not involved in development.

The API alone is at `http://127.0.0.1:8000/api/board`, or `/docs` for the
OpenAPI UI.

To retrain (deterministic given the seed; ~3 min and ~4 min respectively):

```bash
python -m ml.train              # deterioration model + full eval report
python -m ml.train_admission    # admission model + earliness + leakage ablation
python -m sim.simulator -n 500 -o data/   # optional: dump a cohort to CSV
```

## Running on MIMIC-IV

Needs both **MIMIC-IV** and **MIMIC-IV-ED** — two separate credentialed
downloads on the same PhysioNet account. The ED module supplies the stay
boundaries and the triage acuity that `esi` means.

```bash
python -m ml.mimic_items --verify /data/mimic-iv    # check itemids first
python -m ml.train --source mimic --mimic-root /data/mimic-iv
python -m ml.train_admission --source mimic --mimic-root /data/mimic-iv

# serve the board off the real extract
WATCHLIST_MODEL=ml/deterioration_mimic.json python -m uvicorn api.main:app
```

A MIMIC run writes `ml/deterioration_mimic.json`, never over the shipped
simulator-trained model. Use `--mimic-limit 2000` while iterating, and
`--outcome` to change which events define deterioration. Every run prints its
cohort provenance, a vitals QC report, and the label breakdown before it
trains — all three belong in anything you present.

`docs/MIMIC.md` covers the itemids, the unit traps, the range filters, and what
the label does and does not capture.

## Layout

| Path | Role |
|---|---|
| `sim/simulator.py` | Synthetic ED cohort → MIMIC-IV `chartevents` + ground truth |
| `ml/features.py` | chartevents → feature vectors; NEWS2; both feature paths |
| `ml/train.py` | Deterioration model (onset within 120 min) |
| `ml/train_admission.py` | Admission model (who needs a bed) |
| `ml/cohort.py` | `--source sim` / `--source mimic` for both trainers |
| `ml/mimic_items.py` | Itemid catalogue, units, plausible ranges (`--verify`) |
| `ml/outcomes.py` | Deterioration label from real outcomes |
| `api/mimic.py` | MIMIC-IV loader — cohort, vitals QC, labels |
| `api/mimic_io.py` | Table discovery and chunked reading of chartevents |
| `api/mimic_fixture.py` | Synthetic MIMIC tree + loader selftest |
| `api/config.py` | Every tunable constant |
| `api/sources.py` | `CohortSource` seam — simulator today, MIMIC later |
| `api/scoring.py` | Model loading, batch scoring, re-triage + clinical state |
| `api/reasons.py` | SHAP contributions → clinical sentences (`python -m api.reasons`) |
| `api/state.py` | In-memory census |
| `api/interventions.py` | Override / acknowledgement records and their rules |
| `api/scenarios.py` | Three scripted demo patients, emitted as chartevents |
| `api/manual_selftest.py` | End-to-end checks for the manual-control features |
| `api/engine.py` | Replay clock and per-tick loop |
| `api/main.py` | REST + WebSocket |
| `web/` | React dashboard (Vite) — live board, detail panel, bed board |
| `web/src/lib/i18n.jsx` | Interface language (English / Arabic) and writing direction |

## API

- `GET /api/board` — full board snapshot (same payload the WebSocket pushes)
- `GET /api/patients/{subject_id}` — detail incl. risk history and the top 3 reasons for both scores
- `GET /api/patients/{subject_id}/vitals` — charted vitals up to sim time
- `GET|POST /api/clock` — replay speed and pause
- `GET /api/health`
- `GET /api/patients/{subject_id}/explain` — reasons plus per-feature TreeSHAP contributions
- `POST /api/clock/seek` — jump to a time, or `{"reset": true}`
- `WS /ws/board` — full board pushed every tick

Manual control (each returns the new board and broadcasts it to every client):

- `POST /api/patients/{subject_id}/vitals` — chart an observation by hand
- `GET|POST /api/scenarios[/{key}]` — list presets / inject one
- `POST|DELETE /api/patients/{subject_id}/override` — set / withdraw a priority override
- `POST /api/patients/{subject_id}/acknowledge` — acknowledge a re-triage flag
- `GET /api/audit` — manual actions: who, what, when, why
- `GET /api/languages` — languages the reason text is available in

`GET /api/patients/{id}` and `/explain` take `?lang=` (`en`, `ar`) to choose the
language of the reason sentences. An unknown language falls back to English
rather than failing.

## Dashboard notes

Designed to be read from ~3 m in poor light. Risk is encoded three ways at
once — a glyph (▲ ◆ ● –), a word (CRITICAL / HIGH / ELEVATED / LOW) and a
colour — because ED lighting is bad and colour vision is not universal. The
status palette fails CVD separation as a categorical set by design (the skill's
own validator measures warning↔serious at ΔE 13.6), which is exactly why no
signal on this board rests on hue alone.

Re-triage flags fill a solid red rail down the left of the row, outline the ESI
badge, and pulse (suppressed under `prefers-reduced-motion`).

## Manual control

A nurse can act on the board, and everything they do goes through the engine's
own scoring pass. There is no demo mode: `ReplayEngine.rescore()` calls the
same `_score_active()` + `_explain_active()` pair the tick loop calls, so a
hand-charted patient and a replayed one cannot diverge. Each action returns the
board the engine produced and broadcasts it, so a manual change arrives on
every screen exactly like a tick.

**Chart vitals by hand.** Select a patient, enter any subset of hr/rr/spo2/
sbp/dbp/temp. The observation lands in the grid cell covering the current
simulated time — the newest cell the scorer reads — so it takes effect on the
next pass, including while the clock is paused. Omitted vitals carry forward.
Hand-charting takes the patient *off their replay trajectory*: rows after the
current cell are dropped, because a pre-generated future is not their future
once a human is driving. The stay is then held open for
`MANUAL_CONTROL_HOLD_MIN` so they do not depart for want of observations.

To build a trajectory by hand, let the clock run — two entries with no
simulated time between them land in the same five-minute cell, and the second
replaces the first.

**Scenario presets.** One click, no form. `septic_elderly` (ESI 2,
decompensates over ~45 min), `silent_deteriorator` (ESI 4, every opening value
inside its normal range, drifts for 40 minutes until the model implies ESI 2 —
this is the product in one patient), and `stable` (ESI 3, the control, so the
board can be seen *not* reacting). Each emits ordinary chartevents and goes
through `chartevents_to_wide`, so from the moment it lands nothing downstream
can tell it from a replayed patient. Injected patients do not survive a seek,
which rebuilds the census from the source cohort.

**Priority override.** Raise or lower a patient, with a required free-text
reason. The record carries who, what, when and why, and the assigned ESI stays
on the row beside the override — "the model says 0.71 and the nurse says ESI 2"
is more useful than either alone. An override moves the patient by *band*
(escalated above the automatic ranking, de-escalated below it) rather than
editing their score, so the number on the row is always the model's real one.
**No re-score clears an override**, and the re-triage comparison switches to
the effective acuity, so the board stops flagging a mismatch the human has
already resolved.

**Acknowledgement.** A raised re-triage flag can be acknowledged: it keeps the
flag, the red rail and the record, and loses the fill and the pulse. The score
keeps updating underneath. It re-raises on its own when risk reaches whichever
comes first — a climb of `ACK_REESCALATE_RISK_DELTA` (0.15), or half the
remaining distance to 1.0. The second rule is not decoration: a flag
acknowledged at 0.95 would need 1.10 under the fixed rule alone, so it could
never re-raise, which is exactly the alert-fatigue failure acknowledgement is
meant to prevent. The panel shows the patient's actual re-escalation threshold.

Acknowledgement is per-flag: if a flag clears and later raises again, that is a
new clinical event and needs its own look.

**Attribution is a claim, not an identity.** There is no authentication. The
actor is a string the client supplies, so `/api/audit` is honest demo
provenance, not a regulatory audit trail. Wire it to real sessions before
anyone relies on it.

## Languages

The board runs in English or Arabic, switched from the header and remembered
per browser. Arabic sets `dir="rtl"` on the document root and the layout
mirrors — the risk rail moves to the right edge of each row, the board columns
reverse, and the detail panel's accent borders flip.

**The reason sentences are translated on the server, not in the browser.**
`api/reasons.py` builds them from a structure — channel, kind of measurement,
window, value, direction — and only the last step turns that into words, so a
language is a table of strings rather than a second copy of the logic. The
thresholds that decide *which* sentence applies (the noise floor below which
movement is not a trend, the normal range that makes a value "high") are
shared, so a reading can never be a rising trend in one language and steady in
the other. `python -m api.reasons` asserts that parity directly, including that
the digits in both renderings are identical.

Two things deliberately do not change with language:

- **Numbers stay Western.** Arabic-Indic numerals are typographically correct,
  but monitors, charts and pumps across the region display Western digits, and
  a board read at three metres is the wrong place to make someone translate a
  figure they are about to act on.
- **Charts keep time running left to right.** Mirroring the x-axis would make a
  falling trace look like a rising one.

Clinical abbreviations that are used untranslated in practice — ESI, NEWS2 —
stay as they are. Free text a nurse types (an override reason, their name) is
stored and displayed verbatim in either interface language; the screenshots
confirm Arabic names and reasons render correctly inside the English layout
and vice versa.

## Why a score is what it is

Every score carries the three channels driving it, written the way a nurse
would say them:

```
1  Respiratory rate rising 4/min over the last 30 minutes      ▲ raises risk
2  Shock index rising 0.08 over the last hour                  ▲ raises risk
3  SpO2 98%                                                    ▼ lowers risk
```

They lead the detail panel, above the charts. The per-feature TreeSHAP bars
are still there, one scroll down, as the audit trail.

The bed call gets its own set, from the admission model rather than borrowed
from the deterioration one — they are different boosters and a patient can be
heading for a bed for reasons unrelated to why they are deteriorating. In
practice the bed call leans on `minutes_elapsed` and ESI, which is the leakage
`ml/train_admission.py` already ablates; the reasons make that visible on the
patient rather than only in the training report.

**Ranked by channel, not by column.** The three largest raw contributions to
an alert are routinely `rr_d15`, `rr_d30` and `rr_slope30` — one clinical fact
stated three times. SHAP contributions are additive, so `api/reasons.py` sums
them within a channel and ranks those totals. Three reasons therefore mean
three different things about the patient. Each reason quotes the strongest
column that agrees in sign with its channel total, so the fact on screen is
never labelled with a direction it does not push.

**Both directions are shown.** Ranking is by absolute contribution, so a
reason may be something holding the score *down*. "SpO2 98%, lowers risk" is a
real part of why an alert is not higher, and hiding it would make the panel a
sales pitch for the model rather than an account of it.

**Computed in the tick loop, never on the request path.** TreeSHAP is ~60x the
cost of a prediction (85 ms for a 400-patient census against 1.4 ms to score
it), and the per-call overhead is what hurts: one patient alone costs 3.2 ms,
almost all of it building the DMatrix. So explanations are batched into a
single `pred_contribs` pass and memoised on the patient's **grid index** —
vitals arrive on the 5-minute charting grid and `compute_features` reads
`minutes_elapsed` off that grid, so between advances the feature vector is
identical and so is its explanation. The cache is exact, not an approximation.
At 60x replay it cuts the work ~5x and staggers it across ticks; opening a
panel is then a dictionary lookup. `GET /api/health` reports `explained` so
the hit rate is visible.

Measured on the default 400-patient cohort, whose census peaks near 40:

| census | scoring | + deterioration | + both models | budget |
|---|---|---|---|---|
| 36 | 100 ms | 106 ms | 120 ms | 1000 ms |
| 266 | 837 ms | 887 ms | — | 1000 ms |

At a 266-patient census the tick is already close to the budget **before**
explanations — that cost is `compute_features` and the per-patient pandas
slicing in `_score_active`, not SHAP, which is 6% of the tick. Explanations
are not what would need fixing first if you pointed this at a department that
size. Set `WATCHLIST_EXPLAIN_IN_TICK=0` to move them back onto the request
path if you want the tick lean anyway.

## Two things to know before quoting numbers

**Numbers from a `--source sim` run are a property of the simulator, not
evidence about MIMIC-IV.** The deterioration model scores AUROC 0.80 / AUPRC 0.32 against
NEWS2's 0.62 / 0.12, but that ceiling is set by the simulator's prodrome
assumption (physiology starts drifting 90–240 min before onset). Real
performance is unknown until real data lands.

**Subgroup numbers are event-limited.** One deterioration event contributes
~24 consecutive positive rows, so a subgroup with thousands of rows may rest
on a handful of events. `ml/train.py` reports the event count next to every
subgroup AUROC and flags any with fewer than 20. Use `-n 14000` for subgroup
numbers you intend to defend.

Both training scripts print their own caveats; `ml/train_admission.py`
includes a leakage ablation showing how much of its score comes from
`minutes_elapsed`, which in this simulator partly reads the label.

## Verification

`ml.features.check_paths_agree()` asserts that the live scoring path (numpy,
one window) and the training path (pandas, whole panel) produce identical
feature vectors. Run it after touching either — silent drift there means the
served model sees different inputs from the trained one.

`python -m api.mimic_fixture --selftest` builds a synthetic MIMIC-IV tree with
known defects — Fahrenheit temperatures, a mis-filed Celsius itemid, arterial
and cuff pressures at the same minute, disconnected-cuff zeros, fractional
SpO2 — runs the loader over it and asserts the cleaner caught each one, that
the emitted itemids are ones `features.py` already understands, and that the
label produces both classes. 48 checks. Run it after touching anything under
`api/mimic*.py` or `ml/outcomes.py`.

`python -m ml.mimic_items --verify PATH` checks every itemid against that
extract's own `d_items` dictionary.

`python -m api.manual_selftest` drives the four manual-control features
against the real app: scenario injection, hand-charting (including that it
moves the score while the clock is paused), override survival across
re-scores, and acknowledgement re-escalation. 73 checks.
`python -m api.interventions` covers the override and acknowledgement rules on
their own (43 checks), including that a flag acknowledged near 1.0 can still
re-raise. `python -m api.scenarios` checks each preset does what its name
claims — in particular that `silent_deteriorator` starts entirely normal and
never crosses a threshold alarm (25 checks).

`python -m api.reasons` asserts the phrasing contract in every language: every feature the model
can ask about produces a sentence, no sentence leaks a column name, channel
aggregation collapses restatements, and a quoted fact always agrees in sign
with the reason it is quoted under. Run it after touching `api/reasons.py`.

There is no committed test suite yet — those two self-checks stand in for one.
