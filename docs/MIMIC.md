# Running Watchlist on MIMIC-IV

What this pipeline reads, how it cleans it, and how "deterioration" is
defined — written so each choice can be defended or argued with, not just
reproduced.

Every claim here is checkable against the code: `ml/mimic_items.py` is the
item catalogue, `ml/outcomes.py` is the label, `api/mimic.py` is the loader.

---

## 1. What you need

| Project | PhysioNet page | Used for |
|---|---|---|
| MIMIC-IV (v2.2+) | `mimic-iv` | `hosp/patients`, `hosp/admissions`, `icu/icustays`, `icu/inputevents`, `icu/chartevents`, `icu/procedureevents`, `icu/d_items` |
| MIMIC-IV-ED | `mimic-iv-ed` | `ed/edstays`, `ed/triage`, `ed/vitalsign` |

**These are two separate credentialed downloads on the same account.** Having
MIMIC-IV does not give you MIMIC-IV-ED. The ED module is what makes this an ED
triage product: it supplies the stay boundaries and, critically, the triage
acuity that `esi` means. Without it you can still run (`--mimic-cohort icu`),
but you are building something else — see §7.

Unpack them anywhere. The loader finds `hosp/`, `icu/` and `ed/` under the root
you give it, tolerating the version subdirectory some downloads add and the
several places people put the ED module.

```bash
python -m ml.mimic_items --verify /data/mimic-iv     # check itemids first
python -m ml.train --source mimic --mimic-root /data/mimic-iv
```

---

## 2. Which itemids

Run the verifier before quoting any of this. It reads *your* `icu/d_items.csv.gz`
and prints our expected label beside theirs, so a release difference shows up
as a mismatch rather than as a wrong number six steps downstream:

```bash
python -m ml.mimic_items --verify /data/mimic-iv
```

### Vital signs

| Channel | itemid | Label | Unit | Priority |
|---|---|---|---|---|
| `hr` | 220045 | Heart Rate | bpm | primary |
| `rr` | 220210 | Respiratory Rate | insp/min | primary |
| | 224690 | Respiratory Rate (Total) | insp/min | fallback |
| `spo2` | 220277 | O2 saturation pulseoxymetry | % | primary |
| `sbp` | 220179 | Non Invasive Blood Pressure systolic | mmHg | primary |
| | 220050 | Arterial Blood Pressure systolic | mmHg | fallback |
| `dbp` | 220180 | Non Invasive Blood Pressure diastolic | mmHg | primary |
| | 220051 | Arterial Blood Pressure diastolic | mmHg | fallback |
| `temp` | 223762 | Temperature Celsius | °C | primary |
| | 223761 | Temperature Fahrenheit | °F | converted |

**Why two itemids per pressure, and why the cuff wins.** A cuff and an arterial
line both chart "systolic", and they are not the same measurement — an
arterial line reads systematically higher at the systolic peak. Mixing them
puts a step change into a patient's trend line at the moment a line goes in,
which is exactly the kind of artefact a trend-based model will learn. When both
exist at the same minute we keep the cuff, because that is what an ED board
reflects; the line is used only where there is no cuff reading.

**Why `rr` has a fallback.** 224690 is the ventilator's total rate and 220210
the monitored rate. On a non-ventilated ED patient they carry the same thing.
220210 wins when both are present.

MIMIC-IV-ED's `vitalsign` table has no itemids at all — it is already wide, one
column per vital. Those columns are mapped by name, and re-emitted under the
canonical itemids above.

### Outcome itemids

Vasopressors, from `icu/inputevents`:

| itemid | Drug |
|---|---|
| 221906 | Norepinephrine |
| 221289 | Epinephrine |
| 221662 | Dopamine |
| 221749 | Phenylephrine |
| 222315 | Vasopressin |

**Dobutamine (221653) is deliberately excluded.** It is an inotrope given for
cardiac output, not a marker of circulatory collapse, and including it widens
the label beyond what "vasopressor start" is normally taken to mean. Enable it
with `OutcomeSpec(include_inotropes=True)` if you want the broader definition —
and say so if you do.

Ventilation, from `icu/procedureevents`, **off by default**: 225792 (Invasive
Ventilation), 225794 (Non-invasive Ventilation).

---

## 3. Unit handling

Applied in this order. The order matters: range-filtering before unit
conversion would reject every Fahrenheit temperature as implausible.

1. **Fahrenheit → Celsius** for itemid 223761 and for MIMIC-IV-ED's
   `vitalsign.temperature`, which is Fahrenheit despite having no unit column
   to say so. This is the single most common way to get this pipeline quietly
   wrong: unconverted, every patient looks febrile at 98°C and the model learns
   a constant.
2. **Value-based Fahrenheit detection.** A *Celsius-labelled* reading above
   50 is Fahrenheit under a mis-filed itemid, and is converted. The check is
   one-directional on purpose — 45°C is past the survivable ceiling, so there
   is no ambiguity in that direction.
3. **SpO2 as a fraction.** Values in (0, 1.5] are multiplied by 100.

The QC report prints how many readings hit each path, including how many
Fahrenheit conversions were triggered by value rather than by itemid. If that
second number is large in your extract, something upstream is mis-filed and
worth chasing before you trust anything.

---

## 4. Out-of-range values

These are **artefact filters, not clinical normal ranges**. The point is to
drop values no living patient produces while keeping genuinely extreme ones —
a heart rate of 210 is a real tachycardia and is kept; 900 is a lead coming
off and is not.

| Channel | Kept range | What it excludes |
|---|---|---|
| `hr` | 20–250 bpm | lead artefact, monitor zeros |
| `rr` | 2–60 /min | apnoea artefact, counting errors |
| `spo2` | 50–100 % | probe off the finger |
| `sbp` | 40–300 mmHg | disconnected cuff (zeros) |
| `dbp` | 10–200 mmHg | same |
| `temp` | 25–45 °C | unplugged probe, unconverted °F |

Rejected values become **NaN, not clipped**. Clipping would invent a reading at
the boundary and let forward-fill propagate it; NaN degrades honestly to "not
measured", which `chartevents_to_wide` and XGBoost both already handle.

The loader prints exactly how many readings each filter dropped, per channel.
Quote that number rather than "we cleaned the data".

---

## 5. The deterioration label

A stay **deteriorates** if any enabled component is recorded after its vitals
window opens. `onset_minutes` is the **earliest** such event, measured in
minutes from the stay's first charted observation.

That origin is not arbitrary: `ml/features.py` measures `t` from the same
point, so `time_to_onset = onset_minutes - t` is a real time-to-event. Anchor
onset to ED `intime` instead and every label silently shifts by the gap between
arrival and first observation.

| Component | Source | Default |
|---|---|---|
| ICU transfer | `icu/icustays.intime`, joined on `hadm_id` | **on** |
| Vasopressor start | `icu/inputevents.starttime` | **on** |
| Rapid response | — not recorded in MIMIC-IV — | **unavailable** |
| Invasive ventilation | `icu/procedureevents.starttime` | off |
| In-hospital death | `hosp/admissions.deathtime` | off |

The model predicts whether onset falls in the next **120 minutes**
(`HORIZON_MIN`), and is trained only on **pre-onset rows** — every stay is
censored at its own onset, unchanged from the simulator pipeline.

### Three things to say before anyone quotes a number

**MIMIC-IV does not record rapid response activations.** There is no RRT table,
no RRT itemid, no call log in any module. You asked for RRT in the label; it
cannot come from this dataset. The component exists as a hook
(`ml.outcomes.RRT_ITEMIDS`) for a site with its own feed, and the code warns
loudly if you enable it while that list is empty. **Describe the label as "ICU
transfer or vasopressor initiation", not as including RRT.** A judge who knows
MIMIC-IV will ask, and the honest answer lands better than the alternative.

**Vasopressors are only visible once the patient reaches an ICU.**
`inputevents` is an ICU table — pressors started in the ED or on a ward are not
in it. In practice the vasopressor component rarely fires *before* ICU
transfer, so it adds less than its presence in the definition suggests. The
loader reports each component's **unique contribution** — how many stays it
labels that nothing else would — so this is visible in your data rather than
assumed. On the test fixture that number is zero; check yours.

**ICU transfer is a treatment decision, not a physiological event.** It
reflects bed availability, time of day, and unit admission policy alongside how
sick the patient is. A model trained on it partly learns triage behaviour, and
it will inherit whatever disparities exist in who gets transferred. It is the
standard proxy in this literature and it is what reviewers expect — but it is a
proxy. The alternative, a physiology-derived label, would be circular here,
because physiology is the model's input.

### Excluded stays

- **No triage acuity** (`triage.acuity` null) — dropped. `esi` is a model
  feature and the re-triage logic's reference point; there is nothing sensible
  to impute.
- **Fewer than 3 usable observations** — dropped. No trend features are
  computable, so the stay contributes noise.
- **Event before the first vital sign** — kept as a *negative*, label cleared.
  There is no pre-onset window to predict from, and `build_dataset` would
  censor every row of it anyway. Counted separately in the label report.

---

## 6. Reading 330 million rows

`icu/chartevents.csv.gz` is roughly 330M rows, about 30 GB expanded. It is
streamed in chunks, filtered to our itemids and our stays, and only the
survivors are concatenated — a ~0.1% keep rate for this project's vitals.
Timestamps are parsed *after* filtering, because parsing 330M timestamps costs
far more than reading them.

The ED path avoids this entirely: `ed/vitalsign` is small and already wide, so
an ED-cohort run does not touch `chartevents` at all.

Use `--mimic-limit N` while iterating. The full cohort takes minutes; 2000
stays takes seconds.

---

## 7. The `icu` fallback cohort

`--mimic-cohort icu` runs on core MIMIC-IV with no ED module. It works, and it
is **not this product**:

- There is no triage acuity, so `esi` is NaN. The re-triage feature — comparing
  model risk against assigned acuity — is meaningless.
- ICU transfer cannot be a deterioration outcome for patients already in the
  ICU; the loader disables that component automatically and says so.
- The cohort is ICU stays, not ED presentations. Nothing about it supports a
  claim regarding ED triage.

Use it to exercise the code path. Do not present numbers from it as ED results.

---

## 8. Verifying without the data

The loader is tested against a synthetic MIMIC-IV tree that reproduces the
layout, the dtypes, the gzip, and the *defects* — Fahrenheit temperatures, a
mis-filed Celsius itemid, arterial and cuff pressures at the same minute,
disconnected-cuff zeros, fractional SpO2, missing acuity, patients with two
admissions. Each defect is injected at a counted rate and the selftest asserts
the cleaner caught it:

```bash
python -m api.mimic_fixture --selftest      # build, load, assert (48 checks)
python -m api.mimic_fixture --out /tmp/fake-mimic   # keep one to poke at
```

This proves the plumbing. It proves nothing clinical — the fixture's vitals are
crude by design.

---

## 9. What to report

The training run prints, in order: cohort provenance (every table path), the
vitals QC report, the label definition with per-component counts, then the
usual metrics. All of it belongs in your appendix. The model's sidecar JSON
also records the cohort, the outcome definition and a caveat string, so a model
file can always say what made it.

A MIMIC run writes to `ml/deterioration_mimic.json`, never over the shipped
simulator-trained `ml/deterioration.json`. Point the API at it with
`WATCHLIST_MODEL=ml/deterioration_mimic.json`.

Carry the existing caveats forward too: subgroup AUROCs are **event-limited**,
and one deterioration event contributes ~24 consecutive positive rows, so a
subgroup spanning thousands of rows may rest on a handful of events. The
trainer prints the event count beside every subgroup and flags any below 20.
