"""The MIMIC-IV item catalogue: what we read, in what units, within what range.

This module is the thing to point at when someone asks "which itemids?". It is
data, not logic -- every concept the loader extracts is one entry here with its
itemids, its expected unit, its plausible range, and a note on why.

Three rules it enforces, in order:

1. **Map.** Real extracts carry several itemids per concept (a cuff pressure
   and an arterial line are both "systolic"). Each maps to one canonical
   channel.
2. **Normalise units.** Temperature arrives in Fahrenheit under one itemid and
   Celsius under another, and MIMIC-IV-ED charts it in Fahrenheit with no
   itemid at all. Everything is converted to the canonical unit.
3. **Reject the impossible.** A heart rate of 0 or a SpO2 of 1500 is an
   artefact, not a patient. Out-of-range values become NaN, which the feature
   builder already handles, rather than being clipped -- clipping would invent
   a reading at the boundary and let it propagate through forward-fill.

The loader re-emits everything under the CANONICAL itemid for its channel, so
``ml.features.MIMIC_ITEMID_MAP`` and every model downstream stay unchanged. A
Fahrenheit temperature read from 223761 leaves this module as a Celsius value
under 223762.

**Verify these against your own extract before quoting them.** Item labels are
stable across MIMIC-IV releases but this list is written from the published
dictionary, not from your copy of it::

    python -m ml.mimic_items --verify /path/to/mimic-iv

reads ``icu/d_items.csv.gz`` and prints our label next to theirs for every
itemid below, so a mismatch is visible rather than silent.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """One source itemid feeding one canonical channel."""

    itemid: int
    label: str          # the label we expect in d_items
    unit: str           # the unit the raw value arrives in
    note: str = ""


@dataclass(frozen=True)
class Concept:
    """One canonical vital sign and everything that can produce it."""

    channel: str                  # matches ml.features.VITALS
    canonical_itemid: int         # what the loader re-emits under
    unit: str                     # canonical unit
    valid: tuple[float, float]    # physiologically plausible, inclusive
    items: tuple[Item, ...] = ()
    rationale: str = ""

    def itemids(self) -> tuple[int, ...]:
        return tuple(i.itemid for i in self.items)


# ---------------------------------------------------------------------------
# Plausible ranges
# ---------------------------------------------------------------------------
#
# These are ARTEFACT filters, not clinical normal ranges. The point is to drop
# values no living patient produces -- a systolic of 0 from a disconnected
# cuff, a temperature of 0 from an unplugged probe -- while keeping genuinely
# extreme readings, because those are exactly the patients this board exists
# to find. They are deliberately wider than any reference range: a heart rate
# of 210 is kept, a heart rate of 900 is not.
#
# Anything outside becomes NaN. ml.features.chartevents_to_wide forward-fills
# with a limit and leaves long gaps as NaN, and XGBoost consumes NaN natively,
# so a rejected reading degrades to "not measured" rather than to a wrong
# measurement.

VITAL_CONCEPTS: tuple[Concept, ...] = (
    Concept(
        channel="hr",
        canonical_itemid=220045,
        unit="bpm",
        valid=(20.0, 250.0),
        rationale="Below 20 or above 250 on a monitor is lead artefact, not rhythm.",
        items=(
            Item(220045, "Heart Rate", "bpm"),
        ),
    ),
    Concept(
        channel="rr",
        canonical_itemid=220210,
        unit="breaths/min",
        valid=(2.0, 60.0),
        rationale=(
            "Two itemids, both real charting. 224690 is the ventilator's total "
            "rate and 220210 the spontaneous/monitored rate; on a non-ventilated "
            "ED patient they carry the same thing. 220210 wins when both exist."
        ),
        items=(
            Item(220210, "Respiratory Rate", "insp/min"),
            Item(224690, "Respiratory Rate (Total)", "insp/min",
                 "ventilator total; lower priority than 220210"),
        ),
    ),
    Concept(
        channel="spo2",
        canonical_itemid=220277,
        unit="%",
        valid=(50.0, 100.0),
        rationale=(
            "Below 50% is almost always a probe off the finger. Values in "
            "[0, 1] are rescaled as fractions before this filter applies."
        ),
        items=(
            Item(220277, "O2 saturation pulseoxymetry", "%"),
        ),
    ),
    Concept(
        channel="sbp",
        canonical_itemid=220179,
        unit="mmHg",
        valid=(40.0, 300.0),
        rationale=(
            "Cuff and arterial line both chart systolic. They are not "
            "interchangeable -- an arterial line reads systematically higher "
            "at the systolic peak -- so the cuff (220179) takes priority, "
            "matching what an ED board would be showing. The line is used only "
            "when no cuff reading exists at that time."
        ),
        items=(
            Item(220179, "Non Invasive Blood Pressure systolic", "mmHg"),
            Item(220050, "Arterial Blood Pressure systolic", "mmHg",
                 "invasive; lower priority"),
        ),
    ),
    Concept(
        channel="dbp",
        canonical_itemid=220180,
        unit="mmHg",
        valid=(10.0, 200.0),
        rationale="Same cuff-over-line priority as systolic.",
        items=(
            Item(220180, "Non Invasive Blood Pressure diastolic", "mmHg"),
            Item(220051, "Arterial Blood Pressure diastolic", "mmHg",
                 "invasive; lower priority"),
        ),
    ),
    Concept(
        channel="temp",
        canonical_itemid=223762,
        unit="degC",
        valid=(25.0, 45.0),
        rationale=(
            "The single biggest unit trap in MIMIC-IV. 223761 is Fahrenheit "
            "and 223762 Celsius, and MIMIC-IV-ED's vitalsign.temperature is "
            "Fahrenheit with no itemid at all. Everything is converted to "
            "Celsius, and a value-based check catches anything still arriving "
            "in Fahrenheit under a Celsius itemid."
        ),
        items=(
            Item(223762, "Temperature Celsius", "degC"),
            Item(223761, "Temperature Fahrenheit", "degF", "converted to Celsius"),
        ),
    ),
)

CONCEPT_BY_CHANNEL: dict[str, Concept] = {c.channel: c for c in VITAL_CONCEPTS}

# Source itemid -> canonical channel, for filtering chartevents while reading.
SOURCE_ITEMID_TO_CHANNEL: dict[int, str] = {
    item.itemid: c.channel for c in VITAL_CONCEPTS for item in c.items
}

# Source itemid -> priority, lower wins when two itemids chart the same channel
# at the same minute. Position in the Concept.items tuple is the priority.
ITEMID_PRIORITY: dict[int, int] = {
    item.itemid: rank
    for c in VITAL_CONCEPTS
    for rank, item in enumerate(c.items)
}


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

# A Celsius-labelled reading above this is Fahrenheit that slipped through.
# 45 C is past the survivable ceiling; 45 F is a plausible mis-filed room-
# temperature probe. The check is one-directional on purpose: a Fahrenheit
# reading below 50 (=10 C) is discarded by the range filter either way.
FAHRENHEIT_SUSPICION_C = 50.0

# SpO2 charted as a fraction rather than a percentage.
FRACTION_CEILING = 1.5


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


# ---------------------------------------------------------------------------
# Outcome itemids -- see ml/outcomes.py for how they become a label
# ---------------------------------------------------------------------------

# Vasopressors in mimiciv_icu.inputevents. Deliberately the pressors only:
# dobutamine (221653) is an inotrope given for cardiac output, not a marker of
# circulatory collapse, and including it would broaden the label beyond what
# "vasopressor start" is usually taken to mean. Add it explicitly if you want
# the wider definition.
VASOPRESSOR_ITEMIDS: dict[int, str] = {
    221906: "Norepinephrine",
    221289: "Epinephrine",
    221662: "Dopamine",
    221749: "Phenylephrine",
    222315: "Vasopressin",
}

INOTROPE_ITEMIDS: dict[int, str] = {
    221653: "Dobutamine",
}

# mimiciv_icu.procedureevents. Not part of the default label -- offered because
# "intubated within the horizon" is a defensible deterioration outcome and some
# reviewers will ask for it.
VENTILATION_ITEMIDS: dict[int, str] = {
    225792: "Invasive Ventilation",
    225794: "Non-invasive Ventilation",
}


# ---------------------------------------------------------------------------
# verification against the user's own d_items
# ---------------------------------------------------------------------------

def expected_labels() -> dict[int, str]:
    """Every itemid this project reads, with the label we expect for it."""
    out: dict[int, str] = {}
    for c in VITAL_CONCEPTS:
        for item in c.items:
            out[item.itemid] = item.label
    out.update(VASOPRESSOR_ITEMIDS)
    out.update(INOTROPE_ITEMIDS)
    out.update(VENTILATION_ITEMIDS)
    return out


def verify_against_d_items(root: str | pathlib.Path) -> int:
    """Print our label beside the extract's for every itemid we read.

    Returns the number of mismatches. Run this before quoting the catalogue:
    an itemid that means something else in your release is a silent, total
    failure, and it is the first thing worth ruling out.
    """
    from api.mimic_io import find_table, read_table

    path = find_table(root, "d_items")
    if path is None:
        print(f"d_items not found under {root} -- cannot verify")
        return -1

    d = read_table(path, usecols=["itemid", "label", "unitname"])
    have = {int(r.itemid): (str(r.label), str(r.unitname)) for r in d.itertuples()}

    expected = expected_labels()
    mismatches = 0
    print(f"{'itemid':>8}  {'ours':<38}{'theirs':<38}{'unit':<12}")
    print("-" * 98)
    for itemid, ours in sorted(expected.items()):
        theirs, unit = have.get(itemid, ("*** NOT IN d_items ***", ""))
        ok = theirs.strip().lower() == ours.strip().lower()
        if not ok:
            mismatches += 1
        print(f"{itemid:>8}  {ours:<38}{theirs:<38}{unit:<12}{'' if ok else '  <-- MISMATCH'}")

    print()
    if mismatches:
        print(f"{mismatches} mismatch(es). Fix ml/mimic_items.py before trusting any "
              f"number that comes out of this pipeline.")
    else:
        print(f"all {len(expected)} itemids match this extract's dictionary.")
    return mismatches


def describe() -> str:
    """The catalogue as prose, for the docs and for a slide."""
    lines = ["MIMIC-IV item catalogue", "=" * 23, ""]
    for c in VITAL_CONCEPTS:
        lines.append(f"{c.channel}  ->  itemid {c.canonical_itemid} ({c.unit}), "
                     f"plausible {c.valid[0]:g}-{c.valid[1]:g}")
        for rank, item in enumerate(c.items):
            tag = "primary" if rank == 0 else f"fallback {rank}"
            extra = f" -- {item.note}" if item.note else ""
            lines.append(f"      {item.itemid}  {item.label} [{item.unit}] ({tag}){extra}")
        lines.append(f"      why: {c.rationale}")
        lines.append("")
    lines.append("vasopressors (inputevents): " +
                 ", ".join(f"{k} {v}" for k, v in VASOPRESSOR_ITEMIDS.items()))
    lines.append("excluded inotrope: " +
                 ", ".join(f"{k} {v}" for k, v in INOTROPE_ITEMIDS.items()))
    lines.append("ventilation (procedureevents, optional): " +
                 ", ".join(f"{k} {v}" for k, v in VENTILATION_ITEMIDS.items()))
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", metavar="MIMIC_ROOT",
                    help="check every itemid against that extract's icu/d_items")
    args = ap.parse_args()
    if args.verify:
        raise SystemExit(1 if verify_against_d_items(args.verify) else 0)
    print(describe())


if __name__ == "__main__":  # pragma: no cover
    main()
