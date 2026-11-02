"""Deterioration, defined from what actually happened to the patient.

The simulator hands the trainer a ``deteriorates`` flag and an
``onset_minutes`` it invented. Real data has no such column, so this module
builds one from recorded events and hands back exactly the same two fields.
Everything downstream -- ``ml.train.build_dataset``, the horizon, the
censoring, the model -- is unchanged.

The label
---------

A stay **deteriorates** if any of these is recorded after its vitals window
opens, and ``onset_minutes`` is the earliest of them, in minutes from the
stay's first charted observation (the same origin ``ml.features`` uses for
``t``, which is what makes ``time_to_onset`` meaningful):

======================  ==========================================  ==========
component               source                                      default
======================  ==========================================  ==========
ICU transfer            ``icu/icustays.intime``, joined on hadm_id  on
Vasopressor start       ``icu/inputevents.starttime``, pressors      on
Rapid response          -- not recorded in MIMIC-IV --               off
Invasive ventilation    ``icu/procedureevents.starttime``            off
In-hospital death       ``hosp/admissions.deathtime``                off
======================  ==========================================  ==========

Three things to say out loud before anyone quotes a number from this.

**MIMIC-IV does not record rapid response activations.** There is no RRT
table, no RRT itemid, and no call log in any module. The component exists
here as a hook (:data:`RRT_ITEMIDS`) so a site with its own RRT feed can drop
itemids in, but on a stock MIMIC-IV extract it contributes nothing and must be
described as absent rather than as zero. Reporting a label as "ICU transfer,
vasopressor, or RRT" when the third term is structurally empty would overstate
what was measured.

**Vasopressors are only visible once the patient is in an ICU.**
``inputevents`` is an ICU table. Pressors started in the ED or on a ward are
not in it. In practice this means the vasopressor component rarely fires
*before* the ICU-transfer component, so it adds less than it appears to --
:func:`summarise` reports how many stays each component uniquely contributed
so the overlap is visible rather than assumed.

**ICU transfer is a treatment decision, not a physiological event.** It
reflects bed availability, time of day, and unit admission policy alongside
how sick the patient is. It is the standard proxy in this literature and it is
what a judge will expect to see, but it is a proxy, and a model trained on it
partly learns triage behaviour. The alternative -- a physiology-derived label
-- would be circular here, because physiology is the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.mimic_items import (
    INOTROPE_ITEMIDS,
    VASOPRESSOR_ITEMIDS,
    VENTILATION_ITEMIDS,
)

# Rapid-response itemids. Empty on stock MIMIC-IV -- see the module docstring.
# If your site has an RRT feed, put its itemids here and set
# ``OutcomeSpec.rapid_response=True``; they are looked up in procedureevents.
RRT_ITEMIDS: dict[int, str] = {}


@dataclass(frozen=True)
class OutcomeSpec:
    """Which components make up the deterioration label.

    Defaults are the three the brief asked for, minus the one the data cannot
    supply. Turn the others on explicitly and say so when you report.
    """

    icu_transfer: bool = True
    vasopressor: bool = True
    rapid_response: bool = False      # unavailable on stock MIMIC-IV
    ventilation: bool = False
    death: bool = False
    include_inotropes: bool = False   # widens "vasopressor" to dobutamine

    def pressor_itemids(self) -> dict[int, str]:
        items = dict(VASOPRESSOR_ITEMIDS)
        if self.include_inotropes:
            items.update(INOTROPE_ITEMIDS)
        return items

    def describe(self) -> str:
        on = [
            name
            for name, flag in (
                ("ICU transfer", self.icu_transfer),
                ("vasopressor start", self.vasopressor),
                ("rapid response", self.rapid_response),
                ("invasive ventilation", self.ventilation),
                ("in-hospital death", self.death),
            )
            if flag
        ]
        text = " or ".join(on) if on else "(nothing enabled)"
        if self.rapid_response and not RRT_ITEMIDS:
            text += "  [rapid response ENABLED BUT NO ITEMIDS -- contributes nothing]"
        if self.include_inotropes:
            text += "  [vasopressor widened to include dobutamine]"
        return text


@dataclass
class OutcomeEvents:
    """First qualifying event per stay, per component."""

    # stay_id -> event time, one frame per component
    by_component: dict[str, pd.DataFrame] = field(default_factory=dict)

    def earliest(self, stays: pd.DataFrame) -> pd.DataFrame:
        """Collapse components into one onset per stay.

        Returns ``stay_id, onset_time, onset_source`` for stays with any
        event. ``stays`` is only used for its stay_id index.
        """
        frames = []
        for name, df in self.by_component.items():
            if df is None or not len(df):
                continue
            f = df[["stay_id", "event_time"]].copy()
            f["onset_source"] = name
            frames.append(f)
        if not frames:
            return pd.DataFrame(
                columns=["stay_id", "onset_time", "onset_source"]
            ).astype({"stay_id": stays["stay_id"].dtype})

        allev = pd.concat(frames, ignore_index=True)
        allev = allev.sort_values(["stay_id", "event_time"], kind="mergesort")
        first = allev.groupby("stay_id", as_index=False).first()
        return first.rename(columns={"event_time": "onset_time"})


# ---------------------------------------------------------------------------
# component extractors
# ---------------------------------------------------------------------------
#
# Each takes the cohort's stay table -- which must carry stay_id, hadm_id and
# window_start -- plus the relevant MIMIC table, and returns the first
# qualifying event per stay, at or after window_start.


def _first_after(
    stays: pd.DataFrame,
    events: pd.DataFrame,
    on: str,
    time_col: str,
) -> pd.DataFrame:
    """First ``events[time_col]`` per stay that falls at/after window_start.

    Joined on ``on`` (hadm_id in every current caller), which is what keeps a
    patient's second admission from labelling their first.
    """
    if not len(events) or on not in stays.columns:
        return pd.DataFrame(columns=["stay_id", "event_time"])

    link = stays.loc[stays[on].notna(), ["stay_id", on, "window_start"]]
    if not len(link):
        return pd.DataFrame(columns=["stay_id", "event_time"])

    ev = events.loc[events[time_col].notna(), [on, time_col]].copy()
    merged = link.merge(ev, on=on, how="inner")
    merged = merged.loc[merged[time_col] >= merged["window_start"]]
    if not len(merged):
        return pd.DataFrame(columns=["stay_id", "event_time"])

    out = (
        merged.sort_values(["stay_id", time_col], kind="mergesort")
        .groupby("stay_id", as_index=False)
        .first()[["stay_id", time_col]]
        .rename(columns={time_col: "event_time"})
    )
    return out


def icu_transfer_events(stays: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    """First ICU admission for the stay's hospitalisation."""
    return _first_after(stays, icustays, on="hadm_id", time_col="intime")


def vasopressor_events(
    stays: pd.DataFrame, inputevents: pd.DataFrame, spec: OutcomeSpec
) -> pd.DataFrame:
    """First vasopressor infusion started for the stay's hospitalisation.

    ``amount``/``rate`` are deliberately not thresholded: any started pressor
    infusion counts. A rate threshold would be a second clinical judgement to
    defend, and the ones in the literature disagree with each other.
    """
    items = spec.pressor_itemids()
    ev = inputevents.loc[inputevents["itemid"].isin(items)]
    return _first_after(stays, ev, on="hadm_id", time_col="starttime")


def ventilation_events(stays: pd.DataFrame, procedureevents: pd.DataFrame) -> pd.DataFrame:
    ev = procedureevents.loc[procedureevents["itemid"].isin(VENTILATION_ITEMIDS)]
    return _first_after(stays, ev, on="hadm_id", time_col="starttime")


def rapid_response_events(stays: pd.DataFrame, procedureevents: pd.DataFrame) -> pd.DataFrame:
    """Empty on stock MIMIC-IV. See the module docstring."""
    if not RRT_ITEMIDS:
        return pd.DataFrame(columns=["stay_id", "event_time"])
    ev = procedureevents.loc[procedureevents["itemid"].isin(RRT_ITEMIDS)]
    return _first_after(stays, ev, on="hadm_id", time_col="starttime")


def death_events(stays: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    ev = admissions.loc[admissions["deathtime"].notna(), ["hadm_id", "deathtime"]]
    return _first_after(stays, ev, on="hadm_id", time_col="deathtime")


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build_labels(
    stays: pd.DataFrame,
    spec: OutcomeSpec,
    icustays: pd.DataFrame | None = None,
    inputevents: pd.DataFrame | None = None,
    procedureevents: pd.DataFrame | None = None,
    admissions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, OutcomeEvents]:
    """Attach ``deteriorates`` and ``onset_minutes`` to a stay table.

    ``stays`` must carry ``stay_id``, ``hadm_id`` and ``window_start`` -- the
    timestamp of the stay's first charted vital, which is what
    ``ml.features.chartevents_to_wide`` measures ``t`` from. Onset is returned
    on that same clock, so ``onset_minutes - t`` is a real time-to-event.

    Returns the stay table with the two label columns added, plus the raw
    events for auditing.
    """
    empty = pd.DataFrame(columns=["stay_id", "event_time"])
    events = OutcomeEvents()

    if spec.icu_transfer and icustays is not None:
        events.by_component["icu_transfer"] = icu_transfer_events(stays, icustays)
    if spec.vasopressor and inputevents is not None:
        events.by_component["vasopressor"] = vasopressor_events(stays, inputevents, spec)
    if spec.rapid_response and procedureevents is not None:
        events.by_component["rapid_response"] = rapid_response_events(stays, procedureevents)
    if spec.ventilation and procedureevents is not None:
        events.by_component["ventilation"] = ventilation_events(stays, procedureevents)
    if spec.death and admissions is not None:
        events.by_component["death"] = death_events(stays, admissions)
    events.by_component = {k: v for k, v in events.by_component.items() if v is not empty}

    onset = events.earliest(stays)
    out = stays.merge(onset, on="stay_id", how="left")

    out["deteriorates"] = out["onset_time"].notna()
    delta = (out["onset_time"] - out["window_start"]).dt.total_seconds() / 60.0
    out["onset_minutes"] = delta.where(out["deteriorates"], np.nan)

    # An event at or before the first observation cannot be predicted from
    # that stay's vitals -- there is no pre-onset window. Drop the label
    # rather than the row: the stay stays in the cohort as a negative it
    # cannot contribute positives to, and build_dataset's censoring would
    # otherwise discard every row of it anyway.
    already = out["deteriorates"] & (out["onset_minutes"] <= 0)
    out.loc[already, ["deteriorates", "onset_minutes"]] = [False, np.nan]
    out["onset_already_underway"] = already

    return out, events


def summarise(labelled: pd.DataFrame, events: OutcomeEvents, spec: OutcomeSpec) -> str:
    """The paragraph you read out when someone asks how the label is defined."""
    n = len(labelled)
    pos = int(labelled["deteriorates"].sum())
    lines = [
        "DETERIORATION LABEL",
        "-" * 19,
        f"definition        {spec.describe()}",
        f"stays             {n:,}",
        f"deteriorating     {pos:,} ({pos / max(n, 1):.2%})",
    ]

    already = int(labelled.get("onset_already_underway", pd.Series(dtype=bool)).sum())
    if already:
        lines.append(
            f"excluded          {already:,} stay(s) whose event preceded their first "
            f"vital sign -- nothing to predict from"
        )

    if pos:
        om = labelled.loc[labelled["deteriorates"], "onset_minutes"]
        lines.append(
            f"onset from first obs  median {om.median():.0f} min "
            f"(IQR {om.quantile(.25):.0f}-{om.quantile(.75):.0f}, max {om.max():.0f})"
        )

    # Which component fired first, and which ones would have been alone.
    if "onset_source" in labelled.columns:
        counts = labelled.loc[labelled["deteriorates"], "onset_source"].value_counts()
        lines.append("")
        lines.append("earliest component:")
        for name, c in counts.items():
            lines.append(f"  {name:<22}{c:>7,}  ({c / max(pos, 1):.1%})")

    # Unique contribution: stays this component labels that no other would.
    if len(events.by_component) > 1:
        lines.append("")
        lines.append("unique contribution (stays only this component labels):")
        all_stays = {
            k: set(v["stay_id"]) for k, v in events.by_component.items() if len(v)
        }
        for name, ids in all_stays.items():
            others: set = set()
            for other, oids in all_stays.items():
                if other != name:
                    others |= oids
            lines.append(f"  {name:<22}{len(ids - others):>7,}  of {len(ids):,} total")

    if spec.rapid_response and not RRT_ITEMIDS:
        lines.append("")
        lines.append("  WARNING: rapid response is enabled but RRT_ITEMIDS is empty.")
        lines.append("  MIMIC-IV does not record RRT activations. Report the label as")
        lines.append("  'ICU transfer or vasopressor start', not as including RRT.")

    return "\n".join(lines)
