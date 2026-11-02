"""Real MIMIC-IV extract -> the two frames the rest of this project expects.

The engine and the trainer never learn that the data changed. This module
produces a ``patients`` frame and a ``chartevents`` frame in exactly the shape
``sim.simulator`` emits, including the ``deteriorates`` / ``onset_minutes``
columns the trainer labels from -- built here from recorded outcomes rather
than invented (see :mod:`ml.outcomes`).

Cohort modes
------------

``ed`` (default, requires MIMIC-IV-ED)
    One row per ED stay. Vitals from ``ed/vitalsign``, acuity from
    ``ed/triage.acuity``, which is the ESI this product is built around. This
    is the cohort the board is actually for.

``icu`` (fallback, core MIMIC-IV only)
    One row per ICU stay, vitals from ``icu/chartevents``. Provided so the
    pipeline runs on a core-only download, but it is a **different product**:
    there is no triage acuity, so ``esi`` is NaN and the re-triage feature is
    meaningless, and ICU transfer cannot be an outcome for patients already in
    the ICU. Use it to exercise the code, not to make claims about ED triage.

MIMIC-IV-ED is a separate PhysioNet project from MIMIC-IV. If you have one you
do not automatically have the other; both are credentialed downloads on the
same account.

Everything is re-emitted under the canonical itemid for its channel, so
``ml.features.MIMIC_ITEMID_MAP`` and the feature code are untouched -- a
Fahrenheit temperature read from 223761 arrives downstream as Celsius under
223762.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from api.mimic_io import available, find_table, read_filtered, read_table
from ml.mimic_items import (
    CONCEPT_BY_CHANNEL,
    FAHRENHEIT_SUSPICION_C,
    FRACTION_CEILING,
    ITEMID_PRIORITY,
    SOURCE_ITEMID_TO_CHANNEL,
    VITAL_CONCEPTS,
    fahrenheit_to_celsius,
)
from ml.outcomes import OutcomeSpec, build_labels, summarise

log = logging.getLogger("watchlist.mimic")

# MIMIC-IV-ED vitalsign is already wide; these are its vital columns and the
# unit each arrives in. temperature is Fahrenheit -- the single most common
# way to get this pipeline subtly wrong.
ED_VITALSIGN_COLUMNS: dict[str, tuple[str, str]] = {
    "heartrate": ("hr", "bpm"),
    "resprate": ("rr", "breaths/min"),
    "o2sat": ("spo2", "%"),
    "sbp": ("sbp", "mmHg"),
    "dbp": ("dbp", "mmHg"),
    "temperature": ("temp", "degF"),
}


@dataclass
class QCReport:
    """What the cleaner threw away, and why. Printed, and worth reading."""

    n_raw: int = 0
    n_kept: int = 0
    dropped_null: int = 0
    dropped_range: dict[str, int] = field(default_factory=dict)
    converted_f_to_c: int = 0
    sniffed_fahrenheit: int = 0
    rescaled_fraction: int = 0
    deduped_priority: int = 0
    per_channel: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            "VITALS QC",
            "-" * 9,
            f"raw observations      {self.n_raw:,}",
            f"kept                  {self.n_kept:,} ({self.n_kept / max(self.n_raw, 1):.1%})",
            f"dropped, null value   {self.dropped_null:,}",
        ]
        total_range = sum(self.dropped_range.values())
        lines.append(f"dropped, out of range {total_range:,}")
        for ch, n in sorted(self.dropped_range.items(), key=lambda kv: -kv[1]):
            if n:
                lo, hi = CONCEPT_BY_CHANNEL[ch].valid
                lines.append(f"    {ch:<6}{n:>10,}   outside [{lo:g}, {hi:g}]")
        if self.converted_f_to_c:
            lines.append(f"converted F -> C      {self.converted_f_to_c:,} (itemid 223761 / ED vitalsign)")
        if self.sniffed_fahrenheit:
            lines.append(f"  of which by value   {self.sniffed_fahrenheit:,} "
                         f"(Celsius-labelled but > {FAHRENHEIT_SUSPICION_C:g})")
        if self.rescaled_fraction:
            lines.append(f"rescaled SpO2 x100    {self.rescaled_fraction:,} (charted as a fraction)")
        if self.deduped_priority:
            lines.append(f"dropped, lower-priority itemid at same minute  {self.deduped_priority:,}")
        lines.append("")
        lines.append("kept per channel:")
        for ch, n in sorted(self.per_channel.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {ch:<6}{n:>12,}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

def normalise_vitals(long: pd.DataFrame, qc: QCReport) -> pd.DataFrame:
    """Clean a long vitals frame in place-ish, recording what was dropped.

    ``long`` carries ``stay_id, charttime, channel, value, source_itemid,
    source_unit``. Returns the same columns, cleaned. Order of operations
    matters: convert units first, then range-filter, or a Fahrenheit
    temperature gets rejected as out of range before anyone converts it.
    """
    qc.n_raw += len(long)

    df = long.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    null = df["value"].isna()
    qc.dropped_null += int(null.sum())
    df = df.loc[~null]

    # -- units ------------------------------------------------------------
    is_temp = df["channel"] == "temp"

    labelled_f = is_temp & (df["source_unit"] == "degF")
    if labelled_f.any():
        df.loc[labelled_f, "value"] = fahrenheit_to_celsius(df.loc[labelled_f, "value"])
        qc.converted_f_to_c += int(labelled_f.sum())

    # A Celsius-labelled temperature above 50 is Fahrenheit that slipped
    # through a mis-filed itemid. Catching it by value is the only way -- the
    # metadata already lied.
    sniffed = is_temp & ~labelled_f & (df["value"] > FAHRENHEIT_SUSPICION_C)
    if sniffed.any():
        df.loc[sniffed, "value"] = fahrenheit_to_celsius(df.loc[sniffed, "value"])
        qc.converted_f_to_c += int(sniffed.sum())
        qc.sniffed_fahrenheit += int(sniffed.sum())

    # SpO2 occasionally charted as a fraction.
    frac = (df["channel"] == "spo2") & (df["value"] > 0) & (df["value"] <= FRACTION_CEILING)
    if frac.any():
        df.loc[frac, "value"] = df.loc[frac, "value"] * 100.0
        qc.rescaled_fraction += int(frac.sum())

    # -- plausibility ------------------------------------------------------
    keep = pd.Series(True, index=df.index)
    for concept in VITAL_CONCEPTS:
        lo, hi = concept.valid
        m = df["channel"] == concept.channel
        bad = m & ((df["value"] < lo) | (df["value"] > hi))
        n_bad = int(bad.sum())
        qc.dropped_range[concept.channel] = qc.dropped_range.get(concept.channel, 0) + n_bad
        keep &= ~bad
    df = df.loc[keep]

    # -- one reading per stay/channel/minute -------------------------------
    # Two itemids can chart the same channel at the same timestamp (a cuff and
    # an arterial line). Keep the higher-priority source rather than whichever
    # happens to sort last.
    df["_priority"] = df["source_itemid"].map(ITEMID_PRIORITY).fillna(99).astype(int)
    before = len(df)
    df = (
        df.sort_values(["stay_id", "charttime", "channel", "_priority"], kind="mergesort")
        .drop_duplicates(subset=["stay_id", "charttime", "channel"], keep="first")
        .drop(columns="_priority")
    )
    qc.deduped_priority += before - len(df)

    qc.n_kept += len(df)
    for ch, n in df["channel"].value_counts().items():
        qc.per_channel[ch] = qc.per_channel.get(ch, 0) + int(n)
    return df


def to_canonical_chartevents(long: pd.DataFrame) -> pd.DataFrame:
    """Long cleaned vitals -> chartevents in the shape features.py expects.

    Re-emitted under each channel's canonical itemid, which is what keeps
    ``ml.features`` free of any knowledge that the source was real.
    """
    out = pd.DataFrame(
        {
            "subject_id": long["subject_id"].to_numpy(),
            "stay_id": long["stay_id"].to_numpy(),
            "charttime": long["charttime"].to_numpy(),
            "itemid": long["channel"].map(
                {c.channel: c.canonical_itemid for c in VITAL_CONCEPTS}
            ).to_numpy(),
            "valuenum": long["value"].to_numpy(dtype=float),
        }
    )
    return out.sort_values(["stay_id", "charttime", "itemid"], ignore_index=True)


# ---------------------------------------------------------------------------
# the source
# ---------------------------------------------------------------------------

class MimicCohortSource:
    """Real MIMIC-IV extract, satisfying both project interfaces.

    Implements :class:`api.sources.CohortSource` (``load()``) and the
    trainer's ``patients()`` / ``chartevents()``. The extract is read once and
    cached on the instance.

    Parameters
    ----------
    root
        Directory holding the MIMIC-IV modules. ``hosp/``, ``icu/`` and,
        for the ED cohort, ``ed/`` are located automatically wherever they sit.
    cohort
        ``"ed"`` or ``"icu"`` -- see the module docstring. ``"auto"`` picks ED
        when MIMIC-IV-ED is present.
    limit_stays
        Keep only the first N stays by admission time. Use it to iterate:
        the full cohort takes minutes, 2000 stays takes seconds.
    outcomes
        Which events define deterioration.
    min_observations
        Stays with fewer charted vitals than this are dropped -- a stay with
        two readings produces no usable trend features and only adds noise.
    """

    def __init__(
        self,
        root: str | pathlib.Path,
        cohort: str = "auto",
        limit_stays: int | None = None,
        outcomes: OutcomeSpec | None = None,
        min_observations: int = 3,
        verbose: bool = True,
    ) -> None:
        self.root = pathlib.Path(root).expanduser()
        self.cohort_mode = cohort
        self.limit_stays = limit_stays
        self.outcomes = outcomes or OutcomeSpec()
        self.min_observations = min_observations
        self.verbose = verbose

        self._patients: pd.DataFrame | None = None
        self._chartevents: pd.DataFrame | None = None
        self.qc = QCReport()
        self.label_report = ""
        self.tables: dict[str, pathlib.Path] = {}

    # -- interfaces --------------------------------------------------------

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """:class:`api.sources.CohortSource` interface."""
        self._ensure()
        return self._patients, self._chartevents

    def patients(self) -> pd.DataFrame:
        """Trainer interface."""
        self._ensure()
        return self._patients

    def chartevents(self) -> pd.DataFrame:
        """Trainer interface."""
        self._ensure()
        return self._chartevents

    # -- plumbing ----------------------------------------------------------

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _ensure(self) -> None:
        if self._patients is None:
            self._build()

    def _resolve_mode(self) -> str:
        self.tables = available(self.root)
        if not self.tables:
            raise FileNotFoundError(
                f"no MIMIC-IV tables found under {self.root}. Expected something like "
                f"{self.root}/hosp/patients.csv.gz and {self.root}/icu/chartevents.csv.gz."
            )
        has_ed = {"edstays", "triage", "vitalsign"} <= set(self.tables)
        if self.cohort_mode == "ed" and not has_ed:
            missing = {"edstays", "triage", "vitalsign"} - set(self.tables)
            raise FileNotFoundError(
                f"cohort='ed' needs MIMIC-IV-ED; missing {sorted(missing)} under "
                f"{self.root}. MIMIC-IV-ED is a separate PhysioNet download from "
                f"MIMIC-IV -- same credentialed account, different project page. "
                f"Use cohort='icu' to run on core MIMIC-IV alone, but read the "
                f"caveats in api/mimic.py first: it is not an ED triage cohort."
            )
        if self.cohort_mode == "auto":
            return "ed" if has_ed else "icu"
        return self.cohort_mode

    def _build(self) -> None:
        mode = self._resolve_mode()
        self._say(f"\nMIMIC-IV extract: {self.root}")
        self._say(f"cohort mode     : {mode}")
        self._say("tables found    : " + ", ".join(sorted(self.tables)))

        stays = self._load_ed_stays() if mode == "ed" else self._load_icu_stays()
        if self.limit_stays:
            stays = stays.nsmallest(self.limit_stays, "intime").reset_index(drop=True)
        self._say(f"\nstays           : {len(stays):,}")

        vitals = (
            self._load_ed_vitals(stays) if mode == "ed" else self._load_icu_vitals(stays)
        )
        vitals = normalise_vitals(vitals, self.qc)
        self._say("\n" + self.qc.render())

        # Stays with too little charting cannot produce trend features.
        counts = vitals.groupby("stay_id").size()
        usable = counts[counts >= self.min_observations].index
        dropped = len(stays) - len(set(stays["stay_id"]) & set(usable))
        stays = stays.loc[stays["stay_id"].isin(usable)].reset_index(drop=True)
        vitals = vitals.loc[vitals["stay_id"].isin(stays["stay_id"])]
        self._say(f"\ndropped {dropped:,} stay(s) with fewer than "
                  f"{self.min_observations} usable observations")

        # The label clock starts at the first charted vital, matching the
        # origin ml.features uses for t. Anything else and time_to_onset is
        # measured against a different zero from the features.
        window_start = (
            vitals.groupby("stay_id")["charttime"].min().rename("window_start")
        )
        stays = stays.merge(window_start, on="stay_id", how="left")

        labelled = self._label(stays, mode)
        self._patients = self._finalise_patients(labelled)
        assert "subject_id" in vitals.columns, "both vitals paths must carry subject_id"
        self._chartevents = to_canonical_chartevents(vitals)
        self._say(f"\nchartevents     : {len(self._chartevents):,} rows")
        self._say(f"patients        : {len(self._patients):,} rows")

    # -- cohort: ED --------------------------------------------------------

    def _load_ed_stays(self) -> pd.DataFrame:
        edstays = read_table(
            self.tables["edstays"],
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "disposition"],
            parse_dates=["intime", "outtime"],
        )
        triage = read_table(
            self.tables["triage"], usecols=["subject_id", "stay_id", "acuity"]
        )
        pats = read_table(
            self.tables["patients"],
            usecols=["subject_id", "gender", "anchor_age"],
        )

        df = edstays.merge(triage[["stay_id", "acuity"]], on="stay_id", how="left")
        df = df.merge(pats, on="subject_id", how="left")

        before = len(df)
        df = df.loc[df["acuity"].notna()].copy()
        self._say(f"dropped {before - len(df):,} ED stay(s) with no triage acuity")

        df["esi"] = df["acuity"].astype(int)
        df["age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
        df["sex"] = df["gender"].astype(str).str.upper().str[0]
        # disposition is the ground truth for the admission model.
        df["will_admit"] = (
            df["disposition"].astype(str).str.upper().str.startswith("ADMITTED")
        )
        return df[
            ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
             "esi", "age", "sex", "will_admit"]
        ].reset_index(drop=True)

    def _load_ed_vitals(self, stays: pd.DataFrame) -> pd.DataFrame:
        """ED vitalsign is already wide; melt it and tag units."""
        cols = ["subject_id", "stay_id", "charttime"] + list(ED_VITALSIGN_COLUMNS)
        wide = read_table(self.tables["vitalsign"], usecols=cols,
                          parse_dates=["charttime"])
        wide = wide.loc[wide["stay_id"].isin(stays["stay_id"])]

        long = wide.melt(
            id_vars=["subject_id", "stay_id", "charttime"],
            value_vars=list(ED_VITALSIGN_COLUMNS),
            var_name="_col",
            value_name="value",
        )
        mapped = long["_col"].map(ED_VITALSIGN_COLUMNS)
        long["channel"] = [m[0] for m in mapped]
        long["source_unit"] = [m[1] for m in mapped]
        # vitalsign has no itemids; label with the canonical one so the
        # priority dedup has something to work with.
        long["source_itemid"] = long["channel"].map(
            {c.channel: c.canonical_itemid for c in VITAL_CONCEPTS}
        )
        return long.drop(columns="_col")

    # -- cohort: ICU -------------------------------------------------------

    def _load_icu_stays(self) -> pd.DataFrame:
        icustays = read_table(
            self.tables["icustays"],
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
            parse_dates=["intime", "outtime"],
        )
        pats = read_table(
            self.tables["patients"], usecols=["subject_id", "gender", "anchor_age"]
        )
        df = icustays.merge(pats, on="subject_id", how="left")
        df["age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
        df["sex"] = df["gender"].astype(str).str.upper().str[0]
        # No triage in an ICU cohort. NaN rather than a made-up constant: the
        # model treats it as missing, and nobody can mistake it for a real ESI.
        df["esi"] = np.nan
        df["will_admit"] = True  # already admitted; the bed model is moot here
        return df[
            ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
             "esi", "age", "sex", "will_admit"]
        ].reset_index(drop=True)

    def _load_icu_vitals(self, stays: pd.DataFrame) -> pd.DataFrame:
        """Stream chartevents, keeping only our itemids and our stays."""
        wanted_items = set(SOURCE_ITEMID_TO_CHANNEL)
        wanted_stays = set(stays["stay_id"].astype("int64"))
        self._say(f"\nstreaming chartevents for {len(wanted_items)} itemids "
                  f"across {len(wanted_stays):,} stays...")

        def keep(chunk: pd.DataFrame) -> pd.DataFrame:
            return chunk.loc[
                chunk["itemid"].isin(wanted_items) & chunk["stay_id"].isin(wanted_stays)
            ]

        ev = read_filtered(
            self.tables["chartevents"],
            keep=keep,
            usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
            parse_dates=["charttime"],
            dtype={"itemid": "int32", "stay_id": "int64"},
            label="chartevents",
            log=(lambda m: self._say(m)) if self.verbose else None,
        )

        ev["channel"] = ev["itemid"].map(SOURCE_ITEMID_TO_CHANNEL)
        ev["source_itemid"] = ev["itemid"]
        ev["source_unit"] = ev["itemid"].map(
            {it.itemid: it.unit for c in VITAL_CONCEPTS for it in c.items}
        )
        return ev.rename(columns={"valuenum": "value"})[
            ["subject_id", "stay_id", "charttime", "channel", "value",
             "source_itemid", "source_unit"]
        ]

    # -- labels ------------------------------------------------------------

    def _read_optional(self, name: str, **kw: Any) -> pd.DataFrame | None:
        path = self.tables.get(name) or find_table(self.root, name)
        if path is None:
            self._say(f"  {name}: not found -- that component is unavailable")
            return None
        return read_table(path, **kw)

    def _label(self, stays: pd.DataFrame, mode: str) -> pd.DataFrame:
        self._say("\nbuilding outcome labels...")
        spec = self.outcomes

        icustays = None
        if spec.icu_transfer:
            if mode == "icu":
                self._say("  icu_transfer: disabled -- this cohort is already in the ICU")
                spec = replace(spec, icu_transfer=False)
            else:
                icustays = self._read_optional(
                    "icustays", usecols=["subject_id", "hadm_id", "stay_id", "intime"],
                    parse_dates=["intime"],
                )

        inputevents = None
        if spec.vasopressor:
            inputevents = self._read_optional(
                "inputevents",
                usecols=["subject_id", "hadm_id", "stay_id", "starttime", "itemid"],
                parse_dates=["starttime"],
            )

        procedureevents = None
        if spec.ventilation or spec.rapid_response:
            procedureevents = self._read_optional(
                "procedureevents",
                usecols=["subject_id", "hadm_id", "stay_id", "starttime", "itemid"],
                parse_dates=["starttime"],
            )

        admissions = None
        if spec.death:
            admissions = self._read_optional(
                "admissions", usecols=["subject_id", "hadm_id", "deathtime"],
                parse_dates=["deathtime"],
            )

        labelled, events = build_labels(
            stays, spec,
            icustays=icustays, inputevents=inputevents,
            procedureevents=procedureevents, admissions=admissions,
        )
        self.label_report = summarise(labelled, events, spec)
        self._say("\n" + self.label_report)
        return labelled

    def _finalise_patients(self, labelled: pd.DataFrame) -> pd.DataFrame:
        """Exactly the columns sim.simulator emits, same names, same meanings."""
        out = labelled.copy()
        out["hadm_id"] = out["hadm_id"].fillna(-1).astype("int64")
        keep = [
            "subject_id", "hadm_id", "stay_id", "age", "sex", "esi",
            "will_admit", "deteriorates", "onset_minutes", "intime",
            "onset_source", "onset_already_underway",
        ]
        return out[[c for c in keep if c in out.columns]].reset_index(drop=True)

    # -- reporting ---------------------------------------------------------

    def provenance(self) -> str:
        """Everything a reviewer needs to reproduce this cohort."""
        self._ensure()
        p = self._patients
        lines = [
            "COHORT PROVENANCE",
            "-" * 17,
            f"root              {self.root}",
            f"cohort mode       {self.cohort_mode}",
            f"limit_stays       {self.limit_stays or 'none'}",
            f"min_observations  {self.min_observations}",
            f"outcome label     {self.outcomes.describe()}",
            "",
            "tables read:",
        ]
        for name, path in sorted(self.tables.items()):
            lines.append(f"  {name:<16}{path}")
        lines += [
            "",
            f"stays             {len(p):,}",
            f"patients          {p['subject_id'].nunique():,}",
            f"deteriorating     {int(p['deteriorates'].sum()):,} "
            f"({p['deteriorates'].mean():.2%})",
        ]
        if "will_admit" in p.columns:
            lines.append(f"admitted          {int(p['will_admit'].sum()):,} "
                         f"({p['will_admit'].mean():.2%})")
        if p["esi"].notna().any():
            lines.append("")
            lines.append("deterioration rate by ESI:")
            g = p.groupby("esi")["deteriorates"].agg(["size", "sum", "mean"])
            for esi, row in g.iterrows():
                lines.append(f"  ESI {int(esi)}  n={int(row['size']):>7,}  "
                             f"events={int(row['sum']):>5,}  rate={row['mean']:.2%}")
        return "\n".join(lines)
