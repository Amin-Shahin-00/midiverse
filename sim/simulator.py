"""Synthetic ED patient generator emitting data in MIMIC-IV ``chartevents`` shape.

The point of this module is to be *disposable*. Everything downstream
(``ml.features``, ``ml.train``, the API) consumes a ``VitalsSource``: a pair of
tables, one row per patient and one row per charted observation, using MIMIC-IV
column names and itemids. When PhysioNet access comes through, drop in a
``MimicSource`` that reads the real ``chartevents`` and nothing downstream
changes.

Ground truth (deterioration flag, onset time, LOS, disposition) lives only in
the ``patients`` table and never appears in ``chartevents`` -- the same
separation you get with real data, where labels come from outcome tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# MIMIC-IV d_items identifiers for the vitals we chart.
# --------------------------------------------------------------------------
ITEMIDS: dict[str, int] = {
    "hr": 220045,    # Heart Rate
    "rr": 220210,    # Respiratory Rate
    "spo2": 220277,  # O2 saturation pulseoxymetry
    "sbp": 220179,   # Non Invasive Blood Pressure systolic
    "dbp": 220180,   # Non Invasive Blood Pressure diastolic
    "temp": 223762,  # Temperature Celsius
}

UOM: dict[str, str] = {
    "hr": "bpm",
    "rr": "insp/min",
    "spo2": "%",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "temp": "C",
}

VITALS: tuple[str, ...] = tuple(ITEMIDS)

SAMPLE_INTERVAL_MIN = 5

# --------------------------------------------------------------------------
# Epidemiology knobs
# --------------------------------------------------------------------------

# P(ESI = k) at triage. ESI 3 is the bulk of the waiting room.
ESI_MIX = {1: 0.02, 2: 0.15, 3: 0.45, 4: 0.30, 5: 0.08}

# Baseline P(deterioration) by acuity. ESI 4-5 is deliberately non-zero: the
# under-triaged patient left in the waiting room is the case this product
# exists to catch, so the simulator has to actually produce them.
ESI_DETERIORATION_RISK = {1: 0.40, 2: 0.26, 3: 0.13, 4: 0.060, 5: 0.035}

# P(admission) by acuity, before the deterioration bump.
ESI_ADMIT_RISK = {1: 0.95, 2: 0.72, 3: 0.38, 4: 0.12, 5: 0.04}

# Direction and magnitude of the physiologic insult at full severity, in
# native units (bpm, breaths/min, %, mmHg, C).
PHENOTYPES: dict[str, dict[str, float]] = {
    "sepsis":      {"hr": 38, "rr": 10, "spo2": -2.5, "sbp": -26, "dbp": -16, "temp": 1.30},
    "respiratory": {"hr": 22, "rr": 13, "spo2": -7.0, "sbp": -4,  "dbp": -2,  "temp": 0.30},
    "hemorrhage":  {"hr": 34, "rr": 5,  "spo2": -1.5, "sbp": -34, "dbp": -21, "temp": -0.30},
    "cardiac":     {"hr": 26, "rr": 8,  "spo2": -3.0, "sbp": -22, "dbp": -13, "temp": 0.00},
}
PHENOTYPE_MIX = {"sepsis": 0.40, "respiratory": 0.30, "hemorrhage": 0.20, "cardiac": 0.10}

# Measurement + short-term biological noise, per observation.
NOISE_SD = {"hr": 3.0, "rr": 1.2, "spo2": 0.8, "sbp": 5.0, "dbp": 4.0, "temp": 0.12}

# Physiologic limits -- monitors do not report impossible numbers.
LIMITS = {
    "hr": (25.0, 210.0),
    "rr": (5.0, 60.0),
    "spo2": (60.0, 100.0),
    "sbp": (55.0, 240.0),
    "dbp": (25.0, 140.0),
    "temp": (33.0, 42.0),
}

PATIENT_DROPOUT = 0.02   # whole timestep missing (patient off the monitor)
VITAL_DROPOUT = 0.03     # single vital missing (cuff failed, probe off)


class VitalsSource(Protocol):
    """The contract between a data source and everything downstream.

    ``patients()``    -> one row per ED stay, demographics plus (simulator
                         only) ground-truth outcome columns.
    ``chartevents()`` -> MIMIC-IV chartevents: long, one row per observation.
    """

    def patients(self) -> pd.DataFrame: ...

    def chartevents(self) -> pd.DataFrame: ...


@dataclass
class Cohort:
    patients: pd.DataFrame
    chartevents: pd.DataFrame


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class EDSimulator:
    """Generates a cohort of synthetic ED patients and their charted vitals."""

    def __init__(
        self,
        n_patients: int = 900,
        seed: int = 7,
        start: str = "2026-01-05 07:00:00",
    ) -> None:
        self.n_patients = n_patients
        self.rng = np.random.default_rng(seed)
        self.start = pd.Timestamp(start)

    # -- patient level ----------------------------------------------------

    def _draw_patients(self) -> pd.DataFrame:
        rng = self.rng
        n = self.n_patients

        age = np.clip(np.round(rng.normal(52, 20, n)), 18, 98).astype(int)
        sex = rng.choice(["M", "F"], size=n)
        esi = rng.choice(list(ESI_MIX), size=n, p=list(ESI_MIX.values()))

        # Acuity as a signed severity: +2 for ESI 1, -2 for ESI 5.
        sev = 3 - esi

        # P(deterioration) rises with age on top of the acuity baseline.
        base_risk = np.array([ESI_DETERIORATION_RISK[e] for e in esi])
        age_mult = 1.0 + 0.9 * _sigmoid((age - 65) / 9.0)
        p_deteriorate = np.clip(base_risk * age_mult, 0.01, 0.75)
        deteriorates = rng.random(n) < p_deteriorate

        # Disposition: acuity + age, and deterioration almost guarantees a bed.
        p_admit = np.clip(
            np.array([ESI_ADMIT_RISK[e] for e in esi]) + 0.25 * _sigmoid((age - 70) / 10.0),
            0.02,
            0.98,
        )
        p_admit = np.where(deteriorates, np.maximum(p_admit, 0.93), p_admit)
        will_admit = rng.random(n) < p_admit

        # Length of stay in minutes, snapped to the 5-minute charting grid.
        median_los = np.where(will_admit, 390.0, 215.0) * (1.0 + 0.10 * sev)
        los = np.exp(rng.normal(np.log(median_los), 0.40))
        los = np.clip(np.round(los / SAMPLE_INTERVAL_MIN) * SAMPLE_INTERVAL_MIN, 90, 960)

        # Deterioration onset. Leave room for observation before it (otherwise
        # there is nothing to predict from) and after it (otherwise the event
        # is never actually seen).
        los = np.where(deteriorates, np.maximum(los, 240), los)
        onset = np.full(n, np.nan)
        det_idx = np.flatnonzero(deteriorates)
        frac = rng.uniform(0.35, 0.80, det_idx.size)
        onset[det_idx] = (
            np.round((60 + frac * (los[det_idx] - 90)) / SAMPLE_INTERVAL_MIN)
            * SAMPLE_INTERVAL_MIN
        )

        # The prodrome: how long before the clinical event the numbers start
        # moving. This is the whole premise of the product -- if physiology
        # only changed at onset, nothing would be predictable in advance.
        #
        # It is drawn longer than the 120-minute prediction horizon on purpose.
        # A prodrome shorter than the horizon makes part of that horizon
        # unpredictable by construction, and the model would be penalised for
        # failing to see something the data does not contain. Real physiologic
        # derangement precedes clinical deterioration by hours, which is the
        # premise every early warning score rests on.
        prodrome = np.full(n, np.nan)
        prodrome[det_idx] = np.minimum(
            rng.uniform(90, 240, det_idx.size), onset[det_idx]
        )

        phenotype = np.array([None] * n, dtype=object)
        phenotype[det_idx] = rng.choice(
            list(PHENOTYPE_MIX), size=det_idx.size, p=list(PHENOTYPE_MIX.values())
        )
        amplitude = np.full(n, 0.0)
        amplitude[det_idx] = np.clip(rng.normal(1.0, 0.25, det_idx.size), 0.45, 1.70)

        subject_id = 10_000_000 + np.arange(n)
        arrival = self.start + pd.to_timedelta(
            np.sort(rng.uniform(0, 72 * 60, n)).round(), unit="m"
        )

        # Per-patient stable baselines.
        hr0 = rng.normal(78, 9, n) + 4.0 * sev + 0.04 * (age - 50)
        rr0 = rng.normal(16, 1.8, n) + 0.8 * np.maximum(sev, 0)
        spo20 = rng.normal(97.8, 1.0, n) - 0.6 * np.maximum(sev, 0) - 0.012 * (age - 50)
        sbp0 = rng.normal(126, 14, n) + 0.36 * (age - 50) - 3.0 * sev
        dbp0 = 0.62 * sbp0 + rng.normal(0, 5, n)
        temp0 = rng.normal(36.9, 0.28, n) + 0.18 * np.maximum(sev, 0)

        return pd.DataFrame(
            {
                "subject_id": subject_id,
                "hadm_id": np.where(will_admit, 20_000_000 + np.arange(n), -1),
                "stay_id": 30_000_000 + np.arange(n),
                "age": age,
                "sex": sex,
                "esi": esi,
                "intime": arrival,
                "los_minutes": los.astype(int),
                "will_admit": will_admit,
                "deteriorates": deteriorates,
                "onset_minutes": onset,
                "prodrome_minutes": prodrome,
                "phenotype": phenotype,
                "severity": amplitude,
                "hr0": hr0,
                "rr0": rr0,
                "spo20": spo20,
                "sbp0": sbp0,
                "dbp0": dbp0,
                "temp0": temp0,
            }
        )

    # -- trajectory -------------------------------------------------------

    def _severity_curve(self, t: np.ndarray, row: pd.Series) -> np.ndarray:
        """Fraction of the full insult expressed at each minute offset ``t``.

        Zero until the prodrome starts, a slow accelerating creep through the
        prodrome (reaching ~30% of full effect by clinical onset), then a much
        faster climb afterwards. The prodrome segment is the learnable part --
        vitals still inside normal range but moving.
        """
        if not row["deteriorates"]:
            return np.zeros_like(t, dtype=float)

        onset = float(row["onset_minutes"])
        prod = float(row["prodrome_minutes"])
        gain = 0.35  # severity already reached at the moment of clinical onset

        s = np.zeros_like(t, dtype=float)

        pre = (t >= onset - prod) & (t < onset)
        u = (t[pre] - (onset - prod)) / prod
        s[pre] = gain * u**1.8

        post = t >= onset
        v = np.clip((t[post] - onset) / 90.0, 0.0, 1.6)
        s[post] = gain + (1.0 - gain) * v

        return s

    def _wander(self, t: np.ndarray) -> np.ndarray:
        """Smooth low-frequency physiologic drift, unrelated to outcome.

        Without this, any non-zero slope means deterioration and the problem is
        trivial. Real patients wander.
        """
        rng = self.rng
        out = np.zeros_like(t, dtype=float)
        for _ in range(2):
            period = rng.uniform(70, 260)
            phase = rng.uniform(0, 2 * np.pi)
            out += np.sin(2 * np.pi * t / period + phase)
        return out / 2.0

    def _benign_excursion(self, t: np.ndarray) -> np.ndarray:
        """Transient bump from pain, anxiety, ambulation -- a decoy spike."""
        if self.rng.random() > 0.35 or t.size < 2:
            return np.zeros_like(t, dtype=float)
        centre = self.rng.uniform(t.min(), t.max())
        width = self.rng.uniform(12, 35)
        return np.exp(-0.5 * ((t - centre) / width) ** 2)

    def _vitals_for(self, row: pd.Series) -> pd.DataFrame:
        rng = self.rng
        t = np.arange(0, int(row["los_minutes"]) + 1, SAMPLE_INTERVAL_MIN, dtype=float)

        sev = self._severity_curve(t, row) * float(row["severity"])
        shift = PHENOTYPES[row["phenotype"]] if row["deteriorates"] else {}

        decoy = self._benign_excursion(t)
        decoy_shift = {"hr": 15.0, "rr": 3.5, "spo2": -0.8, "sbp": 7.0, "dbp": 4.0, "temp": 0.20}
        wander_scale = {"hr": 3.5, "rr": 1.0, "spo2": 0.5, "sbp": 5.0, "dbp": 3.5, "temp": 0.15}

        cols: dict[str, np.ndarray] = {}
        for v in VITALS:
            series = (
                float(row[f"{v}0"])
                + sev * shift.get(v, 0.0)
                + decoy * decoy_shift[v]
                + self._wander(t) * wander_scale[v]
                + rng.normal(0.0, NOISE_SD[v], t.size)
            )
            lo, hi = LIMITS[v]
            cols[v] = np.clip(series, lo, hi)

        frame = pd.DataFrame(cols)
        frame["charttime"] = row["intime"] + pd.to_timedelta(t, unit="m")

        # Missingness: whole timesteps first, then individual vitals.
        keep = rng.random(t.size) > PATIENT_DROPOUT
        keep[0] = True
        frame = frame.loc[keep].copy()
        for v in VITALS:
            frame.loc[rng.random(len(frame)) < VITAL_DROPOUT, v] = np.nan

        return frame

    # -- assembly ---------------------------------------------------------

    def generate(self) -> Cohort:
        patients = self._draw_patients()

        blocks: list[pd.DataFrame] = []
        for _, row in patients.iterrows():
            frame = self._vitals_for(row)
            long = frame.melt(
                id_vars=["charttime"],
                value_vars=list(VITALS),
                var_name="vital",
                value_name="valuenum",
            ).dropna(subset=["valuenum"])
            long["subject_id"] = row["subject_id"]
            long["hadm_id"] = row["hadm_id"]
            long["stay_id"] = row["stay_id"]
            blocks.append(long)

        events = pd.concat(blocks, ignore_index=True)

        # Round to the precision a monitor actually reports.
        events["valuenum"] = np.where(
            events["vital"].to_numpy() == "temp",
            events["valuenum"].round(1),
            events["valuenum"].round(0),
        )
        events["itemid"] = events["vital"].map(ITEMIDS).astype("int64")
        events["valueuom"] = events["vital"].map(UOM)
        events["value"] = events["valuenum"].astype(str)
        events["storetime"] = events["charttime"] + pd.to_timedelta(
            self.rng.integers(1, 5, len(events)), unit="m"
        )
        events["caregiver_id"] = self.rng.integers(80_000, 80_400, len(events))
        events["warning"] = 0

        events = events[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "caregiver_id",
                "charttime",
                "storetime",
                "itemid",
                "value",
                "valuenum",
                "valueuom",
                "warning",
            ]
        ].sort_values(["stay_id", "charttime", "itemid"], ignore_index=True)

        return Cohort(patients=patients, chartevents=events)


class SimulatedSource:
    """``VitalsSource`` backed by :class:`EDSimulator`."""

    def __init__(self, n_patients: int = 900, seed: int = 7) -> None:
        self._cohort = EDSimulator(n_patients=n_patients, seed=seed).generate()

    def patients(self) -> pd.DataFrame:
        return self._cohort.patients

    def chartevents(self) -> pd.DataFrame:
        return self._cohort.chartevents


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic ED cohort.")
    ap.add_argument("-n", "--n-patients", type=int, default=900)
    ap.add_argument("-s", "--seed", type=int, default=7)
    ap.add_argument("-o", "--out", default=None, help="directory to write CSVs into")
    args = ap.parse_args()

    cohort = EDSimulator(n_patients=args.n_patients, seed=args.seed).generate()
    p = cohort.patients

    print(f"patients      : {len(p)}")
    print(f"chartevents   : {len(cohort.chartevents):,}")
    print(f"deteriorating : {p['deteriorates'].sum()} ({p['deteriorates'].mean():.1%})")
    print(f"admitted      : {p['will_admit'].sum()} ({p['will_admit'].mean():.1%})")
    print("\ndeterioration rate by ESI:")
    print(p.groupby("esi")["deteriorates"].agg(["size", "sum", "mean"]).to_string())

    if args.out:
        import pathlib

        d = pathlib.Path(args.out)
        d.mkdir(parents=True, exist_ok=True)
        p.to_csv(d / "patients.csv", index=False)
        cohort.chartevents.to_csv(d / "chartevents.csv", index=False)
        print(f"\nwrote {d / 'patients.csv'} and {d / 'chartevents.csv'}")


if __name__ == "__main__":
    main()
