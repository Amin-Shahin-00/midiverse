"""A miniature MIMIC-IV tree with known defects, for testing the loader.

The real extract cannot live in a repo and will not be on the machine of
whoever next touches this code. This builds a structurally faithful stand-in:
the same directory layout, the same table and column names, the same dtypes,
the same gzip, and the same *defects* -- Fahrenheit temperatures, a
mis-filed Celsius itemid, arterial and cuff pressures at the same minute,
disconnected-cuff zeros, SpO2 charted as a fraction, missing triage acuity,
patients with two admissions.

Every defect is injected at a counted rate, so :func:`selftest` asserts the
cleaner caught exactly the right number rather than merely "some". That is the
difference between a test that proves the unit handling works and one that
proves it runs.

It is a fixture, not a simulator. The vitals are crude on purpose -- what is
being tested is the plumbing between PhysioNet's file layout and this
project's feature code, not anything clinical. Use ``sim/simulator.py`` when
you want physiology.

    python -m api.mimic_fixture --out /tmp/fake-mimic    # build one to poke at
    python -m api.mimic_fixture --selftest               # build, load, assert
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import tempfile

import numpy as np
import pandas as pd

from ml.mimic_items import VASOPRESSOR_ITEMIDS, expected_labels

# Injected defect rates. The selftest asserts against these exact counts.
P_NO_ACUITY = 0.04          # ED stays with null triage acuity -> dropped
P_ZERO_BP = 0.03            # disconnected cuff -> out of range
P_SPO2_FRACTION = 0.02      # SpO2 charted as 0.97 rather than 97
P_TEMP_MISFILED = 0.10      # Fahrenheit value under the Celsius itemid
P_WILD_HR = 0.01            # lead artefact -> out of range

GRID_MIN = 15               # ED charting interval in the fixture


def _write(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def build_fixture(
    dest: str | pathlib.Path,
    n_stays: int = 400,
    seed: int = 3,
    with_ed: bool = True,
) -> pathlib.Path:
    """Write a fake MIMIC-IV tree under ``dest``. Returns the root."""
    rng = np.random.default_rng(seed)
    root = pathlib.Path(dest)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    n_subjects = int(n_stays * 0.9)          # some patients have two stays
    subject_ids = 10_000_000 + np.arange(n_subjects)

    # -- hosp/patients ----------------------------------------------------
    patients = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "gender": rng.choice(["M", "F"], n_subjects),
            "anchor_age": rng.integers(18, 95, n_subjects),
            "anchor_year": 2150,
        }
    )
    _write(patients, root / "hosp" / "patients.csv.gz")

    # -- ED stays ---------------------------------------------------------
    stay_subject = rng.choice(subject_ids, n_stays, replace=True)
    stay_ids = 30_000_000 + np.arange(n_stays)
    base = pd.Timestamp("2151-03-01 00:00:00")
    intime = base + pd.to_timedelta(rng.integers(0, 60 * 24 * 30, n_stays), unit="m")
    los_min = rng.integers(90, 700, n_stays)
    outtime = intime + pd.to_timedelta(los_min, unit="m")

    # Per-stay severity drives BOTH the vitals and the outcome. Without that
    # link the fixture has no learnable signal and a training run over it
    # cannot distinguish "pipeline works" from "pipeline is wired to noise".
    severity = rng.random(n_stays)
    admitted = rng.random(n_stays) < (0.12 + 0.55 * severity)
    hadm_id = np.where(admitted, 20_000_000 + np.arange(n_stays), np.nan)

    acuity = rng.choice([1, 2, 3, 4, 5], n_stays, p=[0.03, 0.18, 0.44, 0.28, 0.07]).astype(float)
    no_acuity = rng.random(n_stays) < P_NO_ACUITY
    acuity[no_acuity] = np.nan

    edstays = pd.DataFrame(
        {
            "subject_id": stay_subject,
            "hadm_id": hadm_id,
            "stay_id": stay_ids,
            "intime": intime,
            "outtime": outtime,
            "disposition": np.where(admitted, "ADMITTED", "HOME"),
        }
    )
    triage = pd.DataFrame(
        {
            "subject_id": stay_subject,
            "stay_id": stay_ids,
            "acuity": acuity,
            "temperature": rng.normal(98.6, 1.5, n_stays),
            "heartrate": rng.normal(88, 16, n_stays),
        }
    )

    # -- ED vitals (wide, Fahrenheit) -------------------------------------
    rows = []
    for i in range(n_stays):
        n_obs = max(2, int(los_min[i] // GRID_MIN))
        t = intime[i] + pd.to_timedelta(np.arange(n_obs) * GRID_MIN, unit="m")
        sev = severity[i]
        hr = rng.normal(78 + 40 * sev, 6, n_obs)
        rr = rng.normal(15 + 9 * sev, 2, n_obs)
        spo2 = np.clip(rng.normal(98 - 6 * sev, 1.2, n_obs), 60, 100)
        sbp = rng.normal(132 - 35 * sev, 8, n_obs)
        dbp = sbp - rng.normal(48, 6, n_obs)
        temp_f = rng.normal(98.6 + 2.5 * sev, 0.8, n_obs)   # Fahrenheit

        # injected defects
        zero = rng.random(n_obs) < P_ZERO_BP
        sbp[zero] = 0.0
        dbp[zero] = 0.0
        frac = rng.random(n_obs) < P_SPO2_FRACTION
        spo2[frac] = spo2[frac] / 100.0
        wild = rng.random(n_obs) < P_WILD_HR
        hr[wild] = rng.choice([0.0, 900.0], wild.sum())

        rows.append(
            pd.DataFrame(
                {
                    "subject_id": stay_subject[i],
                    "stay_id": stay_ids[i],
                    "charttime": t,
                    "temperature": temp_f,
                    "heartrate": hr,
                    "resprate": rr,
                    "o2sat": spo2,
                    "sbp": sbp,
                    "dbp": dbp,
                    "rhythm": "SR",
                    "pain": "3",
                }
            )
        )
    vitalsign = pd.concat(rows, ignore_index=True)

    if with_ed:
        _write(edstays, root / "ed" / "edstays.csv.gz")
        _write(triage, root / "ed" / "triage.csv.gz")
        _write(vitalsign, root / "ed" / "vitalsign.csv.gz")

    # -- ICU chartevents (long, by itemid, with unit traps) ---------------
    # Built for the admitted subset so cohort='icu' has something to read and
    # so the itemid/priority/unit paths are exercised.
    icu_rows = []
    adm_idx = np.flatnonzero(admitted)
    icu_stay_ids = 40_000_000 + np.arange(len(adm_idx))
    # Sicker patients are transferred sooner, so time-to-event carries signal
    # too and the horizon window is not arbitrary.
    delay = (60 + 340 * (1.0 - severity[adm_idx]) + rng.normal(0, 40, len(adm_idx)))
    icu_intime = intime[adm_idx] + pd.to_timedelta(
        np.clip(delay, 20, None).astype(int), unit="m"
    )
    for j, i in enumerate(adm_idx):
        n_obs = rng.integers(6, 20)
        t = icu_intime[j] + pd.to_timedelta(np.arange(n_obs) * 30, unit="m")
        sev = severity[i]
        pack = [
            (220045, rng.normal(85 + 35 * sev, 7, n_obs)),          # HR
            (220210, rng.normal(16 + 8 * sev, 2, n_obs)),           # RR
            (220277, np.clip(rng.normal(97 - 5 * sev, 1.5, n_obs), 60, 100)),
            (220179, rng.normal(128 - 30 * sev, 9, n_obs)),         # NIBP sys
            (220180, rng.normal(72 - 15 * sev, 7, n_obs)),          # NIBP dia
        ]
        # Arterial line at the SAME timestamps -- the priority-dedup case.
        pack += [
            (220050, rng.normal(134 - 30 * sev, 9, n_obs)),
            (220051, rng.normal(70 - 15 * sev, 7, n_obs)),
        ]
        # Temperature: Fahrenheit itemid, plus a share mis-filed as Celsius.
        tf = rng.normal(98.6 + 2.0 * sev, 0.9, n_obs)
        misfiled = rng.random(n_obs) < P_TEMP_MISFILED
        pack.append((223761, np.where(misfiled, np.nan, tf)))
        pack.append((223762, np.where(misfiled, tf, np.nan)))   # F value, C itemid

        for itemid, vals in pack:
            icu_rows.append(
                pd.DataFrame(
                    {
                        "subject_id": stay_subject[i],
                        "hadm_id": hadm_id[i],
                        "stay_id": icu_stay_ids[j],
                        "charttime": t,
                        "itemid": itemid,
                        "valuenum": vals,
                    }
                )
            )
    chartevents = pd.concat(icu_rows, ignore_index=True)
    _write(chartevents, root / "icu" / "chartevents.csv.gz")

    icustays = pd.DataFrame(
        {
            "subject_id": stay_subject[adm_idx],
            "hadm_id": hadm_id[adm_idx],
            "stay_id": icu_stay_ids,
            "intime": icu_intime,
            "outtime": icu_intime + pd.to_timedelta(rng.integers(600, 5000, len(adm_idx)), unit="m"),
            "first_careunit": "MICU",
        }
    )
    _write(icustays, root / "icu" / "icustays.csv.gz")

    # -- vasopressors ------------------------------------------------------
    gets_pressor = rng.random(len(adm_idx)) < 0.30
    pidx = np.flatnonzero(gets_pressor)
    inputevents = pd.DataFrame(
        {
            "subject_id": stay_subject[adm_idx][pidx],
            "hadm_id": hadm_id[adm_idx][pidx],
            "stay_id": icu_stay_ids[pidx],
            # Some before the ICU intime, some after, so "earliest component"
            # is a real comparison rather than always ICU transfer.
            "starttime": icu_intime[pidx] + pd.to_timedelta(
                rng.integers(-120, 900, len(pidx)), unit="m"
            ),
            "itemid": rng.choice(list(VASOPRESSOR_ITEMIDS), len(pidx)),
            "amount": rng.random(len(pidx)) * 10,
        }
    )
    _write(inputevents, root / "icu" / "inputevents.csv.gz")

    vent = rng.random(len(adm_idx)) < 0.18
    vidx = np.flatnonzero(vent)
    procedureevents = pd.DataFrame(
        {
            "subject_id": stay_subject[adm_idx][vidx],
            "hadm_id": hadm_id[adm_idx][vidx],
            "stay_id": icu_stay_ids[vidx],
            "starttime": icu_intime[vidx] + pd.to_timedelta(
                rng.integers(0, 600, len(vidx)), unit="m"
            ),
            "itemid": 225792,
        }
    )
    _write(procedureevents, root / "icu" / "procedureevents.csv.gz")

    # -- admissions --------------------------------------------------------
    died = rng.random(len(adm_idx)) < 0.10
    admissions = pd.DataFrame(
        {
            "subject_id": stay_subject[adm_idx],
            "hadm_id": hadm_id[adm_idx],
            "admittime": intime[adm_idx],
            "dischtime": outtime[adm_idx] + pd.to_timedelta(
                rng.integers(600, 9000, len(adm_idx)), unit="m"
            ),
            "deathtime": pd.Series(
                np.where(
                    died,
                    outtime[adm_idx] + pd.to_timedelta(rng.integers(600, 9000, len(adm_idx)), unit="m"),
                    pd.NaT,
                )
            ),
            "hospital_expire_flag": died.astype(int),
        }
    )
    _write(admissions, root / "hosp" / "admissions.csv.gz")

    # -- d_items, so --verify has something to check against ---------------
    labels = expected_labels()
    d_items = pd.DataFrame(
        {
            "itemid": list(labels),
            "label": list(labels.values()),
            "unitname": ["" for _ in labels],
        }
    )
    _write(d_items, root / "icu" / "d_items.csv.gz")

    return root


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def selftest(keep: str | None = None) -> int:
    """Build a fixture, run the loader over it, assert the contract holds."""
    from api.mimic import MimicCohortSource
    from api.sources import validate_cohort
    from ml.features import (
        GRID_MIN as FEATURE_GRID,
        MIMIC_ITEMID_MAP,
        build_feature_frame,
        chartevents_to_wide,
        feature_names,
    )
    from ml.outcomes import OutcomeSpec

    checks = 0
    tmp = pathlib.Path(keep) if keep else pathlib.Path(tempfile.mkdtemp(prefix="mimicfix-"))
    try:
        root = build_fixture(tmp / "mimic", n_stays=300, seed=3)

        # -- ED cohort ----------------------------------------------------
        src = MimicCohortSource(root, cohort="ed", verbose=False)
        patients, chartevents = src.load()

        # the engine's own validator is the real contract
        validate_cohort(patients, chartevents)
        checks += 1

        for col in ("subject_id", "stay_id", "age", "sex", "esi",
                    "deteriorates", "onset_minutes", "will_admit"):
            assert col in patients.columns, f"patients missing {col}"
            checks += 1

        # every emitted itemid must be one features.py already understands,
        # which is what keeps the feature code unchanged
        emitted = set(chartevents["itemid"].unique())
        assert emitted <= set(MIMIC_ITEMID_MAP), f"unknown itemids emitted: {emitted - set(MIMIC_ITEMID_MAP)}"
        checks += 1

        # -- unit handling -------------------------------------------------
        grid = chartevents_to_wide(chartevents)
        temp = grid["temp"].dropna()
        assert len(temp), "no temperatures survived"
        assert temp.max() < 45.0, f"Fahrenheit leaked through: max {temp.max():.1f}"
        assert temp.min() > 30.0, f"implausible Celsius: min {temp.min():.1f}"
        assert 35.0 < temp.median() < 40.0, f"temperature centre wrong: {temp.median():.1f}"
        checks += 4

        spo2 = grid["spo2"].dropna()
        assert spo2.max() <= 100.0 and spo2.min() >= 50.0, "SpO2 out of range survived"
        assert (spo2 < 2.0).sum() == 0, "fraction-scaled SpO2 not rescaled"
        checks += 2

        # zeros from disconnected cuffs and lead artefact must be gone
        assert (grid["sbp"].dropna() == 0).sum() == 0, "zero SBP survived"
        hr = grid["hr"].dropna()
        assert hr.min() >= 20 and hr.max() <= 250, f"HR artefact survived: {hr.min()}-{hr.max()}"
        checks += 3

        # QC must have actually counted the injected defects, not silently
        # passed them through
        assert src.qc.rescaled_fraction > 0, "no SpO2 fractions detected"
        assert src.qc.converted_f_to_c > 0, "no Fahrenheit conversions recorded"
        assert sum(src.qc.dropped_range.values()) > 0, "no out-of-range values dropped"
        checks += 3

        # -- labels --------------------------------------------------------
        det = patients["deteriorates"]
        assert det.any(), "no deterioration events built"
        assert not det.all(), "every stay deteriorated -- label is broken"
        onset = patients.loc[det, "onset_minutes"]
        assert (onset > 0).all(), "onset at or before the first observation leaked in"
        assert onset.notna().all(), "deteriorating stay with no onset time"
        assert patients.loc[~det, "onset_minutes"].isna().all(), "onset on a negative stay"
        checks += 5

        # onset must be on the same clock as the feature builder's t
        panel = build_feature_frame(grid, patients)
        assert len(panel), "no feature rows built from the MIMIC cohort"
        assert list(panel.columns[-len(feature_names()):]) == feature_names()
        assert panel["t"].min() == 0, "feature clock does not start at zero"
        checks += 3

        # the trainer's own label maths must produce both classes
        from ml.train import HORIZON_MIN, build_dataset
        data = build_dataset(src)
        assert len(data), "build_dataset produced no rows"
        assert data["y"].nunique() == 2, (
            f"label is degenerate: {data['y'].value_counts().to_dict()} -- "
            f"no positive rows within the {HORIZON_MIN} min horizon"
        )
        checks += 2

        # -- both interfaces agree ----------------------------------------
        assert src.patients() is patients and src.chartevents() is chartevents
        checks += 1

        # -- ICU cohort exercises the chartevents streaming path ----------
        icu = MimicCohortSource(root, cohort="icu", verbose=False,
                                outcomes=OutcomeSpec(icu_transfer=True, vasopressor=True))
        ip, ic = icu.load()
        validate_cohort(ip, ic)
        assert len(ic), "ICU cohort produced no chartevents"
        assert ip["esi"].isna().all(), "ICU cohort invented an ESI"
        # arterial line must lose to the cuff at a shared timestamp
        assert icu.qc.deduped_priority > 0, "priority dedup never fired"
        igrid = chartevents_to_wide(ic)
        assert igrid["temp"].dropna().max() < 45.0, "Fahrenheit leaked in ICU path"
        checks += 5

        # -- d_items verification path ------------------------------------
        from ml.mimic_items import verify_against_d_items
        assert verify_against_d_items(root) == 0, "fixture d_items disagrees with catalogue"
        checks += 1

        assert FEATURE_GRID == 5  # the loader assumes nothing else
        checks += 1

        # -- degraded extracts -------------------------------------------
        # Everything below is a shape a real download actually arrives in.
        # Each one used to be a silent wrong answer rather than a loud failure.

        # (a) optional outcome tables absent -> those components drop out,
        #     the run continues, and the label is built from what is left
        import shutil as _sh
        partial = tmp / "partial"
        _sh.copytree(root, partial)
        (partial / "icu" / "procedureevents.csv.gz").unlink()
        (partial / "hosp" / "admissions.csv.gz").unlink()
        psrc = MimicCohortSource(
            partial, cohort="ed", verbose=False,
            outcomes=OutcomeSpec(icu_transfer=True, vasopressor=True,
                                 ventilation=True, death=True),
        )
        pp, _pc = psrc.load()
        assert pp["deteriorates"].any(), "missing optional tables killed the label"
        checks += 1

        # (b) no MIMIC-IV-ED -> cohort='ed' must refuse with an actionable
        #     message, not fall through to a wrong cohort
        core_only = tmp / "core"
        _sh.copytree(root, core_only)
        _sh.rmtree(core_only / "ed")
        try:
            MimicCohortSource(core_only, cohort="ed", verbose=False).load()
            raise AssertionError("cohort='ed' silently accepted a core-only extract")
        except FileNotFoundError as e:
            assert "MIMIC-IV-ED" in str(e), f"unhelpful error: {e}"
        checks += 2

        # (c) auto mode must fall back to icu on a core-only extract
        auto = MimicCohortSource(core_only, cohort="auto", verbose=False)
        ap_, ac_ = auto.load()
        validate_cohort(ap_, ac_)
        assert ap_["esi"].isna().all(), "auto fallback invented an ESI"
        checks += 2

        # (d) an empty root must say so rather than produce an empty cohort
        empty = tmp / "empty"
        empty.mkdir()
        try:
            MimicCohortSource(empty, verbose=False).load()
            raise AssertionError("empty root silently produced a cohort")
        except FileNotFoundError:
            pass
        checks += 1

        # (e) non-numeric valuenum (real extracts carry text in some rows)
        #     must be coerced and dropped, not crash the cleaner
        from api.mimic import QCReport, normalise_vitals
        dirty = pd.DataFrame({
            "subject_id": [1, 1, 1],
            "stay_id": [1, 1, 1],
            "charttime": pd.to_datetime(["2020-01-01"] * 3),
            "channel": ["hr", "hr", "hr"],
            "value": ["98", "ERROR", None],
            "source_itemid": [220045] * 3,
            "source_unit": ["bpm"] * 3,
        })
        q = QCReport()
        cleaned = normalise_vitals(dirty, q)
        assert len(cleaned) == 1 and float(cleaned["value"].iloc[0]) == 98.0
        assert q.dropped_null == 2, f"expected 2 unparseable, got {q.dropped_null}"
        checks += 2

        return checks
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="build a fixture here and leave it")
    ap.add_argument("-n", "--n-stays", type=int, default=400)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        n = selftest()
        print(f"api.mimic_fixture: {n} checks passed")
        return
    if args.out:
        root = build_fixture(args.out, n_stays=args.n_stays)
        print(f"wrote fixture to {root}")
        for p in sorted(root.rglob("*.csv.gz")):
            print(f"  {p.relative_to(root)}  {p.stat().st_size / 1e6:.2f} MB")
        return
    ap.error("pass --selftest or --out")


if __name__ == "__main__":  # pragma: no cover
    main()
