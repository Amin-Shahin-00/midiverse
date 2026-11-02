"""Vitals -> feature vector.

Two entry points, deliberately kept in lockstep:

* :func:`compute_features` -- one rolling window, one feature vector. This is
  what the live scorer will call per patient per tick.
* :func:`build_feature_frame` -- the same features for every timestep of every
  patient at once, vectorised. This is what training uses.

They are separate implementations of the same maths -- the live path stays in
numpy because pandas overhead dominates on a 13-row window -- so
:func:`check_paths_agree` asserts they produce identical vectors. Run it after
touching either one; silent drift between them means the served model sees
different inputs from the trained one.

Everything here takes MIMIC-IV ``chartevents`` as input and knows nothing about
the simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VITALS: tuple[str, ...] = ("hr", "rr", "spo2", "sbp", "dbp", "temp")

# MIMIC-IV itemid -> canonical vital name. Real extracts carry more synonyms
# (arterial line pressures, Fahrenheit temperature); extend this map, not the
# code below.
MIMIC_ITEMID_MAP: dict[int, str] = {
    220045: "hr",
    220210: "rr",
    220277: "spo2",
    220179: "sbp",
    220050: "sbp",   # Arterial Blood Pressure systolic
    220180: "dbp",
    220051: "dbp",   # Arterial Blood Pressure diastolic
    223762: "temp",  # Celsius
}

# Channels we track trajectory on. The derived ones matter as much as the raw
# ones -- shock index moving is often the earliest thing you can see.
CHANNELS: tuple[str, ...] = VITALS + ("shock_index", "pulse_pressure", "map", "news2")

# Lookbacks in minutes for deltas and slopes.
LOOKBACKS: tuple[int, ...] = (15, 30, 60)

# Window for dispersion features.
STAT_WINDOW_MIN = 60

GRID_MIN = 5  # charting grid; all lookbacks must be multiples of this

STATIC_FEATURES: tuple[str, ...] = ("age", "sex_male", "esi", "minutes_elapsed")


# ---------------------------------------------------------------------------
# NEWS2
# ---------------------------------------------------------------------------

def news2_score(
    hr,
    rr,
    spo2,
    sbp,
    temp,
    on_oxygen=False,
    alert=True,
):
    """Simplified NEWS2 (National Early Warning Score 2).

    Simplifications, all of them because the inputs do not exist in this data:
    the SpO2 scale-2 (hypercapnic) table is not used, supplemental oxygen
    defaults to room air, and ACVPU consciousness defaults to Alert. What is
    left is the six-parameter aggregate score, 0-20.

    Works elementwise on arrays or on plain scalars. NaN in, NaN out.
    """
    scalar = np.isscalar(hr)
    hr, rr, spo2, sbp, temp = (np.asarray(x, dtype=float) for x in (hr, rr, spo2, sbp, temp))

    nan = np.nan

    rr_s = np.select(
        [rr <= 8, rr <= 11, rr <= 20, rr <= 24, rr > 24],
        [3, 1, 0, 2, 3],
        default=nan,
    )
    spo2_s = np.select(
        [spo2 <= 91, spo2 <= 93, spo2 <= 95, spo2 > 95],
        [3, 2, 1, 0],
        default=nan,
    )
    sbp_s = np.select(
        [sbp <= 90, sbp <= 100, sbp <= 110, sbp <= 219, sbp > 219],
        [3, 2, 1, 0, 3],
        default=nan,
    )
    hr_s = np.select(
        [hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130, hr > 130],
        [3, 1, 0, 1, 2, 3],
        default=nan,
    )
    temp_s = np.select(
        [temp <= 35.0, temp <= 36.0, temp <= 38.0, temp <= 39.0, temp > 39.0],
        [3, 1, 0, 1, 2],
        default=nan,
    )

    total = rr_s + spo2_s + sbp_s + hr_s + temp_s
    total = total + np.where(np.asarray(on_oxygen, dtype=bool), 2, 0)
    total = total + np.where(np.asarray(alert, dtype=bool), 0, 3)

    return float(total) if scalar else total


# ---------------------------------------------------------------------------
# chartevents -> a regular wide grid
# ---------------------------------------------------------------------------

def chartevents_to_wide(
    chartevents: pd.DataFrame,
    itemid_map: dict[int, str] | None = None,
    grid_min: int = GRID_MIN,
    ffill_limit: int = 3,
) -> pd.DataFrame:
    """Pivot long chartevents onto a regular per-stay time grid.

    Returns one row per (stay_id, minutes-since-first-observation) with a
    column per vital. Gaps are carried forward for at most ``ffill_limit``
    steps; beyond that the value stays NaN, because a 40-minute-old blood
    pressure is not a current blood pressure.
    """
    itemid_map = itemid_map or MIMIC_ITEMID_MAP

    ev = chartevents.loc[chartevents["itemid"].isin(itemid_map)].copy()
    ev["vital"] = ev["itemid"].map(itemid_map)

    first = ev.groupby("stay_id")["charttime"].transform("min")
    ev["minutes"] = (ev["charttime"] - first).dt.total_seconds() / 60.0
    ev["t"] = (ev["minutes"] / grid_min).round().astype(int) * grid_min

    wide = (
        ev.pivot_table(index=["stay_id", "t"], columns="vital", values="valuenum", aggfunc="last")
        .reset_index()
        .rename_axis(columns=None)
    )
    for v in VITALS:
        if v not in wide.columns:
            wide[v] = np.nan

    # Reindex each stay onto the complete grid so shifts are true time offsets
    # rather than "k observations ago".
    tmax = wide.groupby("stay_id")["t"].max()
    counts = (tmax // grid_min + 1).to_numpy()
    full = pd.DataFrame(
        {
            "stay_id": np.repeat(tmax.index.to_numpy(), counts),
            "t": np.concatenate([np.arange(c) for c in counts]) * grid_min,
        }
    )
    grid = full.merge(wide, on=["stay_id", "t"], how="left")
    grid[list(VITALS)] = grid.groupby("stay_id")[list(VITALS)].ffill(limit=ffill_limit)

    intime = ev.groupby("stay_id")["charttime"].min().rename("intime")
    grid = grid.merge(intime, on="stay_id", how="left")
    grid["charttime"] = grid["intime"] + pd.to_timedelta(grid["t"], unit="m")

    return grid.sort_values(["stay_id", "t"], ignore_index=True)


# ---------------------------------------------------------------------------
# derived channels
# ---------------------------------------------------------------------------

def add_derived_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Add the clinical derivations that both feature paths build on."""
    out = df.copy()
    out["shock_index"] = out["hr"] / out["sbp"]
    out["pulse_pressure"] = out["sbp"] - out["dbp"]
    out["map"] = out["dbp"] + (out["sbp"] - out["dbp"]) / 3.0
    out["news2"] = news2_score(
        out["hr"].to_numpy(),
        out["rr"].to_numpy(),
        out["spo2"].to_numpy(),
        out["sbp"].to_numpy(),
        out["temp"].to_numpy(),
    )
    return out


def feature_names() -> list[str]:
    """Canonical, ordered feature list. Both paths must produce exactly this."""
    names = list(STATIC_FEATURES)
    names += [f"{c}" for c in CHANNELS]
    for c in CHANNELS:
        for lb in LOOKBACKS:
            names += [f"{c}_d{lb}", f"{c}_slope{lb}"]
    for c in CHANNELS:
        names += [f"{c}_min{STAT_WINDOW_MIN}", f"{c}_max{STAT_WINDOW_MIN}", f"{c}_std{STAT_WINDOW_MIN}"]
    return names


# ---------------------------------------------------------------------------
# path 1: one window -> one vector (live scoring)
# ---------------------------------------------------------------------------

def compute_features(
    window: pd.DataFrame,
    age: float,
    sex: str,
    esi: int,
    grid_min: int = GRID_MIN,
) -> dict[str, float]:
    """Turn one patient's recent vitals window into a single feature vector.

    ``window`` is a time-ordered frame on the ``grid_min`` grid whose last row
    is "now" and which holds at least the columns in :data:`VITALS`. It should
    cover at least the longest lookback; shorter windows just yield NaN for the
    features they cannot support, which XGBoost handles natively.
    """
    # This runs once per active patient per tick, so it stays in numpy.
    # Going through pandas here costs ~6ms a call on a 13-row window, almost
    # all of it DataFrame construction overhead. The arithmetic below is
    # identical to add_derived_channels + build_feature_frame; the agreement
    # between the two paths is asserted in check_paths_agree().
    ch: dict[str, np.ndarray] = {v: window[v].to_numpy(dtype=float) for v in VITALS}
    hr, rr, spo2, sbp, dbp, temp = (ch[v] for v in VITALS)

    with np.errstate(divide="ignore", invalid="ignore"):
        ch["shock_index"] = hr / sbp
    ch["pulse_pressure"] = sbp - dbp
    ch["map"] = dbp + (sbp - dbp) / 3.0
    ch["news2"] = np.asarray(news2_score(hr, rr, spo2, sbp, temp), dtype=float)

    if "t" in window.columns:
        minutes = float(window["t"].to_numpy()[-1])
    else:
        minutes = float((len(window) - 1) * grid_min)

    feats: dict[str, float] = {
        "age": float(age),
        "sex_male": 1.0 if str(sex).upper().startswith("M") else 0.0,
        "esi": float(esi),
        "minutes_elapsed": minutes,
    }

    n_tail = STAT_WINDOW_MIN // grid_min + 1
    for c in CHANNELS:
        series = ch[c]
        feats[c] = float(series[-1])

        for lb in LOOKBACKS:
            k = lb // grid_min
            prior = series[-1 - k] if len(series) > k else np.nan
            delta = series[-1] - prior
            feats[f"{c}_d{lb}"] = delta
            feats[f"{c}_slope{lb}"] = delta / lb * 60.0

        tail = series[-n_tail:]
        finite = tail[~np.isnan(tail)]
        if finite.size >= 3:
            feats[f"{c}_min{STAT_WINDOW_MIN}"] = float(finite.min())
            feats[f"{c}_max{STAT_WINDOW_MIN}"] = float(finite.max())
            feats[f"{c}_std{STAT_WINDOW_MIN}"] = float(finite.std(ddof=1))
        else:
            for stat in ("min", "max", "std"):
                feats[f"{c}_{stat}{STAT_WINDOW_MIN}"] = np.nan

    # Canonical order, so callers can rely on the contract rather than on
    # dict insertion order.
    return {k: feats[k] for k in feature_names()}


# ---------------------------------------------------------------------------
# path 2: whole panel -> feature frame (training)
# ---------------------------------------------------------------------------

def build_feature_frame(
    grid: pd.DataFrame,
    patients: pd.DataFrame,
    grid_min: int = GRID_MIN,
) -> pd.DataFrame:
    """Features for every timestep of every stay.

    ``grid`` comes from :func:`chartevents_to_wide`; ``patients`` supplies the
    static columns (``stay_id``, ``subject_id``, ``age``, ``sex``, ``esi``).
    Returns the feature columns plus the keys needed to label and group rows.
    """
    df = add_derived_channels(grid)
    g = df.groupby("stay_id", sort=False)

    parts: dict[str, pd.Series] = {}

    for c in CHANNELS:
        for lb in LOOKBACKS:
            k = lb // grid_min
            delta = df[c] - g[c].shift(k)
            parts[f"{c}_d{lb}"] = delta
            parts[f"{c}_slope{lb}"] = delta / lb * 60.0

    win = STAT_WINDOW_MIN // grid_min + 1
    for c in CHANNELS:
        roll = g[c].rolling(win, min_periods=3)
        parts[f"{c}_min{STAT_WINDOW_MIN}"] = roll.min().reset_index(level=0, drop=True)
        parts[f"{c}_max{STAT_WINDOW_MIN}"] = roll.max().reset_index(level=0, drop=True)
        parts[f"{c}_std{STAT_WINDOW_MIN}"] = roll.std().reset_index(level=0, drop=True)

    out = pd.concat([df, pd.DataFrame(parts, index=df.index)], axis=1)

    static = patients[["stay_id", "subject_id", "age", "sex", "esi"]].copy()
    static["sex_male"] = (static["sex"].str.upper().str[0] == "M").astype(float)
    out = out.merge(static.drop(columns=["sex"]), on="stay_id", how="left")
    out["minutes_elapsed"] = out["t"].astype(float)

    keys = ["stay_id", "subject_id", "t", "charttime"]
    return out[keys + feature_names()]


# ---------------------------------------------------------------------------
# path agreement
# ---------------------------------------------------------------------------

def check_paths_agree(
    grid: pd.DataFrame,
    patients: pd.DataFrame,
    n_samples: int = 40,
    seed: int = 0,
    tol: float = 1e-9,
) -> int:
    """Assert the live path and the training path produce the same vector.

    ``compute_features`` (numpy, one window) and ``build_feature_frame``
    (pandas, whole panel) are separate implementations of the same maths, so
    they can drift apart silently and give the served model different inputs
    from the trained one. Returns the number of rows checked; raises on any
    disagreement.
    """
    panel = build_feature_frame(grid, patients)
    rng = np.random.default_rng(seed)
    static = patients.set_index("stay_id")
    names = feature_names()

    checked = 0
    for i in rng.choice(len(panel), size=min(n_samples, len(panel)), replace=False):
        row = panel.iloc[int(i)]
        stay_id, t = row["stay_id"], row["t"]
        window = grid[(grid["stay_id"] == stay_id) & (grid["t"] <= t)]
        p = static.loc[stay_id]
        live = compute_features(window, p["age"], p["sex"], p["esi"])

        for k in names:
            a, b = live[k], row[k]
            if pd.isna(a) and pd.isna(b):
                continue
            if not np.isclose(a, b, rtol=1e-6, atol=tol, equal_nan=True):
                raise AssertionError(
                    f"feature paths disagree on {k!r} at stay {stay_id} t={t}: "
                    f"live={a!r} panel={b!r}"
                )
        checked += 1
    return checked
