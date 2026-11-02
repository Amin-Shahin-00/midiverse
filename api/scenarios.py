"""Pre-built patients for driving a demo, in the cohort's own format.

Each preset emits **chartevents** -- the same long `stay_id, charttime,
itemid, valuenum` rows a real extract produces -- which the engine pivots with
``ml.features.chartevents_to_wide`` exactly as it pivots everything else. A
scenario patient is therefore not a special case anywhere downstream: it is
scored, explained, flagged and re-triaged by the same code as a replayed one,
and the only thing that makes it "injected" is that it arrived after startup.

The trajectories are scripted rather than sampled, because a demo has to do
the same thing twice. They are shaped from the same clinical priors the
simulator uses -- a rising shock index leading a falling pressure, respiratory
rate moving before saturation -- but they are illustrations, not evidence, and
nothing measured on them says anything about the model's real performance.

The interesting one is ``silent_deteriorator``: it starts inside every normal
range at an ESI 4, and drifts for forty minutes until the model is implying an
ESI 2. That gap is the entire product, and it is the thing to put on screen.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from api.config import NEWS2_MEDIUM
from ml.features import GRID_MIN, MIMIC_ITEMID_MAP

# Canonical itemid per vital, inverted from the map features.py already owns,
# so a scenario cannot drift from the contract the rest of the app reads.
_ITEMID = {}
for _itemid, _name in MIMIC_ITEMID_MAP.items():
    _ITEMID.setdefault(_name, _itemid)

# Injected stays get ids from a separate range so they are recognisable in a
# log and can never collide with a cohort's own.
_INJECTED_SUBJECT_BASE = 90_000_000
_INJECTED_STAY_BASE = 99_000_000
_counter = itertools.count()


@dataclass(frozen=True)
class Scenario:
    """One demo patient: who they are, and how their vitals move."""

    key: str
    label: str
    blurb: str                  # what to expect, shown on the button
    esi: int
    age: int
    sex: str
    duration_min: int           # how much trajectory to generate
    build: Callable[[np.ndarray], dict[str, np.ndarray]]

    def vitals(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        t = np.arange(0, self.duration_min + GRID_MIN, GRID_MIN, dtype=float)
        return t, self.build(t)


def _ramp(t: np.ndarray, start: float, end: float, begin: float, over: float) -> np.ndarray:
    """``start`` until ``begin``, then a straight line to ``end`` over ``over`` minutes."""
    frac = np.clip((t - begin) / max(over, 1e-9), 0.0, 1.0)
    return start + (end - start) * frac


# ---------------------------------------------------------------------------
# the three presets
# ---------------------------------------------------------------------------

def _septic_elderly(t: np.ndarray) -> dict[str, np.ndarray]:
    """ESI 2, already unwell, decompensating fast.

    Starts febrile and tachycardic and loses pressure over half an hour. NEWS2
    crosses its escalation bands early, so this patient should read
    ``unstable_now`` rather than ``early_warning`` -- the board is telling you
    to look at the person in front of you, and the forward-looking score is
    explicitly out of distribution there. That distinction is worth
    demonstrating deliberately.
    """
    return {
        "hr": _ramp(t, 104, 138, 0, 45),
        "rr": _ramp(t, 22, 32, 0, 45),
        "spo2": _ramp(t, 95, 89, 10, 40),
        "sbp": _ramp(t, 104, 78, 5, 45),
        "dbp": _ramp(t, 62, 44, 5, 45),
        "temp": _ramp(t, 38.6, 39.2, 0, 50),
    }


def _silent_deteriorator(t: np.ndarray) -> dict[str, np.ndarray]:
    """ESI 4, entirely normal on arrival, drifting for forty minutes.

    Every opening value sits inside its reference range, so triage was right
    to call this minor. Nothing here would trip a threshold alarm: the
    respiratory rate never leaves the twenties, the saturation stays above 92,
    the systolic never reaches the usual 90 mmHg trigger. What moves is the
    *shape* -- shock index climbing as pressure falls and heart rate rises,
    which is what the trend features are built to see.

    The intended outcome is a re-triage flag on a patient no rule-based system
    would have looked at twice.
    """
    return {
        "hr": _ramp(t, 82, 116, 5, 40),
        "rr": _ramp(t, 17, 26, 10, 35),
        "spo2": _ramp(t, 98, 93, 15, 35),
        "sbp": _ramp(t, 124, 97, 5, 40),
        "dbp": _ramp(t, 78, 60, 5, 40),
        "temp": _ramp(t, 37.2, 37.9, 0, 45),
    }


def _stable(t: np.ndarray) -> dict[str, np.ndarray]:
    """ESI 3, unremarkable throughout. The control.

    A demo that only ever shows deterioration proves nothing about false
    alarms. This patient exists so the board can be seen NOT reacting -- the
    small wobbles below are charting noise, not a trend, and the risk score
    should stay flat through all of it.
    """
    n = len(t)
    wobble = np.sin(t / 25.0)
    return {
        "hr": 78 + 4 * wobble,
        "rr": 16 + 1.2 * np.cos(t / 20.0),
        "spo2": np.full(n, 98.0) - 0.5 * (wobble > 0.8),
        "sbp": 126 + 5 * np.cos(t / 30.0),
        "dbp": 76 + 3 * wobble,
        "temp": 36.9 + 0.12 * wobble,
    }


SCENARIOS: dict[str, Scenario] = {
    s.key: s
    for s in (
        Scenario(
            key="septic_elderly",
            label="Septic elderly",
            blurb="ESI 2 · already unwell, decompensates over ~45 min",
            esi=2, age=81, sex="F", duration_min=90, build=_septic_elderly,
        ),
        Scenario(
            key="silent_deteriorator",
            label="Silent deteriorator",
            blurb="ESI 4 · normal vitals, drifts for 40 min, should trip re-triage",
            esi=4, age=67, sex="M", duration_min=90, build=_silent_deteriorator,
        ),
        Scenario(
            key="stable",
            label="Stable",
            blurb="ESI 3 · no decline — the control, board should stay quiet",
            esi=3, age=44, sex="F", duration_min=90, build=_stable,
        ),
    )
}


def catalogue() -> list[dict]:
    """What the scenario buttons render from."""
    return [
        {
            "key": s.key, "label": s.label, "blurb": s.blurb,
            "esi": s.esi, "age": s.age, "sex": s.sex,
            "duration_minutes": s.duration_min,
        }
        for s in SCENARIOS.values()
    ]


def build_chartevents(
    scenario: Scenario, intime: pd.Timestamp
) -> tuple[dict, pd.DataFrame]:
    """One scenario -> ``(static, chartevents)`` ready for the engine.

    ``chartevents`` is in the ordinary long format, so the engine pivots it
    with the same function it uses for a real cohort. Nothing downstream can
    tell this patient apart from a replayed one.
    """
    n = next(_counter)
    subject_id = _INJECTED_SUBJECT_BASE + n
    stay_id = _INJECTED_STAY_BASE + n

    t, channels = scenario.vitals()
    charttime = intime + pd.to_timedelta(t, unit="m")

    frames = []
    for name, values in channels.items():
        itemid = _ITEMID.get(name)
        if itemid is None:                      # channel features.py cannot read
            continue
        frames.append(
            pd.DataFrame(
                {
                    "subject_id": subject_id,
                    "stay_id": stay_id,
                    "charttime": charttime,
                    "itemid": itemid,
                    "valuenum": np.round(values.astype(float), 1),
                }
            )
        )

    chartevents = pd.concat(frames, ignore_index=True).sort_values(
        ["stay_id", "charttime", "itemid"], ignore_index=True
    )
    static = {
        "subject_id": subject_id,
        "stay_id": stay_id,
        "age": scenario.age,
        "sex": scenario.sex,
        "esi": scenario.esi,
        "scenario": scenario.key,
    }
    return static, chartevents


def selftest() -> int:
    """Check each preset does what its docstring claims, before a demo does."""
    from ml.features import chartevents_to_wide, news2_score

    checks = 0
    for key, s in SCENARIOS.items():
        static, ev = build_chartevents(s, pd.Timestamp("2026-01-01 08:00:00"))
        grid = chartevents_to_wide(ev)

        assert len(grid) > 1, f"{key}: no grid rows"
        assert static["esi"] == s.esi
        # every channel features.py needs must be present and fully charted
        for col in ("hr", "rr", "spo2", "sbp", "dbp", "temp"):
            assert col in grid.columns and grid[col].notna().all(), f"{key}: {col} gaps"
        assert int(grid["t"].diff().dropna().max()) == GRID_MIN, f"{key}: ragged grid"
        checks += 3

        news = news2_score(
            grid["hr"].to_numpy(), grid["rr"].to_numpy(), grid["spo2"].to_numpy(),
            grid["sbp"].to_numpy(), grid["temp"].to_numpy(),
        )
        first, last = grid.iloc[0], grid.iloc[-1]

        if key == "septic_elderly":
            assert news[0] >= NEWS2_MEDIUM, f"septic should start deranged, NEWS2={news[0]}"
            assert last["sbp"] < first["sbp"] - 20, "septic should lose pressure"
            checks += 2

        if key == "silent_deteriorator":
            # the whole point: nothing abnormal at the start
            assert 12 <= first["rr"] <= 20, f"rr starts abnormal: {first['rr']}"
            assert first["spo2"] >= 95, f"spo2 starts abnormal: {first['spo2']}"
            assert 100 <= first["sbp"] <= 140, f"sbp starts abnormal: {first['sbp']}"
            assert 60 <= first["hr"] <= 100, f"hr starts abnormal: {first['hr']}"
            assert news[0] <= 1, f"NEWS2 should start clean, got {news[0]}"
            # and it must never become a threshold alarm on its own
            assert grid["sbp"].min() > 90, "should not cross the usual SBP trigger"
            assert grid["spo2"].min() >= 92, "should not cross the usual SpO2 trigger"
            # but the shape must move
            si_first = first["hr"] / first["sbp"]
            si_last = last["hr"] / last["sbp"]
            assert si_last > si_first * 1.4, f"shock index barely moved: {si_first:.2f}->{si_last:.2f}"
            checks += 8

        if key == "stable":
            assert news.max() <= 1, f"stable patient should never derange, NEWS2 max {news.max()}"
            assert abs(last["hr"] - first["hr"]) < 12, "stable patient drifted"
            assert grid["sbp"].std() < 6, "stable patient's pressure is too noisy"
            checks += 3

    # ids must not collide with a cohort's
    s1, _ = build_chartevents(SCENARIOS["stable"], pd.Timestamp("2026-01-01"))
    s2, _ = build_chartevents(SCENARIOS["stable"], pd.Timestamp("2026-01-01"))
    assert s1["stay_id"] != s2["stay_id"], "injected stay_ids repeat"
    assert s1["stay_id"] >= _INJECTED_STAY_BASE
    checks += 2

    assert len(catalogue()) == 3
    checks += 1
    return checks



if __name__ == "__main__":  # pragma: no cover
    print(f"api.scenarios: {selftest()} checks passed")
