"""In-memory ED census.

One process, one board. No database -- the whole census is a few hundred
patients and is rebuilt from the source on restart.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from api import config as C
from api.interventions import Acknowledgement, Override, board_band
from ml.features import VITALS


def _clean(value: Any) -> Any:
    """NaN and numpy scalars are not JSON. Normalise on the way out."""
    if value is None:
        return None
    if isinstance(value, (bool,)):
        return bool(value)
    if isinstance(value, (int,)):
        return int(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return None if math.isnan(f) or math.isinf(f) else f


@dataclass
class PatientState:
    """Everything the board knows about one patient right now."""

    subject_id: int
    stay_id: int
    age: int
    sex: str
    esi: int
    intime: pd.Timestamp

    # Census lifecycle. "waiting" while the stay is active in the ED,
    # "departed" once charting has stopped.
    status: str = "waiting"

    minutes_in_dept: float = 0.0
    risk: float = float("nan")
    news2: float = float("nan")
    implied_esi: int = 5
    direction: str = "match"

    # Bed demand. None when no admission model has been trained.
    admit_risk: float | None = None
    admit_flag: bool = False

    # What the vitals say right now, independent of any model.
    instability: str = "unknown"
    clinical_state: str = "stable"
    # False once the patient is overtly unstable: the deterioration model was
    # trained on pre-onset rows only, so its score is extrapolation there.
    model_applicable: bool = True

    alert: bool = False
    retriage_flag: bool = False
    _candidate_since: float | None = None
    _clear_since: float | None = None
    retriage_since_min: float | None = None

    vitals: dict[str, float] = field(default_factory=dict)
    risk_history: deque = field(
        default_factory=lambda: deque(maxlen=C.RISK_HISTORY_POINTS)
    )

    # Top channels driving the current score, already phrased for a clinician
    # by api.reasons. Filled in the tick loop and memoised on the patient's
    # grid index, so reading it here is free -- see ReplayEngine._explain_active.
    # Board rows deliberately do not carry this: the board is built to be read
    # from three metres, and three sentences a row is not that. It travels on
    # the detail payload, which is what the panel opens.
    reasons: list[dict[str, Any]] = field(default_factory=list)
    # The same, for the bed call. Separate model, separate explanation -- the
    # features that predict deterioration are not the ones that predict
    # admission, and collapsing them would let the panel attribute a bed call
    # to a reason the admission model never used.
    admit_reasons: list[dict[str, Any]] = field(default_factory=list)

    # -- human decisions ---------------------------------------------------
    # Both survive re-scoring. The engine recomputes risk, NEWS2, state and
    # the re-triage flag on every tick and must not touch either of these:
    # a score is an input to a decision, not a repeal of one.
    override: Override | None = None
    ack: Acknowledgement | None = None
    # Set to True for a patient injected from a scenario preset, so a demo
    # patient is identifiable in the record without behaving differently.
    injected: str | None = None
    # How many hand-charted observations this patient has received. Non-zero
    # means they are off their replay trajectory -- see
    # ReplayEngine.append_observation.
    manual_entries: int = 0

    # Hindsight, simulated cohorts only. Shown on the demo board so you can see
    # how early a flag fired. Never an input to anything.
    truth: dict[str, Any] = field(default_factory=dict)

    def update_retriage(self, candidate: bool, now_min: float) -> None:
        """Debounced flag, in simulated minutes: must persist to raise, must
        stay clear to drop."""
        if candidate:
            self._clear_since = None
            if self._candidate_since is None:
                self._candidate_since = now_min
            held = now_min - self._candidate_since
            if not self.retriage_flag and held >= C.RETRIAGE_SUSTAIN_MIN:
                self.retriage_flag = True
                self.retriage_since_min = now_min
        else:
            self._candidate_since = None
            if self._clear_since is None:
                self._clear_since = now_min
            if self.retriage_flag and now_min - self._clear_since >= C.RETRIAGE_CLEAR_MIN:
                self.retriage_flag = False
                self.retriage_since_min = None

    @property
    def effective_esi(self) -> int:
        """The acuity the board is working to: the nurse's, if they set one.

        The assigned ESI stays visible beside it. The point of an override is
        that two parties disagree, and collapsing that into one number loses
        the only interesting thing about it.
        """
        return self.override.target_esi if self.override else int(self.esi)

    @property
    def escalating(self) -> bool:
        """Should this flag be shouting right now?

        A raised flag escalates unless someone has acknowledged it. That is
        the whole alert-fatigue mechanism: the flag, the record and the score
        all continue; only the shouting stops.
        """
        return self.retriage_flag and self.ack is None

    def apply_ack_policy(self) -> str | None:
        """Re-raise an acknowledgement the score has outgrown.

        Called once per re-score, from the same place for replayed and
        manually-entered vitals. Returns an audit action name when something
        changed, so the caller can record it without re-deriving why.
        """
        if self.ack is None:
            return None

        # The flag cleared on its own: the acknowledgement was about a
        # situation that no longer exists, so it should not silently carry
        # over to a future flag raised for a different reason.
        if C.ACK_CLEARS_WITH_FLAG and not self.retriage_flag:
            self.ack = None
            return "ack_expired"

        if self.ack.should_reescalate(self.risk):
            self.ack.reescalated = True
            self.ack = None
            return "ack_reescalated"
        return None

    def risk_delta(self, lookback_min: float = C.TREND_LOOKBACK_MIN) -> float | None:
        """Change in risk over the last ``lookback_min`` simulated minutes.

        Computed here rather than in the browser so the arrow survives a
        reconnect or a scrub, neither of which preserves client-side history.
        """
        if len(self.risk_history) < 2:
            return None
        target = self.minutes_in_dept - lookback_min
        past = [(m, r) for m, r in self.risk_history if m <= target]
        if not past:
            past = [self.risk_history[0]]
        return float(self.risk - past[-1][1])

    def to_row(self) -> dict[str, Any]:
        """Compact form for the board."""
        return {
            "subject_id": int(self.subject_id),
            "stay_id": int(self.stay_id),
            "age": int(self.age),
            "sex": str(self.sex),
            "esi": int(self.esi),
            "status": self.status,
            "minutes_in_dept": _clean(self.minutes_in_dept),
            "risk": _clean(self.risk),
            "risk_delta": _clean(self.risk_delta()),
            "news2": _clean(self.news2),
            "implied_esi": int(self.implied_esi),
            "direction": self.direction,
            "admit_risk": _clean(self.admit_risk),
            "admit_flag": bool(self.admit_flag),
            "instability": self.instability,
            "clinical_state": self.clinical_state,
            "model_applicable": bool(self.model_applicable),
            "alert": bool(self.alert),
            "retriage_flag": bool(self.retriage_flag),
            "retriage_since_min": _clean(self.retriage_since_min),
            # A flag that is raised but acknowledged: still true, no longer
            # shouting. The board needs both, so it can show the flag without
            # the pulse.
            "escalating": bool(self.escalating),
            "effective_esi": int(self.effective_esi),
            "override": self.override.payload() if self.override else None,
            "ack": self.ack.payload() if self.ack else None,
            "injected": self.injected,
            "manual_entries": int(self.manual_entries),
            "vitals": {k: _clean(v) for k, v in self.vitals.items()},
            "truth": {k: _clean(v) for k, v in self.truth.items()} or None,
        }

    def to_detail(self) -> dict[str, Any]:
        row = self.to_row()
        row["intime"] = self.intime.isoformat()
        row["reasons"] = list(self.reasons)
        row["admit_reasons"] = list(self.admit_reasons)
        row["risk_history"] = [
            {"minutes": _clean(m), "risk": _clean(r)} for m, r in self.risk_history
        ]
        return row


class Census:
    """The live board: active patients plus those who have left."""

    def __init__(self) -> None:
        self.active: dict[int, PatientState] = {}
        self.departed: dict[int, PatientState] = {}

    def get(self, subject_id: int) -> PatientState | None:
        for pool in (self.active, self.departed):
            for p in pool.values():
                if p.subject_id == subject_id:
                    return p
        return None

    def admit_to_board(self, p: PatientState) -> None:
        self.active[p.stay_id] = p

    def discharge(self, stay_id: int) -> None:
        """Move a stay off the live board.

        Model-derived flags are cleared -- they describe a patient who is no
        longer here. The override and the acknowledgement are NOT: they are
        the record of what a person decided, and "what did we do at the time"
        is asked long after the patient has gone.
        """
        p = self.active.pop(stay_id, None)
        if p is not None:
            p.status = "departed"
            p.alert = False
            p.retriage_flag = False
            self.departed[stay_id] = p

    def board(self) -> list[dict[str, Any]]:
        """Active patients, sickest first. That ordering is the product.

        A manual override moves a patient by BAND rather than by editing
        their score: escalated patients sort above the automatic ranking,
        de-escalated ones below it, and within each band the model's ordering
        still applies. The score on the row is always the model's real number,
        so a nurse can see exactly what they overruled.
        """
        pairs = [(p, p.to_row()) for p in self.active.values()]
        pairs.sort(
            key=lambda pr: (
                board_band(pr[0].override),
                pr[1]["risk"] is None,
                -(pr[1]["risk"] or 0.0),
            )
        )
        return [row for _, row in pairs]

    def summary(self) -> dict[str, Any]:
        rows = self.board()
        states = ("unstable_now", "early_warning", "watch", "news2_elevated", "stable")
        # Expected admissions is the sum of probabilities, not a count over a
        # threshold: for planning how many beds you need, the expectation is
        # the better estimator and does not depend on the operating point.
        expected = sum(r["admit_risk"] or 0.0 for r in rows)
        return {
            "census": len(rows),
            "capacity": C.ED_CAPACITY,
            "occupancy_pct": 100.0 * len(rows) / C.ED_CAPACITY if C.ED_CAPACITY else None,
            "expected_admissions": round(expected, 1),
            "alerts": sum(1 for r in rows if r["alert"]),
            "retriage_flags": sum(1 for r in rows if r["retriage_flag"]),
            "escalating": sum(1 for r in rows if r["escalating"]),
            "acknowledged": sum(1 for r in rows if r["ack"]),
            "overrides": sum(1 for r in rows if r["override"]),
            "predicted_admits": sum(1 for r in rows if r["admit_flag"]),
            "by_state": {s_: sum(1 for r in rows if r["clinical_state"] == s_) for s_ in states},
            "departed": len(self.departed),
            "by_esi": {
                str(e): sum(1 for r in rows if r["esi"] == e) for e in range(1, 6)
            },
        }


def vitals_snapshot(window: pd.DataFrame) -> dict[str, float]:
    """Latest non-null value for each vital in the patient's window."""
    out: dict[str, float] = {}
    for v in VITALS:
        if v in window.columns:
            series = window[v].dropna()
            out[v] = float(series.iloc[-1]) if len(series) else float("nan")
    return out
