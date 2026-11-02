"""Replay clock and the per-tick scoring loop.

The engine streams a cohort through the system as if it were happening now. It
knows nothing about where the cohort came from -- it holds a
:class:`~api.sources.CohortSource` and nothing else.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections import deque
from typing import Any, Awaitable, Callable

import pandas as pd

from api import config as C
from api.interventions import (
    AuditEntry,
    InterventionError,
    build_acknowledgement,
    build_override,
)
from api.scenarios import SCENARIOS, build_chartevents
from api.scoring import (
    Scorer,
    clinical_state,
    implied_esi,
    instability,
    model_applicable,
    retriage_candidate,
    retriage_direction,
)
from api.sources import CohortSource, TRUTH_COLUMNS, validate_cohort
from api.state import Census, PatientState, vitals_snapshot
from ml.features import (
    GRID_MIN,
    STAT_WINDOW_MIN,
    VITALS,
    chartevents_to_wide,
    compute_features,
)

log = logging.getLogger("watchlist.engine")

# Rows of history the feature builder needs to fill every lookback.
WINDOW_ROWS = STAT_WINDOW_MIN // GRID_MIN + 1


class ReplayClock:
    """Simulated wall clock, advanced by a speed multiplier."""

    def __init__(self, start: pd.Timestamp, speed: float = C.DEFAULT_SPEED) -> None:
        self.start = start
        self.sim_time = start
        self.speed = speed
        self.paused = False

    def advance(self, real_seconds: float) -> None:
        if not self.paused:
            self.sim_time += pd.Timedelta(minutes=self.speed * real_seconds / 60.0)

    def set_speed(self, speed: float) -> None:
        self.speed = max(C.MIN_SPEED, min(C.MAX_SPEED, float(speed)))

    def payload(self) -> dict[str, Any]:
        return {
            "sim_time": self.sim_time.isoformat(),
            "start": self.start.isoformat(),
            "elapsed_minutes": (self.sim_time - self.start).total_seconds() / 60.0,
            "speed": self.speed,
            "paused": self.paused,
            "tick_seconds": C.TICK_SECONDS,
        }


class ReplayEngine:
    def __init__(
        self,
        source: CohortSource,
        scorer: Scorer,
        admission: Scorer | None = None,
    ) -> None:
        self.source = source
        self.scorer = scorer
        self.admission = admission
        self.census = Census()
        self.clock: ReplayClock | None = None
        self.tick_count = 0
        self.complete = False

        self._grids: dict[int, pd.DataFrame] = {}
        self._static: dict[int, dict[str, Any]] = {}
        self._tmax: dict[int, float] = {}
        self._intime: dict[int, pd.Timestamp] = {}
        self._arrival_order: list[int] = []
        self._next_arrival = 0
        # Last feature vector scored per stay, kept so the explanation
        # endpoint explains exactly the number on the board.
        self._last_features: dict[int, dict[str, float]] = {}
        # Grid index the last feature vector was built from. This is the memo
        # key for explanations -- see _explain_active.
        self._last_idx: dict[int, int] = {}
        # stay_id -> (grid_idx, deterioration reasons, admission reasons).
        # One entry per stay, overwritten in place, so it is bounded by the
        # cohort rather than by uptime.
        self._reason_cache: dict[
            int, tuple[int, list[dict[str, Any]], list[dict[str, Any]]]
        ] = {}
        self.explain_computed = 0  # patients explained since start, for /api/health
        # Manual actions, newest last. Bounded; the record that matters
        # clinically lives on the patient, this is the activity feed.
        self.audit: deque[AuditEntry] = deque(maxlen=C.AUDIT_LOG_SIZE)
        self.injected_count = 0

    # -- setup -------------------------------------------------------------

    def prepare(self) -> None:
        """Load the cohort and pivot it once. Everything after this is slicing."""
        patients, chartevents = self.source.load()
        validate_cohort(patients, chartevents)

        grid = chartevents_to_wide(chartevents)
        # chartevents_to_wide reindexes each stay onto the complete grid, so
        # row i of a stay is always t == i * GRID_MIN. That makes the per-tick
        # window lookup an integer slice instead of a filter.
        for stay_id, g in grid.groupby("stay_id", sort=False):
            g = g.reset_index(drop=True)
            self._grids[int(stay_id)] = g
            self._tmax[int(stay_id)] = float(g["t"].iloc[-1])
            self._intime[int(stay_id)] = g["intime"].iloc[0]

        truth_cols = [c for c in TRUTH_COLUMNS if c in patients.columns]
        for _, row in patients.iterrows():
            sid = int(row["stay_id"])
            if sid not in self._grids:
                continue  # stay with no usable observations
            self._static[sid] = {
                "subject_id": int(row["subject_id"]),
                "age": int(row["age"]),
                "sex": str(row["sex"]),
                "esi": int(row["esi"]),
                "truth": {c: row[c] for c in truth_cols},
            }

        self._arrival_order = sorted(self._static, key=lambda s: self._intime[s])
        first = self._intime[self._arrival_order[0]]
        self.clock = ReplayClock(start=first)
        log.info(
            "engine ready: %d stays, %s -> %s",
            len(self._arrival_order),
            first,
            max(self._intime[s] + pd.Timedelta(minutes=self._tmax[s]) for s in self._arrival_order),
        )

    # -- per tick ----------------------------------------------------------

    def _admit_arrivals(self) -> None:
        assert self.clock is not None
        while self._next_arrival < len(self._arrival_order):
            sid = self._arrival_order[self._next_arrival]
            if self._intime[sid] > self.clock.sim_time:
                break
            s = self._static[sid]
            self.census.admit_to_board(
                PatientState(
                    subject_id=s["subject_id"],
                    stay_id=sid,
                    age=s["age"],
                    sex=s["sex"],
                    esi=s["esi"],
                    intime=self._intime[sid],
                )
            )
            self._next_arrival += 1

    def tick(self) -> dict[str, Any]:
        """Advance the clock, rescore every active patient, return the board."""
        assert self.clock is not None, "call prepare() first"
        self.clock.advance(C.TICK_SECONDS)
        self.tick_count += 1

        if not self.clock.paused:
            self._admit_arrivals()

            # Departures first, so we never score a stay that has ended.
            for stay_id in list(self.census.active):
                elapsed = (self.clock.sim_time - self._intime[stay_id]).total_seconds() / 60.0
                if elapsed > self._tmax[stay_id] + C.DEPART_GRACE_MIN:
                    self.census.discharge(stay_id)

            self._score_active()
            if C.EXPLAIN_IN_TICK:
                self._explain_active()

            self.complete = (
                self._next_arrival >= len(self._arrival_order) and not self.census.active
            )

        return self.board_payload()

    def _score_active(self) -> None:
        assert self.clock is not None
        stay_ids: list[int] = []
        rows: list[dict[str, float]] = []
        windows: dict[int, pd.DataFrame] = {}

        for stay_id, p in self.census.active.items():
            elapsed = (self.clock.sim_time - p.intime).total_seconds() / 60.0
            p.minutes_in_dept = max(elapsed, 0.0)

            # Only observations already charted at sim_time are visible. This
            # is the whole point of the replay -- no peeking at the future.
            idx = int(elapsed // GRID_MIN) + 1
            g = self._grids[stay_id]
            idx = max(1, min(idx, len(g)))
            window = g.iloc[max(0, idx - WINDOW_ROWS) : idx]
            windows[stay_id] = window

            feat = compute_features(window, p.age, p.sex, p.esi)
            self._last_features[stay_id] = feat
            self._last_idx[stay_id] = idx
            rows.append(feat)
            stay_ids.append(stay_id)

        if not stay_ids:
            return

        # Both models consume the identical feature vector, so build the
        # matrix once and score it twice. Falls back to separate matrices if
        # the two feature contracts ever diverge.
        X = self.scorer.build_matrix(rows)
        risks = self.scorer.score_matrix(X)
        if self.admission is None:
            admits = [None] * len(rows)
        elif self.admission.features == self.scorer.features:
            admits = self.admission.score_matrix(X)
        else:
            admits = self.admission.score(rows)

        for stay_id, risk, admit, feat in zip(stay_ids, risks, admits, rows):
            p = self.census.active[stay_id]
            p.risk = float(risk)
            p.news2 = float(feat.get("news2", float("nan")))
            p.vitals = vitals_snapshot(windows[stay_id])
            p.implied_esi = implied_esi(p.risk)
            p.direction = retriage_direction(p.esi, p.risk)
            p.alert = p.risk >= self.scorer.alert_threshold

            p.instability = instability(p.news2)
            p.model_applicable = model_applicable(p.news2)
            p.clinical_state = clinical_state(
                p.risk, p.news2, self.scorer.alert_threshold
            )

            if admit is not None and self.admission is not None:
                p.admit_risk = float(admit)
                p.admit_flag = p.admit_risk >= self.admission.alert_threshold

            # The re-triage comparison uses the EFFECTIVE acuity, so once a
            # nurse has overridden priority the flag is measured against their
            # call rather than against the triage number they overruled.
            # Otherwise the board would keep flagging a mismatch the human has
            # already resolved.
            p.update_retriage(
                retriage_candidate(p.effective_esi, p.risk), p.minutes_in_dept
            )

            # Acknowledgement policy runs HERE, inside the shared scoring
            # loop, which is the only reason a manually-charted patient and a
            # replayed one cannot diverge: both reach this line.
            transition = p.apply_ack_policy()
            if transition:
                self._record(
                    transition, p, by=(p.ack.by if p.ack else "system"),
                    detail={"risk": round(p.risk, 4)},
                )

            p.risk_history.append((round(p.minutes_in_dept, 1), p.risk))

            truth = self._static[stay_id]["truth"]
            if truth:
                onset = truth.get("onset_minutes")
                p.truth = {
                    "deteriorates": bool(truth.get("deteriorates", False)),
                    "onset_minutes": onset,
                    "minutes_to_onset": (
                        float(onset) - p.minutes_in_dept
                        if onset is not None and pd.notna(onset)
                        else None
                    ),
                    "will_admit": truth.get("will_admit"),
                }

    # -- explanations ------------------------------------------------------

    def _explain_active(self) -> None:
        """Attach the top reasons to every active patient, off the request path.

        TreeSHAP is ~60x the cost of a prediction, so this would be the most
        expensive thing in the tick if it ran naively. Two things make it
        cheap enough to do for the whole census every tick:

        **The memo key is the grid index, not the clock.** Vitals arrive on
        the GRID_MIN charting grid and ``compute_features`` reads its
        ``minutes_elapsed`` off the grid, not off ``sim_time`` -- so between
        grid advances a patient's feature vector is identical and so is its
        explanation. Caching on ``idx`` is therefore exact, not an
        approximation: at 60x replay it drops the work by ~5x, and each
        patient's grid advances on its own tick, so the cost spreads out
        instead of spiking.

        **What is left is one batched call.** Per-call overhead dominates
        TreeSHAP at this size (3.2 ms for one patient, 85 ms for 400), so
        explaining the stragglers one at a time would cost more than
        explaining everybody at once. Everything that needs recomputing this
        tick goes through a single ``pred_contribs`` pass.
        """
        pending: list[tuple[int, dict[str, float]]] = []
        for stay_id, p in self.census.active.items():
            feat = self._last_features.get(stay_id)
            if feat is None:
                continue
            idx = self._last_idx.get(stay_id)
            cached = self._reason_cache.get(stay_id)
            if cached is not None and cached[0] == idx:
                p.reasons, p.admit_reasons = cached[1], cached[2]
            else:
                pending.append((stay_id, feat))

        if not pending:
            return

        # Only bites on a cohort far larger than a real ED census. Degrade by
        # explaining the sickest first rather than by missing the tick.
        if len(pending) > C.EXPLAIN_MAX_PER_TICK:
            pending.sort(
                key=lambda kv: -(self.census.active[kv[0]].risk or 0.0)
            )
            pending = pending[: C.EXPLAIN_MAX_PER_TICK]

        rows = [feat for _, feat in pending]
        why_risk, why_admit = self._reasons_for_rows(rows)
        for i, (stay_id, _) in enumerate(pending):
            self._reason_cache[stay_id] = (
                self._last_idx.get(stay_id), why_risk[i], why_admit[i]
            )
            p = self.census.active[stay_id]
            p.reasons, p.admit_reasons = why_risk[i], why_admit[i]
        self.explain_computed += len(pending)

    def _reasons_for_rows(
        self, rows: list[dict[str, float]]
    ) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
        """Reasons from both models for the same batch of feature vectors.

        The bed call gets its own explanation rather than borrowing the
        deterioration model's: they are different boosters, and a patient can
        be heading for a bed for reasons that have nothing to do with why they
        are deteriorating. Sharing one explanation between them would put
        words in the admission model's mouth.
        """
        why_risk = self.scorer.reasons_batch(rows)
        if self.admission is None:
            return why_risk, [[] for _ in rows]
        return why_risk, self.admission.reasons_batch(rows)

    def reasons_for(
        self, stay_id: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Cached ``(deterioration, admission)`` reasons for one stay.

        Computed on demand if the tick loop has not got to it --
        ``EXPLAIN_IN_TICK`` off, or a patient past the per-tick ceiling.
        """
        cached = self._reason_cache.get(stay_id)
        idx = self._last_idx.get(stay_id)
        if cached is not None and cached[0] == idx:
            return cached[1], cached[2]
        feat = self._last_features.get(stay_id)
        if feat is None:
            return [], []
        why_risk, why_admit = self._reasons_for_rows([feat])
        self._reason_cache[stay_id] = (idx, why_risk[0], why_admit[0])
        self.explain_computed += 1
        return why_risk[0], why_admit[0]

    # -- manual control ----------------------------------------------------
    #
    # Everything below mutates state and then calls rescore(), which is the
    # SAME _score_active + _explain_active pair the tick loop runs. No manual
    # action computes a risk, sets a flag, or decides a clinical state on its
    # own -- it changes an input and asks the engine to catch up. That is what
    # keeps a hand-charted patient and a replayed one from ever diverging.

    def rescore(self) -> dict[str, Any]:
        """Re-run scoring and explanation now, outside the tick.

        Manual input has to take effect immediately -- including while the
        clock is paused, which is exactly when someone is demonstrating
        something. It reuses the tick's own functions rather than a shortcut.
        """
        self._score_active()
        if C.EXPLAIN_IN_TICK:
            self._explain_active()
        return self.board_payload()

    def _now(self) -> tuple[str, str]:
        """(simulated ISO, real ISO) -- every manual act is stamped with both.

        The simulated time is when it happened on the board; the wall time is
        when the human did it. During a 60x replay those are wildly different
        and each answers a different question afterwards.
        """
        sim = self.clock.sim_time.isoformat() if self.clock else ""
        return sim, dt.datetime.now().astimezone().isoformat(timespec="seconds")

    def _record(self, action: str, p: PatientState, by: str, detail: dict) -> AuditEntry:
        sim, wall = self._now()
        entry = AuditEntry(
            action=action, subject_id=int(p.subject_id), stay_id=int(p.stay_id),
            by=by, at_sim_time=sim, at_wall_time=wall, detail=detail,
        )
        self.audit.append(entry)
        log.info("manual: %s #%s by %s %s", action, p.subject_id, by, detail)
        return entry

    def _require(self, subject_id: int) -> PatientState:
        p = self.census.get(subject_id)
        if p is None:
            raise InterventionError(f"subject_id {subject_id} is not in the census")
        if p.stay_id not in self.census.active:
            raise InterventionError(f"subject_id {subject_id} has left the department")
        return p

    # -- 1. manual vitals --------------------------------------------------

    def append_observation(
        self, subject_id: int, values: dict[str, float], by: str | None = None
    ) -> dict[str, Any]:
        """Chart a new set of vitals for a patient, now.

        The observation lands in the grid cell covering the current simulated
        time, which is the newest cell the scorer reads -- so it is visible on
        the next scoring pass rather than whenever the clock catches up.

        **Hand-charting takes the patient off their replay trajectory.** Rows
        after the current cell are dropped, because a pre-generated future is
        not this patient's future once a human is driving. Their stay is held
        open for ``MANUAL_CONTROL_HOLD_MIN`` so they do not immediately depart
        for want of new observations, and each entry extends that again.

        Omitted vitals carry forward from the previous row, matching what
        ``chartevents_to_wide``'s forward-fill does for a partial real
        observation.
        """
        p = self._require(subject_id)
        assert self.clock is not None

        unknown = set(values) - set(VITALS)
        if unknown:
            raise InterventionError(
                f"unknown vitals {sorted(unknown)}; expected any of {list(VITALS)}"
            )
        clean = {k: float(v) for k, v in values.items() if v is not None}
        if not clean:
            raise InterventionError("no vitals supplied")

        g = self._grids[p.stay_id]
        elapsed = max(0.0, (self.clock.sim_time - p.intime).total_seconds() / 60.0)
        slot = int(elapsed // GRID_MIN)
        t_new = slot * GRID_MIN

        keep = g.loc[g["t"] < t_new]
        previous = keep.iloc[-1] if len(keep) else g.iloc[0]

        row = {c: previous[c] for c in g.columns}
        row["stay_id"] = p.stay_id
        row["t"] = t_new
        row["intime"] = p.intime
        row["charttime"] = p.intime + pd.Timedelta(minutes=t_new)
        row.update(clean)

        self._grids[p.stay_id] = pd.concat(
            [keep, pd.DataFrame([row])], ignore_index=True
        )
        self._tmax[p.stay_id] = float(t_new) + C.MANUAL_CONTROL_HOLD_MIN
        # The feature vector changed without the grid index advancing, so the
        # explanation memo for this stay is stale by construction.
        self._reason_cache.pop(p.stay_id, None)

        p.manual_entries += 1
        self._record(
            "vitals", p, by=by or C.DEFAULT_ACTOR,
            detail={"t": t_new, **{k: round(v, 1) for k, v in clean.items()}},
        )
        return self.rescore()

    # -- 2. scenario injection ---------------------------------------------

    def inject_scenario(
        self, key: str, by: str | None = None
    ) -> tuple[dict[str, Any], int]:
        """Put a scripted demo patient on the live board. Returns (board, subject_id).

        The preset emits ordinary chartevents and they go through
        ``chartevents_to_wide`` -- the same pivot the cohort went through at
        startup. From the moment it is registered the patient is
        indistinguishable from a replayed one: same window slicing, same
        features, same model, same flags.

        Arrival is stamped at the current simulated time, so the trajectory
        unfolds at whatever speed the clock is running. It does not survive a
        seek, which rebuilds the census from the source cohort.
        """
        scenario = SCENARIOS.get(key)
        if scenario is None:
            raise InterventionError(
                f"unknown scenario {key!r}; available: {sorted(SCENARIOS)}"
            )
        assert self.clock is not None, "call prepare() first"
        if self.injected_count >= C.MAX_INJECTED_PATIENTS:
            raise InterventionError(
                f"injection limit reached ({C.MAX_INJECTED_PATIENTS}); reset the replay"
            )

        static, chartevents = build_chartevents(scenario, intime=self.clock.sim_time)
        grid = chartevents_to_wide(chartevents).reset_index(drop=True)

        stay_id = int(static["stay_id"])
        self._grids[stay_id] = grid
        self._tmax[stay_id] = float(grid["t"].iloc[-1])
        self._intime[stay_id] = grid["intime"].iloc[0]
        self._static[stay_id] = {
            "subject_id": static["subject_id"], "age": static["age"],
            "sex": static["sex"], "esi": static["esi"], "truth": {},
        }

        p = PatientState(
            subject_id=static["subject_id"], stay_id=stay_id,
            age=static["age"], sex=static["sex"], esi=static["esi"],
            intime=self._intime[stay_id],
        )
        p.injected = scenario.key
        self.census.admit_to_board(p)
        self.injected_count += 1

        self._record(
            "inject", p, by=by or C.DEFAULT_ACTOR,
            detail={"scenario": scenario.key, "label": scenario.label,
                    "esi": scenario.esi},
        )
        return self.rescore(), int(static["subject_id"])

    # -- 3. priority override ----------------------------------------------

    def set_override(
        self,
        subject_id: int,
        direction: str,
        reason: str | None,
        by: str | None = None,
        target_esi: int | None = None,
    ) -> dict[str, Any]:
        """Record a nurse's priority decision against a patient.

        Applied to the EFFECTIVE acuity, so overriding twice steps twice. The
        model's score is untouched and stays on the row -- an override moves
        the patient's position on the board, it does not rewrite what the
        model said.
        """
        p = self._require(subject_id)
        sim, wall = self._now()
        override = build_override(
            direction=direction,
            current_esi=p.effective_esi,
            target_esi=target_esi,
            reason=reason,
            by=by,
            at_sim_minutes=p.minutes_in_dept,
            at_sim_time=sim,
            at_wall_time=wall,
            risk=None if p.risk != p.risk else p.risk,   # NaN-safe
        )
        p.override = override
        self._record("override", p, by=override.by, detail=override.payload())
        return self.rescore()

    def clear_override(self, subject_id: int, by: str | None = None) -> dict[str, Any]:
        """Withdraw an override. The record of it stays in the audit log."""
        p = self._require(subject_id)
        if p.override is None:
            raise InterventionError("patient has no override to clear")
        previous = p.override.payload()
        p.override = None
        self._record("override_cleared", p, by=by or C.DEFAULT_ACTOR,
                     detail={"was": previous})
        return self.rescore()

    # -- 4. acknowledgement -------------------------------------------------

    def acknowledge(
        self, subject_id: int, by: str | None = None, note: str | None = None
    ) -> dict[str, Any]:
        """Accept a raised re-triage flag: stop it escalating, keep it raised.

        The score keeps updating and the flag stays in the record. The risk at
        this moment is frozen into the acknowledgement so
        :meth:`PatientState.apply_ack_policy` can re-raise it if the patient
        gets materially worse -- and that check runs in the scoring loop, not
        here, so it applies to replayed and hand-charted patients alike.
        """
        p = self._require(subject_id)
        if not p.retriage_flag:
            raise InterventionError("patient has no raised re-triage flag to acknowledge")
        if p.ack is not None:
            raise InterventionError("flag is already acknowledged")

        sim, wall = self._now()
        ack = build_acknowledgement(
            by=by, note=note, at_sim_minutes=p.minutes_in_dept,
            at_sim_time=sim, at_wall_time=wall,
            risk=None if p.risk != p.risk else p.risk,
        )
        p.ack = ack
        self._record("acknowledge", p, by=ack.by, detail=ack.payload())
        return self.rescore()

    def audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Newest first, for the activity panel."""
        return [e.payload() for e in list(self.audit)[-limit:][::-1]]

    # -- demo controls -----------------------------------------------------

    def seek(self, elapsed_minutes: float) -> dict[str, Any]:
        """Jump the replay to an arbitrary point and rebuild the census there.

        Everything is derived from the precomputed grids, so the state at any
        time is reconstructible: clear the board, re-admit everyone who has
        arrived by then, discharge everyone already gone, and score once.

        Score history does NOT survive a seek -- it is accumulated tick by
        tick, and replaying every intermediate tick to rebuild it would take
        longer than the scrub is worth. Trend arrows therefore start empty
        after a jump and fill in as the clock runs.
        """
        assert self.clock is not None, "call prepare() first"
        elapsed = max(0.0, float(elapsed_minutes))

        self.census = Census()
        self._next_arrival = 0
        self._last_features.clear()
        self._last_idx.clear()
        self._reason_cache.clear()
        self.clock.sim_time = self.clock.start + pd.Timedelta(minutes=elapsed)

        self._admit_arrivals()
        for stay_id in list(self.census.active):
            gone = (self.clock.sim_time - self._intime[stay_id]).total_seconds() / 60.0
            if gone > self._tmax[stay_id] + C.DEPART_GRACE_MIN:
                self.census.discharge(stay_id)
        self._score_active()
        if C.EXPLAIN_IN_TICK:
            self._explain_active()
        self.complete = (
            self._next_arrival >= len(self._arrival_order) and not self.census.active
        )
        log.info("seek to +%.0f min -> census %d", elapsed, len(self.census.active))
        return self.board_payload()

    def reset(self) -> dict[str, Any]:
        """Back to the start of the cohort."""
        self.tick_count = 0
        return self.seek(0.0)

    @property
    def total_minutes(self) -> float:
        """Length of the whole replay, for the scrub bar's extent."""
        if not self._arrival_order:
            return 0.0
        start = self.clock.start if self.clock else self._intime[self._arrival_order[0]]
        end = max(
            self._intime[s] + pd.Timedelta(minutes=self._tmax[s])
            for s in self._arrival_order
        )
        return (end - start).total_seconds() / 60.0

    # -- output ------------------------------------------------------------

    def board_payload(self) -> dict[str, Any]:
        assert self.clock is not None
        return {
            "clock": self.clock.payload(),
            "tick": self.tick_count,
            "complete": self.complete,
            "alert_threshold": self.scorer.alert_threshold,
            "horizon_minutes": self.scorer.horizon_min,
            "admit_threshold": (
                self.admission.alert_threshold if self.admission is not None else None
            ),
            "news2_bands": {"medium": C.NEWS2_MEDIUM, "high": C.NEWS2_HIGH},
            "total_minutes": self.total_minutes,
            "summary": self.census.summary(),
            "patients": self.census.board(),
        }

    # -- loop --------------------------------------------------------------

    async def run(
        self, on_board: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    ) -> None:
        """Tick forever at wall-clock pace, pushing each board out."""
        while True:
            started = asyncio.get_running_loop().time()
            try:
                payload = self.tick()
                if on_board is not None:
                    await on_board(payload)
            except Exception:  # a bad tick must not kill the clock
                log.exception("tick %d failed", self.tick_count)
            drift = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0.0, C.TICK_SECONDS - drift))

    # -- history for the detail endpoint -----------------------------------

    def vitals_history(self, stay_id: int) -> list[dict[str, Any]]:
        """Charted vitals up to sim_time. Never returns the future."""
        assert self.clock is not None
        g = self._grids.get(stay_id)
        if g is None:
            return []
        elapsed = (self.clock.sim_time - self._intime[stay_id]).total_seconds() / 60.0
        idx = max(0, min(int(elapsed // GRID_MIN) + 1, len(g)))
        cols = ["t", "hr", "rr", "spo2", "sbp", "dbp", "temp"]
        out = g.iloc[:idx][cols].where(pd.notna(g.iloc[:idx][cols]), None)
        return out.to_dict(orient="records")
