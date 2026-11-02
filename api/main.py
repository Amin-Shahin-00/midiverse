"""FastAPI app: REST snapshot endpoints plus a WebSocket board feed.

Run from the ``watchlist/`` directory::

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api import config as C
from api.engine import ReplayEngine
from api.interventions import InterventionError
from api.reasons import LANGS, translate_reasons
from api.scenarios import catalogue
from api.scoring import admission_scorer, deterioration_scorer
from api.sources import CohortSource, SimulatedCohortSource
from ml.features import GRID_MIN, VITALS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("watchlist.api")


class Hub:
    """Fan-out for board updates. Drops slow clients rather than blocking."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


hub = Hub()


def build_engine(source: CohortSource | None = None) -> ReplayEngine:
    """Single place to swap the data source.

    Replace the default with ``MimicCohortSource(path)`` once PhysioNet access
    lands; nothing else in ``api/`` changes.
    """
    source = source or SimulatedCohortSource(
        n_patients=C.DEMO_PATIENTS, seed=C.DEMO_SEED
    )
    admission = admission_scorer()
    if admission is None:
        log.warning(
            "no admission model at %s -- board will omit bed demand. "
            "Run `python -m ml.train_admission` to enable it.",
            C.ADMISSION_MODEL_PATH,
        )
    return ReplayEngine(
        source=source, scorer=deterioration_scorer(), admission=admission
    )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    engine = build_engine()
    # Generating and pivoting the cohort is CPU-bound and takes seconds; keep
    # it off the event loop so startup does not block the server.
    await asyncio.to_thread(engine.prepare)
    app.state.engine = engine

    task = asyncio.create_task(engine.run(on_board=hub.broadcast))
    log.info("replay started at %.0fx", engine.clock.speed)
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="Watchlist ED triage support", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=C.CORS_ORIGINS,  # set WATCHLIST_CORS_ORIGINS to override
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _engine() -> ReplayEngine:
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(503, "engine still starting")
    return engine


class SeekRequest(BaseModel):
    elapsed_minutes: float | None = Field(
        default=None, ge=0, description="simulated minutes from the start of the replay"
    )
    reset: bool = False


class VitalsEntry(BaseModel):
    """A hand-charted observation. Every field optional; omitted vitals carry
    forward from the previous row, exactly as a partial real observation would
    after forward-fill."""

    hr: float | None = Field(default=None, ge=0, le=300, description="beats/min")
    rr: float | None = Field(default=None, ge=0, le=90, description="breaths/min")
    spo2: float | None = Field(default=None, ge=0, le=100, description="%")
    sbp: float | None = Field(default=None, ge=0, le=320, description="mmHg")
    dbp: float | None = Field(default=None, ge=0, le=250, description="mmHg")
    temp: float | None = Field(default=None, ge=20, le=46, description="degrees C")
    by: str | None = None

    def vitals(self) -> dict[str, float]:
        d = self.model_dump(exclude_none=True)
        d.pop("by", None)
        return {k: v for k, v in d.items() if k in VITALS}


class OverrideRequest(BaseModel):
    direction: str = Field(description="'up' = sicker than triaged, 'down' = less sick")
    reason: str = Field(
        min_length=C.OVERRIDE_REASON_MIN_CHARS,
        max_length=C.OVERRIDE_REASON_MAX_CHARS,
        description="why -- required, and kept in the record",
    )
    by: str | None = None
    target_esi: int | None = Field(default=None, ge=1, le=5)


class AckRequest(BaseModel):
    by: str | None = None
    note: str | None = Field(default=None, max_length=C.OVERRIDE_REASON_MAX_CHARS)


class InjectRequest(BaseModel):
    by: str | None = None


class ClockUpdate(BaseModel):
    speed: float | None = Field(
        default=None, ge=C.MIN_SPEED, le=C.MAX_SPEED,
        description="simulated minutes per wall-clock minute",
    )
    paused: bool | None = None


@app.get("/api/health")
def health() -> dict[str, Any]:
    engine = getattr(app.state, "engine", None)
    return {
        "ok": engine is not None,
        "ticks": getattr(engine, "tick_count", 0),
        "census": len(engine.census.active) if engine else 0,
        # Patients explained since start. Divided by ticks this is the memo's
        # hit rate made visible: it should sit near census/(GRID_MIN per tick),
        # not at census.
        "explained": getattr(engine, "explain_computed", 0),
    }


@app.get("/api/board")
def board() -> dict[str, Any]:
    """Current board snapshot -- the same payload the WebSocket pushes."""
    return _engine().board_payload()


@app.get("/api/clock")
def get_clock() -> dict[str, Any]:
    return _engine().clock.payload()


@app.post("/api/clock")
def set_clock(update: ClockUpdate) -> dict[str, Any]:
    """Change replay speed or pause. Useful mid-demo."""
    clock = _engine().clock
    if update.speed is not None:
        clock.set_speed(update.speed)
    if update.paused is not None:
        clock.paused = update.paused
    return clock.payload()


@app.post("/api/clock/seek")
async def seek(req: SeekRequest) -> dict[str, Any]:
    """Jump the replay to a point in time, or reset it to the start.

    Score history does not survive a jump -- see ``ReplayEngine.seek``.
    """
    engine = _engine()
    if req.reset:
        payload = await asyncio.to_thread(engine.reset)
    elif req.elapsed_minutes is not None:
        payload = await asyncio.to_thread(engine.seek, req.elapsed_minutes)
    else:
        raise HTTPException(422, "pass elapsed_minutes or reset=true")
    await hub.broadcast(payload)  # move every connected board at once
    return payload


@app.get("/api/languages")
def languages() -> dict[str, Any]:
    """Languages the reason text is available in, and their writing direction."""
    return {
        "languages": [
            {"code": code, "rtl": lang.rtl} for code, lang in LANGS.items()
        ]
    }


@app.get("/api/patients/{subject_id}")
def patient_detail(subject_id: int, lang: str = "en") -> dict[str, Any]:
    """Patient detail. ``lang`` re-words the reason text; nothing else changes.

    Reasons are computed and cached once in English -- the TreeSHAP pass
    behind them is language-independent -- and re-worded here per request, so
    switching language costs a string format, not a model call.
    """
    engine = _engine()
    p = engine.census.get(subject_id)
    if p is None:
        raise HTTPException(404, f"subject_id {subject_id} is not in the census")
    if not p.reasons and p.stay_id in engine._last_features:
        # The tick loop normally fills these. It has not here: either
        # EXPLAIN_IN_TICK is off, or this patient fell past
        # EXPLAIN_MAX_PER_TICK on a very large census. Either way the panel
        # gets its reasons -- reasons_for caches on the same grid index, so
        # this costs one TreeSHAP call and only the first time.
        p.reasons, p.admit_reasons = engine.reasons_for(p.stay_id)
    detail = p.to_detail()
    detail["reasons"] = translate_reasons(detail["reasons"], lang)
    detail["admit_reasons"] = translate_reasons(detail["admit_reasons"], lang)
    detail["lang"] = lang
    return detail


@app.get("/api/patients/{subject_id}/vitals")
def patient_vitals(subject_id: int) -> dict[str, Any]:
    engine = _engine()
    p = engine.census.get(subject_id)
    if p is None:
        raise HTTPException(404, f"subject_id {subject_id} is not in the census")
    return {
        "subject_id": subject_id,
        "stay_id": p.stay_id,
        "intime": p.intime.isoformat(),
        "grid_minutes": GRID_MIN,
        "observations": engine.vitals_history(p.stay_id),
    }


@app.get("/api/patients/{subject_id}/explain")
def patient_explain(subject_id: int, lang: str = "en") -> dict[str, Any]:
    """Why this patient has this score.

    Two layers, both from the same exact TreeSHAP pass: ``reasons`` is the top
    channels written as clinical sentences, ``contributions`` is the
    per-feature breakdown behind them. The panel leads with the first and
    keeps the second underneath, because "show me the actual features" is the
    second question anyone asks.

    The detail endpoint already carries ``reasons`` straight off the tick
    loop's cache -- a panel that only wants the sentences needs nothing from
    here.
    """
    engine = _engine()
    p = engine.census.get(subject_id)
    if p is None:
        raise HTTPException(404, f"subject_id {subject_id} is not in the census")
    feat = engine._last_features.get(p.stay_id)
    if feat is None:
        raise HTTPException(409, "patient has not been scored yet")

    out: dict[str, Any] = {
        "subject_id": subject_id,
        "risk": p.risk,
        "model_applicable": p.model_applicable,
        "deterioration": engine.scorer.explain(feat),
    }
    if engine.admission is not None:
        out["admission"] = {"risk": p.admit_risk, **engine.admission.explain(feat)}

    for block in ("deterioration", "admission"):
        if block in out and out[block].get("reasons"):
            out[block]["reasons"] = translate_reasons(out[block]["reasons"], lang)
    out["lang"] = lang
    return out


# ---------------------------------------------------------------------------
# Manual control
# ---------------------------------------------------------------------------
#
# Each of these mutates engine state and returns the board the engine produced
# by re-running its own scoring pass -- then broadcasts that same payload, so
# every connected screen moves together and a manual action is
# indistinguishable from a tick on the wire.
#
# The engine work is CPU-bound (a scoring pass plus TreeSHAP), so it runs in a
# worker thread rather than blocking the event loop and stalling the clock.


async def _apply(fn, *args, **kwargs) -> dict[str, Any]:
    """Run an engine mutation off the loop, broadcast the result, return it."""
    try:
        payload = await asyncio.to_thread(fn, *args, **kwargs)
    except InterventionError as e:
        raise HTTPException(409, str(e)) from e
    if isinstance(payload, tuple):        # inject_scenario returns (board, id)
        payload = payload[0]
    await hub.broadcast(payload)
    return payload


@app.post("/api/patients/{subject_id}/vitals")
async def enter_vitals(subject_id: int, entry: VitalsEntry) -> dict[str, Any]:
    """Chart a new set of vitals by hand and re-score immediately.

    Goes through ``ReplayEngine.append_observation`` -> ``rescore()`` ->
    ``_score_active()``, which is the identical path the replay clock takes.
    Nothing here computes a risk or sets a flag on its own.

    Note that hand-charting takes the patient off their pre-generated replay
    trajectory; see the engine method for why.
    """
    engine = _engine()
    return await _apply(
        engine.append_observation, subject_id, entry.vitals(), entry.by
    )


@app.get("/api/scenarios")
def list_scenarios() -> dict[str, Any]:
    """The presets the demo buttons render from."""
    return {"scenarios": catalogue()}


@app.post("/api/scenarios/{key}")
async def inject_scenario(key: str, req: InjectRequest | None = None) -> dict[str, Any]:
    """Inject a scripted patient onto the live board. One call, no form."""
    engine = _engine()
    by = req.by if req else None
    try:
        payload, subject_id = await asyncio.to_thread(engine.inject_scenario, key, by)
    except InterventionError as e:
        raise HTTPException(409, str(e)) from e
    await hub.broadcast(payload)
    return {"subject_id": subject_id, **payload}


@app.post("/api/patients/{subject_id}/override")
async def set_override(subject_id: int, req: OverrideRequest) -> dict[str, Any]:
    """Raise or lower a patient's priority, with a reason, on the record.

    The model keeps scoring underneath and the score stays on the row. The
    override changes where the patient sits on the board, never what the model
    said about them, and no later re-score clears it.
    """
    engine = _engine()
    return await _apply(
        engine.set_override, subject_id, req.direction, req.reason, req.by, req.target_esi
    )


@app.delete("/api/patients/{subject_id}/override")
async def clear_override(subject_id: int, by: str | None = None) -> dict[str, Any]:
    """Withdraw an override. The audit entry for it remains."""
    engine = _engine()
    return await _apply(engine.clear_override, subject_id, by)


@app.post("/api/patients/{subject_id}/acknowledge")
async def acknowledge(subject_id: int, req: AckRequest | None = None) -> dict[str, Any]:
    """Acknowledge a raised re-triage flag.

    Stops it escalating; does not clear it, and does not stop the score. If
    risk later climbs ``ACK_REESCALATE_RISK_DELTA`` above where it was when
    acknowledged, the scoring loop re-raises it.
    """
    engine = _engine()
    return await _apply(
        engine.acknowledge, subject_id, req.by if req else None, req.note if req else None
    )


@app.get("/api/audit")
def audit(limit: int = 100) -> dict[str, Any]:
    """Manual actions, newest first: who, what, when, why."""
    return {"entries": _engine().audit_log(limit=limit)}


@app.websocket("/ws/board")
async def ws_board(ws: WebSocket) -> None:
    await hub.connect(ws)
    try:
        engine = getattr(app.state, "engine", None)
        if engine is not None:
            await ws.send_json(engine.board_payload())  # don't wait for the next tick
        while True:
            await ws.receive_text()  # client keepalive; board is push-only
    except WebSocketDisconnect:
        hub.disconnect(ws)
    except Exception:
        hub.disconnect(ws)
