"""End-to-end checks for the four manual-control features.

These run against the real FastAPI app and the real engine, because the whole
claim being tested is that manual input and the replay clock reach the board
through the same code. A test that called the scoring functions directly would
prove nothing about that.

    python -m api.manual_selftest
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def _board(c) -> dict:
    return c.get("/api/board").json()


def _advance(minutes: float = 5.0) -> None:
    """Move the simulated clock the way a tick would, without running the loop.

    Manual vitals chart into the grid cell covering the current simulated
    time, so two entries with no time between them land in the same cell and
    the second replaces the first -- correct behaviour, and useless for
    building a trajectory. The clock stays paused so nothing else moves and
    the test stays deterministic.
    """
    import pandas as pd

    app.state.engine.clock.sim_time += pd.Timedelta(minutes=minutes)


def _chart(c, subject_id: int, **vitals):
    """Advance five simulated minutes, then hand-chart an observation."""
    _advance()
    r = c.post(f"/api/patients/{subject_id}/vitals", json=vitals)
    assert r.status_code == 200, r.text
    return r.json()


def _row(c, subject_id: int) -> dict:
    for r in _board(c)["patients"]:
        if r["subject_id"] == subject_id:
            return r
    raise AssertionError(f"#{subject_id} not on the board")


def selftest() -> int:  # noqa: C901 - a sequence of scenarios, not branching logic
    checks = 0
    with TestClient(app) as c:
        c.post("/api/clock", json={"paused": True})   # determinism
        c.post("/api/clock/seek", json={"elapsed_minutes": 2400})

        # ================================================================
        # 2. scenario presets  (done first: later checks use these patients)
        # ================================================================
        cat = c.get("/api/scenarios").json()["scenarios"]
        assert len(cat) == 3, cat
        assert {s["key"] for s in cat} == {
            "septic_elderly", "silent_deteriorator", "stable"
        }
        checks += 2

        injected = {}
        for key in ("septic_elderly", "silent_deteriorator", "stable"):
            r = c.post(f"/api/scenarios/{key}", json={"by": "demo"})
            assert r.status_code == 200, r.text
            body = r.json()
            injected[key] = body["subject_id"]
            # one call, no form: the response IS the new board
            assert "patients" in body and "summary" in body
            checks += 2

        for key, sid in injected.items():
            row = _row(c, sid)
            assert row["injected"] == key
            assert row["risk"] is not None, f"{key} was not scored on arrival"
            checks += 2

        # an injected patient is scored by the same model as everyone else:
        # it has reasons, a clinical state and a NEWS2 like any other row
        d = c.get(f"/api/patients/{injected['septic_elderly']}").json()
        assert d["reasons"], "injected patient got no SHAP reasons"
        assert d["clinical_state"] in (
            "unstable_now", "early_warning", "watch", "news2_elevated", "stable"
        )
        checks += 2

        # the stable control must not be alarming
        stable = _row(c, injected["stable"])
        assert not stable["retriage_flag"], "the stable control raised a flag"
        checks += 1

        assert c.post("/api/scenarios/nonsense").status_code == 409
        checks += 1

        # ================================================================
        # 1. manual vitals entry
        # ================================================================
        sid = injected["stable"]
        before = _row(c, sid)

        r = c.post(f"/api/patients/{sid}/vitals",
                   json={"hr": 148, "rr": 34, "spo2": 86, "sbp": 74,
                         "dbp": 42, "temp": 39.4, "by": "RN Okafor"})
        assert r.status_code == 200, r.text
        after_payload = r.json()
        # the response is a full board payload, same shape a tick broadcasts
        assert {"clock", "tick", "summary", "patients"} <= set(after_payload)
        checks += 2

        after = _row(c, sid)
        assert after["risk"] > before["risk"], (
            f"hand-charted collapse did not raise risk: "
            f"{before['risk']:.3f} -> {after['risk']:.3f}"
        )
        assert after["news2"] > before["news2"], "NEWS2 did not react to manual vitals"
        assert after["vitals"]["hr"] == 148, after["vitals"]
        assert after["manual_entries"] == 1
        checks += 4

        # it reached the board WHILE PAUSED -- manual input must not wait for
        # a tick it is not going to get
        assert c.get("/api/clock").json()["paused"] is True
        checks += 1

        # the observation is in the patient's charted history, not a side table
        hist = c.get(f"/api/patients/{sid}/vitals").json()["observations"]
        assert hist[-1]["hr"] == 148, "manual entry missing from vitals history"
        checks += 1

        # partial entries carry forward the rest
        c.post(f"/api/patients/{sid}/vitals", json={"spo2": 99})
        part = _row(c, sid)
        assert part["vitals"]["spo2"] == 99
        assert part["vitals"]["hr"] == 148, "omitted vital did not carry forward"
        checks += 2

        # and it is explained by the same SHAP path
        ex = c.get(f"/api/patients/{sid}/explain").json()
        assert ex["deterioration"]["reasons"], "no reasons after manual entry"
        checks += 1

        assert c.post("/api/patients/999999/vitals", json={"hr": 80}).status_code == 409
        assert c.post(f"/api/patients/{sid}/vitals", json={}).status_code == 409
        checks += 2

        # ================================================================
        # 3. manual override
        # ================================================================
        sid = injected["stable"]
        base = _row(c, sid)
        esi0 = base["esi"]

        # a reason is mandatory -- pydantic rejects it before the engine sees it
        assert c.post(f"/api/patients/{sid}/override",
                      json={"direction": "up"}).status_code == 422
        assert c.post(f"/api/patients/{sid}/override",
                      json={"direction": "up", "reason": "x"}).status_code == 422
        checks += 2

        r = c.post(f"/api/patients/{sid}/override",
                   json={"direction": "up", "reason": "looks grey, family worried",
                         "by": "RN Okafor"})
        assert r.status_code == 200, r.text
        row = _row(c, sid)
        ov = row["override"]
        assert ov is not None
        assert ov["direction"] == "up"
        assert ov["target_esi"] == esi0 - 1          # ESI counts DOWN with severity
        assert ov["prior_esi"] == esi0
        assert ov["by"] == "RN Okafor"                        # who
        assert ov["reason"] == "looks grey, family worried"   # why
        assert ov["at_sim_time"] and ov["at_wall_time"]       # when
        assert row["effective_esi"] == esi0 - 1
        assert row["esi"] == esi0, "override overwrote the assigned ESI"
        checks += 9

        # an escalated patient sorts above the automatic ranking, without
        # their score being rewritten
        board = _board(c)
        assert board["patients"][0]["subject_id"] == sid, (
            "overridden-up patient is not at the top of the board"
        )
        assert board["patients"][0]["risk"] == row["risk"], "score was rewritten"
        checks += 2

        # THE KEY PROPERTY: a re-score must not silently discard it
        risk_before = row["risk"]
        for i in range(6):
            _chart(c, sid, hr=120 + 6 * i, rr=24 + 2 * i, sbp=104 - 6 * i)
        survived = _row(c, sid)
        assert survived["override"] is not None, "re-scoring discarded the override"
        assert survived["override"]["reason"] == "looks grey, family worried"
        assert survived["override"]["by"] == "RN Okafor"
        assert survived["risk"] != risk_before, (
            f"score did not keep updating underneath the override "
            f"({risk_before:.4f} -> {survived['risk']:.4f})"
        )
        checks += 4

        # overriding again steps from the EFFECTIVE acuity, not the assigned one
        c.post(f"/api/patients/{sid}/override",
               json={"direction": "up", "reason": "deteriorating further"})
        assert _row(c, sid)["effective_esi"] == max(1, esi0 - 2)
        checks += 1

        # withdrawing leaves the audit trail behind
        assert c.delete(f"/api/patients/{sid}/override").status_code == 200
        assert _row(c, sid)["override"] is None
        assert c.delete(f"/api/patients/{sid}/override").status_code == 409
        checks += 3

        # ================================================================
        # 4. acknowledge on alerts
        # ================================================================
        # Drive a patient into a sustained re-triage flag with manual vitals.
        sid = injected["silent_deteriorator"]
        assert c.post(f"/api/patients/{sid}/acknowledge").status_code == 409, (
            "acknowledged a flag that was never raised"
        )
        checks += 1

        flagged = False
        for i in range(40):
            _chart(c, sid, hr=min(140, 96 + 3 * i), rr=min(32, 20 + i),
                   spo2=max(89, 97 - i), sbp=max(80, 122 - 3 * i),
                   dbp=max(48, 74 - 2 * i))
            if _row(c, sid)["retriage_flag"]:
                flagged = True
                break
        assert flagged, "could not raise a re-triage flag with manual vitals"
        checks += 1

        row = _row(c, sid)
        assert row["escalating"] is True, "a fresh flag should be escalating"
        checks += 1

        r = c.post(f"/api/patients/{sid}/acknowledge",
                   json={"by": "Dr Mensah", "note": "reviewed, awaiting bloods"})
        assert r.status_code == 200, r.text
        row = _row(c, sid)
        assert row["retriage_flag"] is True, "acknowledging cleared the flag"
        assert row["escalating"] is False, "acknowledging did not stop escalation"
        assert row["ack"]["by"] == "Dr Mensah"
        assert row["ack"]["note"] == "reviewed, awaiting bloods"
        risk_at_ack = row["ack"]["risk_at_ack"]
        reescalate_at = row["ack"]["reescalate_at_risk"]
        # The threshold is whichever of the two rules comes first, and it must
        # always be reachable -- this patient is acknowledged near the top of
        # the scale, which is where a fixed delta would silence them forever.
        assert risk_at_ack < reescalate_at < 1.0, (
            f"unreachable re-escalation threshold: ack at {risk_at_ack}, "
            f"needs {reescalate_at}"
        )
        checks += 6

        assert c.post(f"/api/patients/{sid}/acknowledge").status_code == 409, (
            "double-acknowledged the same flag"
        )
        checks += 1

        # the score keeps moving underneath an acknowledged flag
        risk_at_ack_row = _row(c, sid)["risk"]
        _chart(c, sid, hr=142, rr=33, spo2=88)
        moved = _row(c, sid)
        assert moved["risk"] != risk_at_ack_row, "score froze after acknowledgement"
        assert moved["ack"] is not None, "a small change should not re-raise"
        checks += 2

        # ...and a MATERIAL climb re-raises it
        reraised = False
        for _ in range(30):
            _chart(c, sid, hr=168, rr=42, spo2=78, sbp=60, dbp=34, temp=39.6)
            row = _row(c, sid)
            if row["ack"] is None and row["retriage_flag"]:
                reraised = True
                break
        assert reraised, (
            f"acknowledged flag never re-raised despite risk climbing to "
            f"{_row(c, sid)['risk']:.3f} from {risk_at_ack:.3f}"
        )
        assert _row(c, sid)["escalating"] is True, "re-raised flag is not escalating"
        checks += 2

        # ================================================================
        # audit trail: who / what / when / why, oldest preserved
        # ================================================================
        entries = c.get("/api/audit").json()["entries"]
        actions = {e["action"] for e in entries}
        assert {"inject", "vitals", "override", "override_cleared",
                "acknowledge"} <= actions, actions
        assert any(e["action"] == "ack_reescalated" for e in entries), (
            "re-escalation was not recorded"
        )
        for e in entries:
            assert e["by"] and e["at_sim_time"] and e["at_wall_time"]
            assert e["subject_id"] and e["stay_id"]
        ov_entry = next(e for e in entries if e["action"] == "override")
        assert ov_entry["detail"]["reason"], "override audit lost the reason"
        checks += 5

        # ================================================================
        # the board still ticks normally afterwards
        # ================================================================
        c.post("/api/clock", json={"paused": False})
        engine = app.state.engine
        before_tick = engine.tick_count
        engine.tick()
        assert engine.tick_count == before_tick + 1
        # overrides and acks survive a real tick, not just a manual re-score
        assert _row(c, injected["silent_deteriorator"])["retriage_flag"] is True
        checks += 2

    return checks


if __name__ == "__main__":  # pragma: no cover
    print(f"api.manual_selftest: {selftest()} checks passed")
