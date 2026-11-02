"""What a human did to the board, and the rules about it surviving.

Three kinds of manual act live here: a priority **override**, an
**acknowledgement** of a re-triage flag, and the **audit entry** that records
either. The engine owns when they happen; this module owns what they are and
when they stop applying.

The governing rule is that the model may not quietly undo a person. A score
recomputed a second later must not erase a decision a nurse made, and an
acknowledgement must not hide a patient who is now materially sicker than when
they were acknowledged. Those are opposite failure modes of the same design
question -- how long does a human decision hold -- and both are answered here
rather than scattered through the scoring loop.

Nothing in this module touches a model or a DataFrame. It is decisions and
timestamps, which makes the escalation rules directly testable; see
:func:`selftest`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from api import config as C


class InterventionError(ValueError):
    """Rejected manual action -- surfaced to the client as a 4xx."""


def _clean_reason(reason: str | None) -> str:
    if not C.OVERRIDE_REQUIRES_REASON:
        return (reason or "").strip()
    text = (reason or "").strip()
    if len(text) < C.OVERRIDE_REASON_MIN_CHARS:
        raise InterventionError(
            f"a reason of at least {C.OVERRIDE_REASON_MIN_CHARS} characters is "
            f"required to override priority"
        )
    return text[: C.OVERRIDE_REASON_MAX_CHARS]


def _clean_actor(actor: str | None) -> str:
    return (actor or "").strip()[:120] or C.DEFAULT_ACTOR


@dataclass
class Override:
    """A nurse's priority decision, standing until they withdraw it.

    ``target_esi`` is the acuity the human asserts. It never replaces the
    model's score -- the board keeps showing both, because "the model says
    0.71 and the nurse says ESI 2" is more informative than either alone, and
    hiding the disagreement is how a tool stops being trusted.
    """

    direction: str                 # "up" (sicker) or "down" (less sick)
    target_esi: int
    prior_esi: int
    reason: str
    by: str
    at_sim_minutes: float          # minutes into this patient's stay
    at_sim_time: str               # simulated wall clock, ISO
    at_wall_time: str              # real wall clock, ISO
    risk_at_override: float | None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Acknowledgement:
    """A re-triage flag someone has seen and taken responsibility for.

    Acknowledging silences the escalation, not the score. The patient keeps
    being re-scored, the flag stays in the record, and the acknowledgement
    carries the risk it was made at so :meth:`should_reescalate` can tell
    later whether the situation still resembles the one that was accepted.
    """

    by: str
    at_sim_minutes: float
    at_sim_time: str
    at_wall_time: str
    risk_at_ack: float
    note: str = ""
    reescalated: bool = False

    @property
    def reescalate_at(self) -> float:
        """The risk that will re-raise this flag.

        Whichever comes first: a fixed climb, or covering half the remaining
        distance to certainty. The second term is what stops a flag
        acknowledged at 0.95 from being unre-raisable -- a fixed +0.15 would
        ask for 1.10 and never arrive, silencing the alert permanently on the
        patients it matters most for. This is always strictly below 1.0.
        """
        headroom = max(0.0, 1.0 - self.risk_at_ack)
        return min(
            self.risk_at_ack + C.ACK_REESCALATE_RISK_DELTA,
            self.risk_at_ack + C.ACK_REESCALATE_HEADROOM_FRAC * headroom,
        )

    def should_reescalate(self, risk: float | None) -> bool:
        """Has the score climbed far enough to deserve a second look?

        Only upward movement counts. A patient improving after an
        acknowledgement is the expected case and must not re-alarm, or the
        button means nothing.
        """
        if risk is None or self.reescalated:
            return False
        return float(risk) >= self.reescalate_at

    def payload(self) -> dict[str, Any]:
        d = asdict(self)
        d["reescalate_at_risk"] = round(self.reescalate_at, 4)
        return d


@dataclass
class AuditEntry:
    """One manual act, in the order it happened.

    Kept even after the thing it refers to is gone -- a cleared override and a
    re-escalated acknowledgement both leave their entries behind, because the
    question asked afterwards is always "what did we do at the time", not
    "what is true now".
    """

    action: str                    # override | override_cleared | acknowledge | ...
    subject_id: int
    stay_id: int
    by: str
    at_sim_time: str
    at_wall_time: str
    detail: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# override construction
# ---------------------------------------------------------------------------

ESI_MIN, ESI_MAX = 1, 5


def next_target_esi(current: int, direction: str) -> int:
    """One step sicker or less sick, clamped to the ESI scale.

    ESI counts DOWN with severity: 1 is a resuscitation case, 5 is minor. So
    raising priority decrements. Getting this backwards is silent and awful,
    which is why it is one named function rather than arithmetic at three call
    sites.
    """
    if direction == "up":
        return max(ESI_MIN, int(current) - 1)
    if direction == "down":
        return min(ESI_MAX, int(current) + 1)
    raise InterventionError(f"direction must be 'up' or 'down', got {direction!r}")


def build_override(
    *,
    direction: str,
    current_esi: int,
    target_esi: int | None,
    reason: str | None,
    by: str | None,
    at_sim_minutes: float,
    at_sim_time: str,
    at_wall_time: str,
    risk: float | None,
) -> Override:
    """Validate a requested override and freeze it into a record."""
    if direction not in ("up", "down"):
        raise InterventionError(f"direction must be 'up' or 'down', got {direction!r}")

    target = int(target_esi) if target_esi is not None else next_target_esi(current_esi, direction)
    if not ESI_MIN <= target <= ESI_MAX:
        raise InterventionError(f"target_esi must be {ESI_MIN}-{ESI_MAX}, got {target}")
    if target == int(current_esi):
        raise InterventionError(
            f"patient is already at effective ESI {target}; nothing to override"
        )

    # Direction and target have to agree, or the record lies about intent.
    if direction == "up" and target > int(current_esi):
        raise InterventionError(
            f"direction 'up' means sicker, which is a LOWER ESI number; "
            f"target {target} is less acute than the current {current_esi}"
        )
    if direction == "down" and target < int(current_esi):
        raise InterventionError(
            f"direction 'down' means less sick, which is a HIGHER ESI number; "
            f"target {target} is more acute than the current {current_esi}"
        )

    return Override(
        direction=direction,
        target_esi=target,
        prior_esi=int(current_esi),
        reason=_clean_reason(reason),
        by=_clean_actor(by),
        at_sim_minutes=round(float(at_sim_minutes), 1),
        at_sim_time=at_sim_time,
        at_wall_time=at_wall_time,
        risk_at_override=None if risk is None else round(float(risk), 4),
    )


def build_acknowledgement(
    *,
    by: str | None,
    note: str | None,
    at_sim_minutes: float,
    at_sim_time: str,
    at_wall_time: str,
    risk: float | None,
) -> Acknowledgement:
    """Acknowledgement needs no reason -- 'I have seen it' is the whole claim.

    A reason is optional and kept if given. Requiring one here would push
    nurses to type "ok" a hundred times a shift, which degrades the override
    reason field's credibility by association.
    """
    return Acknowledgement(
        by=_clean_actor(by),
        at_sim_minutes=round(float(at_sim_minutes), 1),
        at_sim_time=at_sim_time,
        at_wall_time=at_wall_time,
        risk_at_ack=round(float(risk or 0.0), 4),
        note=(note or "").strip()[: C.OVERRIDE_REASON_MAX_CHARS],
    )


# ---------------------------------------------------------------------------
# board ordering
# ---------------------------------------------------------------------------

# Overridden patients move, rather than having their score quietly rewritten.
# A band is applied before risk in the sort, so an escalated patient sits at
# the top of the board with their real score still on display and a
# de-escalated one drops below the automatic ranking without disappearing.
BAND_OVERRIDDEN_UP = 0
BAND_AUTOMATIC = 1
BAND_OVERRIDDEN_DOWN = 2


def board_band(override: Override | None) -> int:
    if override is None:
        return BAND_AUTOMATIC
    return BAND_OVERRIDDEN_UP if override.direction == "up" else BAND_OVERRIDDEN_DOWN


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def selftest() -> int:
    """Assert the rules that stop a human decision being quietly undone."""
    checks = 0

    # ESI counts down with severity
    assert next_target_esi(4, "up") == 3
    assert next_target_esi(2, "down") == 3
    assert next_target_esi(1, "up") == 1          # clamped
    assert next_target_esi(5, "down") == 5        # clamped
    checks += 4

    kw = dict(
        at_sim_minutes=30.0, at_sim_time="2026-01-01T10:00:00",
        at_wall_time="2026-01-01T10:00:00", risk=0.5,
    )
    o = build_override(direction="up", current_esi=4, target_esi=None,
                       reason="looks unwell, diaphoretic", by="RN Patel", **kw)
    assert o.target_esi == 3 and o.prior_esi == 4
    assert o.by == "RN Patel" and o.reason.startswith("looks unwell")
    checks += 2

    # a reason is mandatory, and whitespace is not a reason
    for bad in (None, "", "  ", "x"):
        try:
            build_override(direction="up", current_esi=4, target_esi=None,
                           reason=bad, by=None, **kw)
            raise AssertionError(f"accepted empty reason {bad!r}")
        except InterventionError:
            checks += 1

    # direction and target must agree; ESI counts down, so 'up' to a higher
    # number is a contradiction and must be refused rather than recorded
    try:
        build_override(direction="up", current_esi=3, target_esi=4,
                       reason="valid reason", by=None, **kw)
        raise AssertionError("accepted 'up' toward a less acute ESI")
    except InterventionError:
        checks += 1
    try:
        build_override(direction="down", current_esi=3, target_esi=2,
                       reason="valid reason", by=None, **kw)
        raise AssertionError("accepted 'down' toward a more acute ESI")
    except InterventionError:
        checks += 1
    try:
        build_override(direction="up", current_esi=3, target_esi=3,
                       reason="valid reason", by=None, **kw)
        raise AssertionError("accepted a no-op override")
    except InterventionError:
        checks += 1

    # unattributed actions get the configured default, never blank
    o2 = build_override(direction="down", current_esi=2, target_esi=None,
                        reason="seen by doctor, discharged", by=None, **kw)
    assert o2.by == C.DEFAULT_ACTOR
    checks += 1

    # -- acknowledgement escalation -------------------------------------
    ack = build_acknowledgement(by="RN Patel", note=None, at_sim_minutes=10.0,
                                at_sim_time="t", at_wall_time="w", risk=0.50)
    assert not ack.should_reescalate(0.50)                       # unchanged
    assert not ack.should_reescalate(0.30)                       # improving
    assert not ack.should_reescalate(0.50 + C.ACK_REESCALATE_RISK_DELTA - 0.01)
    assert ack.should_reescalate(0.50 + C.ACK_REESCALATE_RISK_DELTA)
    assert ack.should_reescalate(0.95)
    assert not ack.should_reescalate(None)                       # unscored
    checks += 6

    # once it has re-escalated it does not keep firing
    ack.reescalated = True
    assert not ack.should_reescalate(0.99)
    checks += 1

    # the payload tells the UI the exact number that will re-raise it, so the
    # threshold is visible rather than folklore
    fresh = build_acknowledgement(by=None, note=None, at_sim_minutes=0.0,
                                  at_sim_time="t", at_wall_time="w", risk=0.50)
    assert fresh.payload()["reescalate_at_risk"] == round(
        0.50 + C.ACK_REESCALATE_RISK_DELTA, 4
    )
    checks += 1

    # SATURATION. Acknowledge a patient who is already critical and the
    # absolute rule alone would demand a risk above 1.0 -- unreachable, so the
    # flag could never re-raise however far they fell. The headroom rule has to
    # take over, and the threshold has to stay reachable.
    for risk_at_ack in (0.90, 0.946, 0.99, 0.999):
        high = build_acknowledgement(by=None, note=None, at_sim_minutes=0.0,
                                     at_sim_time="t", at_wall_time="w",
                                     risk=risk_at_ack)
        assert high.reescalate_at < 1.0, (
            f"ack at {risk_at_ack} can never re-raise: needs {high.reescalate_at}"
        )
        assert high.reescalate_at > risk_at_ack, "threshold must require a climb"
        assert high.should_reescalate(1.0), f"ack at {risk_at_ack} ignores a max score"
        assert not high.should_reescalate(risk_at_ack), "re-raised without a climb"
        checks += 4

    # the absolute rule still governs at low risk, where headroom is plentiful
    low = build_acknowledgement(by=None, note=None, at_sim_minutes=0.0,
                                at_sim_time="t", at_wall_time="w", risk=0.20)
    assert abs(low.reescalate_at - (0.20 + C.ACK_REESCALATE_RISK_DELTA)) < 1e-9
    checks += 1

    # -- board bands ------------------------------------------------------
    assert board_band(None) == BAND_AUTOMATIC
    assert board_band(o) == BAND_OVERRIDDEN_UP
    assert board_band(o2) == BAND_OVERRIDDEN_DOWN
    assert BAND_OVERRIDDEN_UP < BAND_AUTOMATIC < BAND_OVERRIDDEN_DOWN
    checks += 4

    return checks


if __name__ == "__main__":  # pragma: no cover
    print(f"api.interventions: {selftest()} checks passed")
