"""SHAP contributions -> sentences a charge nurse can act on.

The model's feature names are systematic (``rr_slope30``, ``shock_index_d15``)
and are the wrong unit of explanation for a board that gets read in five
seconds. Two things happen here.

**Aggregation.** The three largest contributions to an alert are routinely
``rr_d15``, ``rr_d30`` and ``rr_slope30`` -- one clinical fact stated three
times. SHAP contributions are additive, so summing them within a channel is a
sound attribution of that channel's total effect on the score. We rank
*channels*, not columns, and phrase each from its own largest member. Three
reasons therefore mean three different things about the patient.

**Phrasing.** Each (channel, kind, window) becomes a sentence carrying the
direction, the magnitude, the unit and the window: "Respiratory rate rising
4/min over the last 30 minutes". Never a column name -- those are kept
alongside in ``lead_feature`` so a number on screen can still be traced back
to the feature that produced it, which is the first thing anyone technical
asks.

**Languages.** The sentences are built from a structure -- channel, kind of
measurement, window, value, direction -- and only the last step turns that
structure into words. Adding a language is therefore a table of strings
(:data:`LANGS`), not a second copy of the logic, and every language is
guaranteed to describe exactly the same facts. Phrasing stays server-side for
the same reason it always did: the API, any export and the dashboard cannot
drift into describing one feature three different ways, in any language.

Nothing here imports xgboost or touches a model. It is pure formatting over a
feature dict and a contribution dict, which makes it directly testable --
see :func:`selftest`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from ml.features import CHANNELS, STATIC_FEATURES


class Channel:
    """How one physiological channel is written down.

    ``normal`` is the range drawn behind the vitals traces in the dashboard,
    repeated here because "28/min" and "28/min (high)" are different
    sentences to a nurse, and only the second one is worth reading.
    """

    __slots__ = (
        "key", "name", "mid", "unit", "space", "decimals", "normal", "noise",
    )

    def __init__(
        self,
        key: str,
        name: str,
        unit: str = "",
        space: bool = True,
        decimals: int = 0,
        normal: tuple[float, float] | None = None,
        noise: float = 0.0,
        mid: str | None = None,
    ) -> None:
        self.key = key
        self.name = name
        # How the channel reads mid-sentence ("...for a heart rate trend").
        # Acronyms keep their case: a blanket .lower() gives "spo2" and "map",
        # which is precisely the raw-looking output this module exists to
        # prevent. Defaults to the lowercase form for ordinary words.
        self.mid = mid if mid is not None else name.lower()
        self.unit = unit
        self.space = space          # "12 bpm" vs "12/min"
        self.decimals = decimals
        self.normal = normal
        # Movement at or below this is reported as "unchanged" rather than as
        # a trend. Charting noise dressed up as a trend is how a board loses
        # its audience.
        self.noise = noise

    def fmt(self, value: float, signed: bool = False) -> str:
        """A magnitude with its unit attached."""
        v = round(float(value), self.decimals)
        if self.decimals == 0:
            v = int(v)
        text = f"{v:+}" if signed else f"{v}"
        return f"{text} {self.unit}".rstrip() if self.space else f"{text}{self.unit}"


# Order is irrelevant; lookup is by key. Units and normal ranges match
# web/src/lib/risk.js -- if you change one, change both.
CHANNELS_SPEC: dict[str, Channel] = {
    "hr": Channel("hr", "Heart rate", "bpm", True, 0, (60, 100), noise=2),
    "rr": Channel("rr", "Respiratory rate", "/min", False, 0, (12, 20), noise=1),
    "spo2": Channel("spo2", "SpO2", "%", False, 0, (95, 100), noise=1, mid="SpO2"),
    "sbp": Channel("sbp", "Systolic BP", "mmHg", True, 0, (100, 140), noise=3),
    "dbp": Channel("dbp", "Diastolic BP", "mmHg", True, 0, (60, 90), noise=3),
    "temp": Channel("temp", "Temperature", "°C", True, 1, (36.1, 38.0), noise=0.15),
    "shock_index": Channel("shock_index", "Shock index", "", True, 2, (0.5, 0.9), noise=0.03),
    "pulse_pressure": Channel("pulse_pressure", "Pulse pressure", "mmHg", True, 0, (30, 50), noise=3),
    "map": Channel("map", "MAP", "mmHg", True, 0, (70, 100), noise=3, mid="MAP"),
    "news2": Channel("news2", "NEWS2", "points", True, 0, None, noise=0.5, mid="NEWS2"),
}

# Static features are their own single-column channels.
STATIC_SPEC: dict[str, str] = {
    "age": "Age",
    "sex_male": "Sex",
    "esi": "Assigned ESI",
    "minutes_elapsed": "Time in department",
}

# ---------------------------------------------------------------------------
# languages
# ---------------------------------------------------------------------------
#
# A language supplies WORDS ONLY. Everything numeric -- decimals, plausible
# ranges, the noise floor below which movement is not a trend -- lives on
# Channel above and is shared, so no translation can accidentally change what
# counts as a rising respiratory rate.
#
# Arabic note: Western digits are used deliberately. Arabic-Indic numerals
# (٤٢) are correct typographically but clinical systems across the region
# overwhelmingly display Western digits on monitors and charts, and a board
# read at three metres is the wrong place to make someone translate a number.


@dataclass(frozen=True)
class Lang:
    """Every string needed to turn a feature into a sentence."""

    code: str
    rtl: bool
    names: dict[str, tuple[str, str]]     # channel -> (display, mid-sentence)
    units: dict[str, str]                 # channel -> unit, may be ""
    band_high: str
    band_low: str
    # templates; {n}=name, {v}=value with unit, {w}=window phrase, {m}=minutes
    level: str
    rising: str
    falling: str
    unchanged: str
    slope_rising: str
    slope_falling: str
    slope_flat: str
    peaked: str
    dipped: str
    swinging: str
    steady: str
    missing_level: str
    missing_trend: str
    window_hour: str
    window_minutes: str
    window_hours: str
    age: str
    male: str
    female: str
    esi: str
    in_department: str
    not_recorded: str
    hours_short: str
    minutes_short: str


EN = Lang(
    code="en",
    rtl=False,
    names={
        "hr": ("Heart rate", "heart rate"),
        "rr": ("Respiratory rate", "respiratory rate"),
        "spo2": ("SpO2", "SpO2"),
        "sbp": ("Systolic BP", "systolic BP"),
        "dbp": ("Diastolic BP", "diastolic BP"),
        "temp": ("Temperature", "temperature"),
        "shock_index": ("Shock index", "shock index"),
        "pulse_pressure": ("Pulse pressure", "pulse pressure"),
        "map": ("MAP", "MAP"),
        "news2": ("NEWS2", "NEWS2"),
    },
    units={"hr": "bpm", "rr": "/min", "spo2": "%", "sbp": "mmHg", "dbp": "mmHg",
           "temp": "\u00b0C", "shock_index": "", "pulse_pressure": "mmHg",
           "map": "mmHg", "news2": "points"},
    band_high=" (high)",
    band_low=" (low)",
    level="{n} {v}{band}",
    rising="{n} rising {v} over {w}",
    falling="{n} falling {v} over {w}",
    unchanged="{n} unchanged over {w}",
    slope_rising="{n} rising {v} per hour on the {m}-minute trend",
    slope_falling="{n} falling {v} per hour on the {m}-minute trend",
    slope_flat="{n} flat on the {m}-minute trend",
    peaked="{n} peaked at {v} in {w}",
    dipped="{n} dipped to {v} in {w}",
    swinging="{n} swinging by \u00b1{v} through {w}",
    steady="{n} steady through {w}",
    missing_level="No {mid} charted yet",
    missing_trend="Not enough charting yet for a {mid} trend over {w}",
    window_hour="the last hour",
    window_minutes="the last {m} minutes",
    window_hours="the last {h} hours",
    age="Age {v}",
    male="Male",
    female="Female",
    esi="Triaged ESI {v}",
    in_department="{d} in the department",
    not_recorded="{n} not recorded",
    hours_short="h",
    minutes_short="m",
)

AR = Lang(
    code="ar",
    rtl=True,
    names={
        "hr": ("\u0645\u0639\u062f\u0644 \u0627\u0644\u0646\u0628\u0636", "\u0645\u0639\u062f\u0644 \u0627\u0644\u0646\u0628\u0636"),
        "rr": ("\u0645\u0639\u062f\u0644 \u0627\u0644\u062a\u0646\u0641\u0633", "\u0645\u0639\u062f\u0644 \u0627\u0644\u062a\u0646\u0641\u0633"),
        "spo2": ("\u062a\u0634\u0628\u0639 \u0627\u0644\u0623\u0643\u0633\u062c\u064a\u0646", "\u062a\u0634\u0628\u0639 \u0627\u0644\u0623\u0643\u0633\u062c\u064a\u0646"),
        "sbp": ("\u0627\u0644\u0636\u063a\u0637 \u0627\u0644\u0627\u0646\u0642\u0628\u0627\u0636\u064a", "\u0627\u0644\u0636\u063a\u0637 \u0627\u0644\u0627\u0646\u0642\u0628\u0627\u0636\u064a"),
        "dbp": ("\u0627\u0644\u0636\u063a\u0637 \u0627\u0644\u0627\u0646\u0628\u0633\u0627\u0637\u064a", "\u0627\u0644\u0636\u063a\u0637 \u0627\u0644\u0627\u0646\u0628\u0633\u0627\u0637\u064a"),
        "temp": ("\u062f\u0631\u062c\u0629 \u0627\u0644\u062d\u0631\u0627\u0631\u0629", "\u062f\u0631\u062c\u0629 \u0627\u0644\u062d\u0631\u0627\u0631\u0629"),
        "shock_index": ("\u0645\u0624\u0634\u0631 \u0627\u0644\u0635\u062f\u0645\u0629", "\u0645\u0624\u0634\u0631 \u0627\u0644\u0635\u062f\u0645\u0629"),
        "pulse_pressure": ("\u0636\u063a\u0637 \u0627\u0644\u0646\u0628\u0636", "\u0636\u063a\u0637 \u0627\u0644\u0646\u0628\u0636"),
        "map": ("\u0645\u062a\u0648\u0633\u0637 \u0627\u0644\u0636\u063a\u0637 \u0627\u0644\u0634\u0631\u064a\u0627\u0646\u064a", "\u0645\u062a\u0648\u0633\u0637 \u0627\u0644\u0636\u063a\u0637 \u0627\u0644\u0634\u0631\u064a\u0627\u0646\u064a"),
        "news2": ("NEWS2", "NEWS2"),
    },
    units={
        "hr": "\u0646\u0628\u0636\u0629/\u062f", "rr": "/\u062f", "spo2": "%",
        "sbp": "\u0645\u0644\u0645 \u0632\u0626\u0628\u0642", "dbp": "\u0645\u0644\u0645 \u0632\u0626\u0628\u0642",
        "temp": "\u00b0\u0645", "shock_index": "", "pulse_pressure": "\u0645\u0644\u0645 \u0632\u0626\u0628\u0642",
        "map": "\u0645\u0644\u0645 \u0632\u0626\u0628\u0642", "news2": "\u0646\u0642\u0637\u0629",
    },
    band_high=" (\u0645\u0631\u062a\u0641\u0639)",
    band_low=" (\u0645\u0646\u062e\u0641\u0636)",
    level="{n} {v}{band}",
    rising="{n} \u064a\u0631\u062a\u0641\u0639 {v} \u062e\u0644\u0627\u0644 {w}",
    falling="{n} \u064a\u0646\u062e\u0641\u0636 {v} \u062e\u0644\u0627\u0644 {w}",
    unchanged="{n} \u062f\u0648\u0646 \u062a\u063a\u064a\u064a\u0631 \u062e\u0644\u0627\u0644 {w}",
    slope_rising="{n} \u064a\u0631\u062a\u0641\u0639 {v} \u0641\u064a \u0627\u0644\u0633\u0627\u0639\u0629 \u0639\u0644\u0649 \u0627\u062a\u062c\u0627\u0647 {m} \u062f\u0642\u064a\u0642\u0629",
    slope_falling="{n} \u064a\u0646\u062e\u0641\u0636 {v} \u0641\u064a \u0627\u0644\u0633\u0627\u0639\u0629 \u0639\u0644\u0649 \u0627\u062a\u062c\u0627\u0647 {m} \u062f\u0642\u064a\u0642\u0629",
    slope_flat="{n} \u062b\u0627\u0628\u062a \u0639\u0644\u0649 \u0627\u062a\u062c\u0627\u0647 {m} \u062f\u0642\u064a\u0642\u0629",
    peaked="\u0628\u0644\u063a {n} \u0630\u0631\u0648\u062a\u0647 {v} \u062e\u0644\u0627\u0644 {w}",
    dipped="\u0627\u0646\u062e\u0641\u0636 {n} \u0625\u0644\u0649 {v} \u062e\u0644\u0627\u0644 {w}",
    swinging="{n} \u064a\u062a\u0630\u0628\u0630\u0628 \u0628\u0645\u0642\u062f\u0627\u0631 \u00b1{v} \u062e\u0644\u0627\u0644 {w}",
    steady="{n} \u0645\u0633\u062a\u0642\u0631 \u062e\u0644\u0627\u0644 {w}",
    missing_level="\u0644\u0645 \u064a\u064f\u0633\u062c\u0644 {mid} \u0628\u0639\u062f",
    missing_trend="\u0644\u0627 \u062a\u0648\u062c\u062f \u062a\u0633\u062c\u064a\u0644\u0627\u062a \u0643\u0627\u0641\u064a\u0629 \u0644\u0627\u062a\u062c\u0627\u0647 {mid} \u062e\u0644\u0627\u0644 {w}",
    window_hour="\u0622\u062e\u0631 \u0633\u0627\u0639\u0629",
    window_minutes="\u0622\u062e\u0631 {m} \u062f\u0642\u064a\u0642\u0629",
    window_hours="\u0622\u062e\u0631 {h} \u0633\u0627\u0639\u0627\u062a",
    age="\u0627\u0644\u0639\u0645\u0631 {v}",
    male="\u0630\u0643\u0631",
    female="\u0623\u0646\u062b\u0649",
    esi="\u0641\u0631\u0632 ESI {v}",
    in_department="{d} \u0641\u064a \u0627\u0644\u0642\u0633\u0645",
    not_recorded="{n} \u063a\u064a\u0631 \u0645\u0633\u062c\u0644",
    hours_short="\u0633",
    minutes_short="\u062f",
)

LANGS: dict[str, Lang] = {"en": EN, "ar": AR}
DEFAULT_LANG = "en"


def get_lang(lang: str | None) -> Lang:
    """Unknown or missing language falls back to English rather than failing.

    A bad Accept-Language header should not take a triage board down.
    """
    return LANGS.get((lang or DEFAULT_LANG).lower().split("-")[0], EN)


_KINDS = ("d", "slope", "min", "max", "std")
_SUFFIX = re.compile(rf"^(?P<channel>.+?)_(?P<kind>{'|'.join(_KINDS)})(?P<window>\d+)$")


def parse_feature(name: str) -> tuple[str, str, int | None]:
    """``"rr_slope30"`` -> ``("rr", "slope", 30)``; ``"rr"`` -> ``("rr", "level", None)``.

    Channel names contain underscores (``shock_index``), so the suffix is
    matched non-greedily against the known kinds rather than split on ``_``.
    """
    if name in STATIC_SPEC:
        return name, "static", None
    m = _SUFFIX.match(name)
    if m and m.group("channel") in CHANNELS_SPEC:
        return m.group("channel"), m.group("kind"), int(m.group("window"))
    return name, "level", None


def _window_phrase(minutes: int | None, lang: Lang = EN) -> str:
    if minutes is None:
        return ""
    if minutes == 60:
        return lang.window_hour
    if minutes % 60 == 0:
        return lang.window_hours.format(h=minutes // 60)
    return lang.window_minutes.format(m=minutes)


def _band(ch: Channel, value: float, lang: Lang = EN) -> str:
    """`` (high)`` / `` (low)`` / ``""`` against the channel's normal range."""
    if ch.normal is None:
        return ""
    lo, hi = ch.normal
    if value < lo:
        return lang.band_low
    if value > hi:
        return lang.band_high
    return ""


def _duration(minutes: float, lang: Lang = EN) -> str:
    m = max(0, int(round(minutes)))
    h, rem = divmod(m, 60)
    if h:
        return f"{h}{lang.hours_short} {rem:02d}{lang.minutes_short}"
    return f"{m}{lang.minutes_short}"


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _static_phrase(name: str, value: Any, lang: Lang = EN) -> str:
    if _is_missing(value):
        return lang.not_recorded.format(n=STATIC_SPEC[name])
    if name == "age":
        return lang.age.format(v=int(value))
    if name == "sex_male":
        return lang.male if float(value) >= 0.5 else lang.female
    if name == "esi":
        return lang.esi.format(v=int(value))
    if name == "minutes_elapsed":
        return lang.in_department.format(d=_duration(float(value), lang))
    return f"{STATIC_SPEC[name]} {value}"


def _fmt(ch: Channel, value: float, lang: Lang) -> str:
    """A magnitude with its unit, in the requested language.

    Decimals come from the Channel and are shared, so a value rounds
    identically in every language; only the unit word is translated.
    """
    v = round(float(value), ch.decimals)
    if ch.decimals == 0:
        v = int(v)
    unit = lang.units.get(ch.key, "")
    if not unit:
        return f"{v}"
    return f"{v} {unit}" if ch.space else f"{v}{unit}"


def phrase(name: str, value: Any, lang: str | Lang = EN) -> str:
    """One feature and its value as a clinical sentence.

    The sentence describes the patient. Whether the fact raises or lowers the
    score is carried separately, in ``effect`` -- conflating "respiratory rate
    is rising" with "this is why the number is big" is how explanation panels
    end up asserting things the model never said.

    ``lang`` selects the wording only. Which sentence applies -- trend versus
    unchanged, high versus normal -- is decided by the shared numeric
    thresholds on :class:`Channel`, so a reading can never be a rising trend
    in one language and steady in another.
    """
    L = lang if isinstance(lang, Lang) else get_lang(lang)
    channel, kind, window = parse_feature(name)

    if kind == "static":
        return _static_phrase(channel, value, L)

    ch = CHANNELS_SPEC.get(channel)
    if ch is None:                                   # unknown column, stay honest
        return f"{channel} = {value}" if not _is_missing(value) else f"{channel} not available"

    display, mid = L.names.get(ch.key, (ch.name, ch.mid))
    when = _window_phrase(window, L)

    if _is_missing(value):
        if kind == "level":
            return L.missing_level.format(mid=mid)
        return L.missing_trend.format(mid=mid, w=when)

    v = float(value)

    if kind == "level":
        return L.level.format(n=display, v=_fmt(ch, v, L), band=_band(ch, v, L))

    if kind == "d":
        if abs(v) <= ch.noise:
            return L.unchanged.format(n=display, w=when)
        tpl = L.rising if v > 0 else L.falling
        return tpl.format(n=display, v=_fmt(ch, abs(v), L), w=when)

    if kind == "slope":
        # Already a per-hour rate; the window says what it was measured over.
        if abs(v) <= ch.noise:
            return L.slope_flat.format(n=display, m=window)
        tpl = L.slope_rising if v > 0 else L.slope_falling
        return tpl.format(n=display, v=_fmt(ch, abs(v), L), m=window)

    if kind == "max":
        return L.peaked.format(n=display, v=_fmt(ch, v, L), w=when)

    if kind == "min":
        return L.dipped.format(n=display, v=_fmt(ch, v, L), w=when)

    if kind == "std":
        if abs(v) <= ch.noise:
            return L.steady.format(n=display, w=when)
        return L.swinging.format(n=display, v=_fmt(ch, v, L), w=when)

    return L.level.format(n=display, v=_fmt(ch, v, L), band="")


def short_label(name: str) -> str:
    """Terse label for the per-feature bar chart, where a sentence will not fit."""
    channel, kind, window = parse_feature(name)
    if kind == "static":
        return STATIC_SPEC[channel]
    ch = CHANNELS_SPEC.get(channel)
    base = ch.name if ch else channel
    return {
        "level": lambda: base,
        "d": lambda: f"{base} · change over {window} min",
        "slope": lambda: f"{base} · {window} min trend",
        "min": lambda: f"{base} · low, {window} min",
        "max": lambda: f"{base} · high, {window} min",
        "std": lambda: f"{base} · variability, {window} min",
    }.get(kind, lambda: base)()


# ---------------------------------------------------------------------------
# channel aggregation
# ---------------------------------------------------------------------------

def _channel_of(name: str) -> str:
    channel, _, _ = parse_feature(name)
    return channel


def group_contributions(
    contributions: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Sum per-column contributions into per-channel totals.

    Returns ``{channel: {"total": float, "members": [(name, contribution)...]}}``
    with members sorted strongest-first. The total is the channel's whole
    effect on the log-odds; the leading member is the specific fact to quote.
    """
    groups: dict[str, dict[str, Any]] = {}
    for name, contribution in contributions.items():
        key = _channel_of(name)
        g = groups.setdefault(key, {"total": 0.0, "members": []})
        g["total"] += float(contribution)
        g["members"].append((name, float(contribution)))

    for g in groups.values():
        # Strongest first, but only ever quote a member that pushes the same
        # way the channel as a whole does. A channel can net downwards while
        # containing one upward column; quoting that column would print a
        # fact under a "lowers risk" label, which is worse than saying
        # nothing. Fall back to the plain strongest if every member opposes
        # the total, which can only happen when the total is ~0 and the
        # reason is about to be dropped anyway.
        agree = 1.0 if g["total"] >= 0 else -1.0
        g["members"].sort(
            key=lambda kv: (-(kv[1] * agree > 0), -abs(kv[1]))
        )
    return groups


def top_reasons(
    features: dict[str, float],
    contributions: dict[str, float],
    top_k: int = 3,
    min_contribution: float = 1e-4,
    lang: str | Lang = EN,
) -> list[dict[str, Any]]:
    """The ``top_k`` channels driving this score, each as one sentence.

    Ranked by the absolute size of the channel's total contribution, because
    "what is driving this number" includes the things holding it down --
    a nurse reading "SpO2 98%, lowering risk" learns something real about why
    the alert is not higher. The direction is always stated in words, never
    left to the colour of a bar.
    """
    groups = group_contributions(contributions)
    ranked = sorted(groups.items(), key=lambda kv: -abs(kv[1]["total"]))

    out: list[dict[str, Any]] = []
    for channel, g in ranked:
        total = g["total"]
        if abs(total) < min_contribution:
            continue
        lead_name, lead_contribution = g["members"][0]
        lead_value = features.get(lead_name)
        out.append(
            {
                "channel": channel,
                "text": phrase(lead_name, lead_value, lang),
                "effect": "increases" if total > 0 else "decreases",
                "contribution": round(total, 4),
                "lead_feature": lead_name,
                "lead_value": (
                    None if _is_missing(lead_value) else round(float(lead_value), 3)
                ),
                "lead_contribution": round(lead_contribution, 4),
                "n_features": len(g["members"]),
            }
        )
        if len(out) >= top_k:
            break
    return out


def translate_reasons(
    reasons: list[dict[str, Any]], lang: str | Lang
) -> list[dict[str, Any]]:
    """Re-word an already-computed reason list into another language.

    The expensive part of a reason is the TreeSHAP pass behind it, and that
    is language-independent -- so reasons are computed and cached once, in
    English, and re-worded here at serve time. ``lead_feature`` and
    ``lead_value`` are all the structure a sentence needs, and both travel on
    the reason already.

    Everything except ``text`` is left exactly as it was: the channel, the
    contribution and the feature behind it are facts, not prose.
    """
    L = lang if isinstance(lang, Lang) else get_lang(lang)
    if L is EN:
        return reasons
    out = []
    for r in reasons:
        out.append({**r, "text": phrase(r["lead_feature"], r.get("lead_value"), L)})
    return out


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def selftest() -> int:
    """Assert the phrasing contract. Returns the number of checks made.

    There is no test suite in this repo yet, so this stands in: run
    ``python -m api.reasons`` after editing anything above.
    """
    checks = 0

    def eq(got: str, want: str) -> None:
        nonlocal checks
        assert got == want, f"\n got: {got!r}\nwant: {want!r}"
        checks += 1

    # parsing, including the underscored channel names
    assert parse_feature("rr_slope30") == ("rr", "slope", 30)
    assert parse_feature("shock_index_d15") == ("shock_index", "d", 15)
    assert parse_feature("pulse_pressure_min60") == ("pulse_pressure", "min", 60)
    assert parse_feature("news2") == ("news2", "level", None)
    assert parse_feature("minutes_elapsed") == ("minutes_elapsed", "static", None)
    checks += 5

    # the sentence from the brief
    eq(phrase("rr_d30", 4.1), "Respiratory rate rising 4/min over the last 30 minutes")
    eq(phrase("spo2_d15", -3.0), "SpO2 falling 3% over the last 15 minutes")
    eq(phrase("hr", 128), "Heart rate 128 bpm (high)")
    eq(phrase("hr", 78), "Heart rate 78 bpm")
    eq(phrase("sbp_min60", 88), "Systolic BP dipped to 88 mmHg in the last hour")
    eq(phrase("hr_max60", 131), "Heart rate peaked at 131 bpm in the last hour")
    eq(phrase("temp", 38.6), "Temperature 38.6 °C (high)")
    eq(phrase("shock_index_d60", 0.21), "Shock index rising 0.21 over the last hour")
    eq(phrase("hr_std60", 9.4), "Heart rate swinging by ±9 bpm through the last hour")
    eq(phrase("rr_slope30", 8.2), "Respiratory rate rising 8/min per hour on the 30-minute trend")
    eq(phrase("minutes_elapsed", 260), "4h 20m in the department")
    eq(phrase("esi", 4), "Triaged ESI 4")
    eq(phrase("sex_male", 1.0), "Male")
    eq(phrase("age", 78), "Age 78")

    # noise floor: charting jitter is not a trend
    eq(phrase("rr_d15", 0.4), "Respiratory rate unchanged over the last 15 minutes")
    eq(phrase("temp_d60", -0.1), "Temperature unchanged over the last hour")

    # missing data is stated, never silently dropped
    eq(phrase("rr", float("nan")), "No respiratory rate charted yet")
    eq(phrase("hr_d60", None), "Not enough charting yet for a heart rate trend over the last hour")

    # acronyms keep their case mid-sentence. A blanket .lower() produced
    # "spo2" and "map" here, which reads like a column name -- the exact
    # failure this module exists to prevent.
    eq(phrase("spo2", None), "No SpO2 charted yet")
    eq(phrase("map_d60", None), "Not enough charting yet for a MAP trend over the last hour")
    eq(phrase("news2_d30", None),
       "Not enough charting yet for a NEWS2 trend over the last 30 minutes")
    eq(phrase("shock_index", float("nan")), "No shock index charted yet")

    # every feature the model can ask about must phrase without raising, and
    # no sentence may leak a raw column name
    names = list(STATIC_FEATURES)
    for c in CHANNELS:
        names.append(c)
        for lb in (15, 30, 60):
            names += [f"{c}_d{lb}", f"{c}_slope{lb}"]
        names += [f"{c}_min60", f"{c}_max60", f"{c}_std60"]
    for name in names:
        for value in (1.0, -1.0, 0.0, 42.0, float("nan")):
            text = phrase(name, value)
            assert text and not text[0].islower(), f"{name!r} -> {text!r}"
            assert "_" not in text, f"{name!r} leaked a column name: {text!r}"
            for mangled in ("spo2", "news2", " map "):
                assert mangled not in text, (
                    f"{name!r} lowercased an acronym: {text!r}"
                )
            checks += 1

    # aggregation: three columns of one channel collapse to one reason
    features = {"rr_d15": 2.0, "rr_d30": 4.1, "rr_slope30": 8.2, "hr": 118.0, "spo2": 98.0}
    contributions = {"rr_d15": 0.3, "rr_d30": 0.5, "rr_slope30": 0.4, "hr": 0.6, "spo2": -0.25}
    reasons = top_reasons(features, contributions, top_k=3)
    assert [r["channel"] for r in reasons] == ["rr", "hr", "spo2"], reasons
    assert reasons[0]["contribution"] == 1.2          # 0.3 + 0.5 + 0.4
    assert reasons[0]["lead_feature"] == "rr_d30"     # strongest single column
    assert reasons[0]["text"] == "Respiratory rate rising 4/min over the last 30 minutes"
    assert reasons[0]["effect"] == "increases"
    assert reasons[2]["effect"] == "decreases"        # SpO2 is holding the score down
    assert reasons[0]["n_features"] == 3
    checks += 7

    # near-zero contributions are not worth a line on the board
    assert top_reasons({"hr": 80.0}, {"hr": 1e-9}, top_k=3) == []
    checks += 1

    # -- languages --------------------------------------------------------
    #
    # The risk with a translation is not that a word is wrong -- it is that
    # the two languages quietly stop describing the same patient. These check
    # parity, not vocabulary.

    every = list(STATIC_FEATURES)
    for c in CHANNELS:
        every.append(c)
        for lb in (15, 30, 60):
            every += [f"{c}_d{lb}", f"{c}_slope{lb}"]
        every += [f"{c}_min60", f"{c}_max60", f"{c}_std60"]

    digits = re.compile(r"[-+]?\d+(?:\.\d+)?")

    def _strip_names(text: str) -> str:
        """Drop channel names so only measured numbers remain.

        "SpO2" and "NEWS2" carry a digit in the name itself, which is not a
        measurement and differs between languages.
        """
        for L in LANGS.values():
            for display, mid in L.names.values():
                text = text.replace(display, " ").replace(mid, " ")
        return text.replace("ESI", " ")
    for name in every:
        for value in (4.1, -3.0, 0.0, 128.0, float("nan")):
            en = phrase(name, value, "en")
            ar = phrase(name, value, "ar")

            # every feature must phrase in every language
            assert en and ar, f"{name} produced nothing"
            # a translation must not silently fall back to English
            if not _is_missing(value) or True:
                assert ar != en or name in ("news2",), (
                    f"{name!r} was not translated: {ar!r}"
                )
            # no column names leak, in any language
            assert "_" not in ar, f"{name!r} leaked a column name in ar: {ar!r}"
            # THE NUMBERS MUST MATCH. If a language rounds or converts
            # differently the two panels are describing different patients,
            # which is far worse than an awkward phrase.
            assert digits.findall(_strip_names(en)) == digits.findall(_strip_names(ar)), (
                f"{name!r} numbers differ between languages:\n"
                f"  en {en!r}\n  ar {ar!r}"
            )
            checks += 4

    # The *decision* of which sentence applies is shared, so a reading that is
    # "unchanged" in English cannot be a trend in Arabic. Probe either side of
    # the noise floor for every channel.
    for c in CHANNELS:
        ch = CHANNELS_SPEC[c]
        for value, should_move in ((ch.noise * 0.5, False), (ch.noise * 4 + 1, True)):
            en = phrase(f"{c}_d30", value, "en")
            ar = phrase(f"{c}_d30", value, "ar")
            en_moved = "unchanged" not in en
            ar_moved = AR.unchanged.split("{n}")[-1].strip().split()[0] not in ar
            assert en_moved == should_move, f"{c}: en noise floor wrong at {value}"
            assert en_moved == ar_moved, (
                f"{c}: languages disagree on whether {value} is a trend\n"
                f"  en {en!r}\n  ar {ar!r}"
            )
            checks += 2

    # an unknown or missing language falls back to English rather than failing
    assert get_lang(None) is EN
    assert get_lang("de") is EN
    assert get_lang("AR") is AR and get_lang("ar-SA") is AR
    assert AR.rtl is True and EN.rtl is False
    checks += 5

    # the reason payload is translated end to end, structure untouched
    feats = {"rr_d30": 4.1, "hr": 118.0}
    contribs = {"rr_d30": 0.5, "hr": 0.6}
    r_en = top_reasons(feats, contribs, top_k=2, lang="en")
    r_ar = top_reasons(feats, contribs, top_k=2, lang="ar")
    assert [r["channel"] for r in r_en] == [r["channel"] for r in r_ar]
    assert [r["contribution"] for r in r_en] == [r["contribution"] for r in r_ar]
    assert [r["lead_feature"] for r in r_en] == [r["lead_feature"] for r in r_ar]
    assert all(a["text"] != e["text"] for a, e in zip(r_ar, r_en))
    checks += 4

    # re-wording a cached reason list must change only the prose
    cached = top_reasons({"rr_d30": 4.1, "hr": 118.0},
                         {"rr_d30": 0.5, "hr": 0.6}, top_k=2, lang="en")
    reworded = translate_reasons(cached, "ar")
    assert translate_reasons(cached, "en") is cached          # no-op for English
    for a, b in zip(cached, reworded):
        assert a["text"] != b["text"], "re-wording did not translate"
        assert {k: v for k, v in a.items() if k != "text"} == {
            k: v for k, v in b.items() if k != "text"
        }, "re-wording changed a fact, not just the prose"
        checks += 2
    checks += 1

    # every language must define a string for every channel the model can use,
    # or a patient silently gets an English word in an Arabic sentence
    for code, L in LANGS.items():
        for c in CHANNELS:
            assert c in L.names, f"{code} is missing a name for {c}"
            assert c in L.units, f"{code} is missing a unit for {c}"
            checks += 2

    # the quoted fact must push the way the reason says it does. Here the
    # channel nets downward but holds one large upward column: quoting that
    # column under "decreases" would be a false statement about the model.
    features = {"map": 95.0, "map_d60": 16.0}
    contributions = {"map": 0.40, "map_d60": -0.55}
    r = top_reasons(features, contributions, top_k=1)[0]
    assert r["effect"] == "decreases"
    assert r["lead_feature"] == "map_d60", r          # not the bigger "map"
    assert r["lead_contribution"] < 0                 # agrees with the total
    checks += 3

    # every reason, on real-shaped input, must quote a member that agrees
    # with its channel total
    for r in top_reasons(
        {"hr": 118.0, "hr_d30": -2.0, "spo2": 98.0},
        {"hr": 0.7, "hr_d30": -0.1, "spo2": -0.25},
        top_k=3,
    ):
        assert (r["lead_contribution"] > 0) == (r["effect"] == "increases"), r
        checks += 1

    return checks


if __name__ == "__main__":  # pragma: no cover
    print(f"api.reasons: {selftest()} checks passed")
