"""Model loading, batch scoring, and the board's clinical interpretation."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import xgboost as xgb

from api import config as C
from api.reasons import short_label, top_reasons


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


class Scorer:
    """Wraps a saved booster and its feature contract.

    The feature order comes from the sidecar metadata written at training
    time, not from a list repeated here -- a reordered feature vector scores
    silently and wrongly, which is the worst kind of bug to ship.
    """

    def __init__(self, model_path: pathlib.Path, meta_path: pathlib.Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found -- run the matching training script first."
            )
        self.meta = json.loads(pathlib.Path(meta_path).read_text())
        self.features: list[str] = self.meta["features"]
        self.alert_threshold: float = float(self.meta["threshold"])

        horizon = self.meta.get("horizon_minutes")
        self.horizon_min: int | None = int(horizon) if horizon else None
        self.target: str = self.meta.get("target", "unknown")

        self.booster = xgb.Booster()
        self.booster.load_model(str(model_path))

    def build_matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        """Feature dicts -> the model's column order, as a float32 matrix."""
        X = pd.DataFrame(rows).reindex(columns=self.features)
        return X.to_numpy(dtype=np.float32)

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        """Score a matrix already in this model's column order.

        Lets two models that share a feature contract reuse one matrix
        instead of each rebuilding it -- assembling the DataFrame costs more
        than the prediction does at census scale.
        """
        if X.shape[0] == 0:
            return np.empty(0, dtype=float)
        return self.booster.inplace_predict(X)

    def score(self, rows: list[dict[str, float]]) -> np.ndarray:
        """Score a batch of feature dicts. Returns risk in [0, 1]."""
        if not rows:
            return np.empty(0, dtype=float)
        return self.score_matrix(self.build_matrix(rows))

    # -- explanation ------------------------------------------------------
    #
    # TreeSHAP is exact for tree ensembles (``pred_contribs``) rather than a
    # surrogate fit -- contributions sum to the log-odds of the score. That
    # exactness is the reason this stack is gradient boosting and not a
    # network, and it is what lets the panel claim the model is explaining
    # itself.
    #
    # It is also ~60x the cost of a prediction (85 ms for a 400-patient
    # census against 1.4 ms to score it), and the per-call overhead is brutal:
    # one patient alone costs 3.2 ms, almost all of it building the DMatrix.
    # So every path here is batched, and the engine memoises on top of it.

    def contribution_matrix(self, X: np.ndarray) -> np.ndarray:
        """TreeSHAP for a whole matrix at once. Shape ``(n, n_features + 1)``.

        The trailing column is the bias term. One call for the census, never
        one call per patient -- see the cost note above.
        """
        if X.shape[0] == 0:
            return np.empty((0, len(self.features) + 1), dtype=float)
        dmatrix = xgb.DMatrix(X, feature_names=self.features)
        return self.booster.predict(dmatrix, pred_contribs=True)

    def _explanation(
        self,
        row: dict[str, float],
        contribs: np.ndarray,
        top_k: int,
        reason_k: int,
    ) -> dict:
        """One patient's contribution vector -> the explanation payload."""
        signed = {
            name: float(v)
            for name, v in zip(self.features, contribs[:-1])
            if abs(v) > 1e-6
        }
        pairs = [
            {
                "feature": name,
                "label": humanize(name),
                "value": (
                    None
                    if _missing(row.get(name))
                    else round(float(row[name]), 2)
                ),
                "contribution": round(v, 4),
                "direction": "increases" if v > 0 else "decreases",
            }
            for name, v in signed.items()
        ]
        pairs.sort(key=lambda d: -abs(d["contribution"]))
        return {
            "baseline_logodds": round(float(contribs[-1]), 4),
            "reasons": top_reasons(row, signed, top_k=reason_k),
            "contributions": pairs[:top_k],
        }

    def explain_batch(
        self,
        rows: list[dict[str, float]],
        top_k: int = C.EXPLAIN_TOP_K,
        reason_k: int = C.REASON_TOP_K,
    ) -> list[dict]:
        """Explanations for many patients in a single TreeSHAP pass."""
        if not rows:
            return []
        contribs = self.contribution_matrix(self.build_matrix(rows))
        return [
            self._explanation(row, contribs[i], top_k, reason_k)
            for i, row in enumerate(rows)
        ]

    def explain(
        self,
        row: dict[str, float],
        top_k: int = C.EXPLAIN_TOP_K,
        reason_k: int = C.REASON_TOP_K,
    ) -> dict:
        """Explanation for one patient. Prefer :meth:`explain_batch` in loops."""
        return self.explain_batch([row], top_k, reason_k)[0]

    def reasons_batch(
        self, rows: list[dict[str, float]], reason_k: int = C.REASON_TOP_K
    ) -> list[list[dict]]:
        """Just the plain-language reasons, for the per-tick path.

        Skips assembling the per-feature bar-chart rows, which the board does
        not use and which cost more to build than the reasons themselves.
        """
        if not rows:
            return []
        contribs = self.contribution_matrix(self.build_matrix(rows))
        out = []
        for i, row in enumerate(rows):
            signed = {
                name: float(v)
                for name, v in zip(self.features, contribs[i][:-1])
                if abs(v) > 1e-6
            }
            out.append(top_reasons(row, signed, top_k=reason_k))
        return out


def humanize(name: str) -> str:
    """Terse label for the per-feature bar chart.

    Naming lives in :mod:`api.reasons` so the bar labels and the sentences
    above them cannot drift into calling the same channel two things.
    """
    return short_label(name)


def deterioration_scorer() -> Scorer:
    return Scorer(C.MODEL_PATH, C.META_PATH)


def admission_scorer() -> Scorer | None:
    """Optional -- the board degrades gracefully if it was never trained."""
    if not C.ADMISSION_MODEL_PATH.exists():
        return None
    return Scorer(C.ADMISSION_MODEL_PATH, C.ADMISSION_META_PATH)


# ---------------------------------------------------------------------------
# re-triage
# ---------------------------------------------------------------------------

def implied_esi(risk: float) -> int:
    """Map a risk score onto the acuity level it behaves like."""
    for edge, level in C.IMPLIED_ESI_BANDS:
        if risk >= edge:
            return level
    return C.IMPLIED_ESI_BANDS[-1][1]


def retriage_direction(assigned_esi: int, risk: float) -> str:
    """``"up"`` = sicker than triaged, ``"down"`` = less sick, ``"match"``."""
    gap = int(assigned_esi) - implied_esi(risk)  # ESI counts down with severity
    if gap >= C.RETRIAGE_ESI_GAP:
        return "up"
    if gap <= -C.RETRIAGE_ESI_GAP:
        return "down"
    return "match"


def retriage_candidate(assigned_esi: int, risk: float) -> bool:
    """Would this reading, on its own, justify a re-triage flag?

    The engine additionally requires the condition to hold for
    ``RETRIAGE_SUSTAIN_MIN`` simulated minutes before it raises the flag.
    """
    direction = retriage_direction(assigned_esi, risk)
    if direction == "up":
        return risk >= C.RETRIAGE_MIN_RISK
    if direction == "down":
        return C.FLAG_OVER_TRIAGE
    return False


# ---------------------------------------------------------------------------
# clinical state
# ---------------------------------------------------------------------------

def instability(news2: float) -> str:
    """Current derangement, straight from NEWS2. No model involved."""
    if news2 is None or (isinstance(news2, float) and np.isnan(news2)):
        return "unknown"
    if news2 >= C.NEWS2_HIGH:
        return "high"
    if news2 >= C.NEWS2_MEDIUM:
        return "medium"
    return "low"


def model_applicable(news2: float) -> bool:
    """Is the deterioration score in-distribution for this patient?

    The model is trained only on pre-onset rows -- every stay is censored at
    the moment deterioration begins. Scoring an already-deteriorated patient
    is therefore extrapolation: the number saturates near 1.0 and is not a
    calibrated probability of anything.

    We cannot observe onset live, so NEWS2 stands in for it. Once a patient is
    overtly unstable the board should be driving off that, not off a
    forward-looking score whose training population excluded them.
    """
    return instability(news2) != "high"


def clinical_state(risk: float, news2: float, alert_threshold: float) -> str:
    """One label combining what the model says with what the vitals say.

    ``unstable_now``   -- NEWS2 is already in escalation territory. Act on the
                          patient in front of you; the risk score here is
                          extrapolation (see :func:`model_applicable`).
    ``early_warning``  -- the model is alerting while NEWS2 is still clean.
                          This is the case the product exists for.
    ``watch``          -- model alerting, NEWS2 mildly deranged.
    ``news2_elevated`` -- NEWS2 up but the model is not alerting.
    ``stable``         -- neither.
    """
    band = instability(news2)
    if band == "high":
        return "unstable_now"

    alerting = risk >= alert_threshold
    if alerting and band == "low":
        return "early_warning"
    if alerting:
        return "watch"
    if band == "medium":
        return "news2_elevated"
    return "stable"
