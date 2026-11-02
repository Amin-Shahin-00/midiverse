"""Train the deterioration model.

Every timestep of every stay becomes one scored row. The label is prospective:
does clinical deterioration *begin* in the next 120 minutes.

Run with ``python -m ml.train`` from the ``watchlist/`` directory.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBClassifier

from ml import features as F
from ml.cohort import (
    add_source_arguments,
    model_paths,
    provenance_meta,
    source_from_args,
)

HORIZON_MIN = 120       # prediction horizon for the label
MIN_HISTORY_MIN = 15    # need at least one lookback before a row is scorable
TARGET_SENSITIVITY = 0.80

# Rows are heavily clustered: one deterioration event contributes ~24
# consecutive positive rows. A subgroup AUROC backed by a handful of events is
# noise however many rows it spans, so we report the event count and flag it.
MIN_EVENTS_FOR_SUBGROUP = 20

# Defaults; the actual destination comes from ml.cohort.model_paths so a
# --source mimic run cannot overwrite the shipped simulator model.
MODEL_PATH = pathlib.Path("ml/deterioration.json")
META_PATH = pathlib.Path("ml/deterioration_meta.json")


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

def build_dataset(source, censor_post_onset: bool = True) -> pd.DataFrame:
    """Score every timestep, attach the prospective label.

    With ``censor_post_onset`` (the default, used for the deterioration model)
    rows at or after onset are dropped. Once a patient has deteriorated there
    is nothing left to predict, and keeping those rows would teach the model to
    recognise established shock -- which the nurse can already see.

    The admission model passes ``censor_post_onset=False``: disposition is a
    property of the whole stay, and post-onset rows are perfectly valid
    evidence for "this person is getting a bed".
    """
    patients = source.patients()
    grid = F.chartevents_to_wide(source.chartevents())
    frame = F.build_feature_frame(grid, patients)

    cols = ["stay_id", "deteriorates", "onset_minutes", "esi", "age", "sex"]
    if "will_admit" in patients.columns:
        cols.append("will_admit")
    truth = patients[cols]
    frame = frame.merge(truth, on="stay_id", how="left", suffixes=("", "_p"))

    time_to_onset = frame["onset_minutes"] - frame["t"]
    frame["time_to_onset"] = time_to_onset
    frame["y"] = (
        frame["deteriorates"].fillna(False).astype(bool)
        & (time_to_onset > 0)
        & (time_to_onset <= HORIZON_MIN)
    ).astype(int)

    keep = frame["t"] >= MIN_HISTORY_MIN
    if censor_post_onset:
        post_onset = frame["deteriorates"].fillna(False).astype(bool) & (time_to_onset <= 0)
        keep = keep & (~post_onset)
    return frame.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def auroc_auprc(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, score), average_precision_score(y, score)


def threshold_at_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """Lowest score that still keeps ``target`` of the positives above it."""
    pos = score[y == 1]
    return float(np.quantile(pos, 1.0 - target))


def operating_point(y: np.ndarray, score: np.ndarray, thr: float) -> dict[str, float]:
    pred = score >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "threshold": thr,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "alerts_per_1000": 1000.0 * (tp + fp) / len(y),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def print_operating_point(name: str, op: dict[str, float]) -> None:
    print(f"  {name}")
    print(f"    threshold        {op['threshold']:.4f}")
    print(f"    sensitivity      {op['sensitivity']:.3f}")
    print(f"    specificity      {op['specificity']:.3f}")
    print(f"    PPV              {op['ppv']:.3f}")
    print(f"    alerts / 1000    {op['alerts_per_1000']:.1f}")
    print(f"    tp/fp/fn/tn      {op['tp']}/{op['fp']}/{op['fn']}/{op['tn']}")


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-patients", type=int, default=5000)
    ap.add_argument("-s", "--seed", type=int, default=7)
    add_source_arguments(ap)
    args = ap.parse_args()

    model_path, meta_path = model_paths(args, "ml/deterioration")

    rule("DATA")
    source, provenance = source_from_args(args)
    data = build_dataset(source)
    patients = source.patients()
    print(provenance)

    feats = F.feature_names()
    print(f"patients           {len(patients)}")
    print(f"deteriorating      {int(patients['deteriorates'].sum())} "
          f"({patients['deteriorates'].mean():.1%})")
    print(f"scorable rows      {len(data):,}")
    print(f"features           {len(feats)}")
    print(f"positive rows      {int(data['y'].sum()):,} ({data['y'].mean():.2%})")
    pos_esi45 = int(data.loc[data["esi"] >= 4, "y"].sum())
    print(f"  of which ESI 4-5 {pos_esi45:,} ({pos_esi45 / max(data['y'].sum(), 1):.1%})")

    # -- split by patient, never by row ------------------------------------
    groups = data["subject_id"].to_numpy()
    tr_idx, te_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed).split(data, groups=groups)
    )
    train_all, test = data.iloc[tr_idx], data.iloc[te_idx]

    inner_groups = train_all["subject_id"].to_numpy()
    itr, iva = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed).split(
            train_all, groups=inner_groups
        )
    )
    train, val = train_all.iloc[itr], train_all.iloc[iva]

    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"{name:>6}: {len(part):>7,} rows  {part['subject_id'].nunique():>5} patients  "
              f"{int(part['y'].sum()):>5} positives ({part['y'].mean():.2%})")
    overlap = set(train_all["subject_id"]) & set(test["subject_id"])
    print(f"patient overlap train/test: {len(overlap)}")

    X_tr, y_tr = train[feats], train["y"].to_numpy()
    X_va, y_va = val[feats], val["y"].to_numpy()
    X_te, y_te = test[feats], test["y"].to_numpy()

    # -- model --------------------------------------------------------------
    rule("MODEL")
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"scale_pos_weight   {spw:.1f}")

    params = dict(
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=40,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1,
    )
    model = XGBClassifier(
        n_estimators=1200,
        scale_pos_weight=spw,
        early_stopping_rounds=50,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    n_rounds = int(model.best_iteration) + 1
    print(f"best iteration     {model.best_iteration}")

    p_te = model.predict_proba(X_te)[:, 1]

    # Pick the operating threshold on out-of-fold predictions across every
    # training patient, not on the single validation split. Rows cluster by
    # patient, so a 240-patient split estimates the score distribution far too
    # coarsely and the chosen threshold does not transfer to test.
    oof = np.full(len(train_all), np.nan)
    gkf = GroupKFold(n_splits=5)
    for fold_tr, fold_va in gkf.split(train_all, groups=train_all["subject_id"].to_numpy()):
        ft, fv = train_all.iloc[fold_tr], train_all.iloc[fold_va]
        w = float((ft["y"] == 0).sum() / max((ft["y"] == 1).sum(), 1))
        fm = XGBClassifier(n_estimators=n_rounds, scale_pos_weight=w, **params)
        fm.fit(ft[feats], ft["y"], verbose=False)
        oof[fold_va] = fm.predict_proba(fv[feats])[:, 1]
    y_oof = train_all["y"].to_numpy()
    print(f"OOF AUROC          {roc_auc_score(y_oof, oof):.3f} "
          f"(threshold selection set, {train_all['subject_id'].nunique()} patients)")

    # NEWS2 as the score itself -- the baseline to beat.
    news_te = np.nan_to_num(test["news2"].to_numpy(dtype=float), nan=0.0)
    news_oof = np.nan_to_num(train_all["news2"].to_numpy(dtype=float), nan=0.0)

    # -- discrimination -----------------------------------------------------
    rule("DISCRIMINATION (held-out patients)")
    m_auroc, m_auprc = auroc_auprc(y_te, p_te)
    n_auroc, n_auprc = auroc_auprc(y_te, news_te)
    prev = y_te.mean()
    print(f"{'':22}{'AUROC':>8}{'AUPRC':>8}")
    print(f"{'XGBoost':22}{m_auroc:>8.3f}{m_auprc:>8.3f}")
    print(f"{'NEWS2 (raw score)':22}{n_auroc:>8.3f}{n_auprc:>8.3f}")
    print(f"{'chance':22}{0.5:>8.3f}{prev:>8.3f}   <- AUPRC chance = prevalence")
    print(f"\nlift over NEWS2: AUROC {m_auroc - n_auroc:+.3f}, "
          f"AUPRC {m_auprc - n_auprc:+.3f} ({m_auprc / max(n_auprc, 1e-9):.1f}x)")

    # -- operating point ----------------------------------------------------
    rule(f"OPERATING POINT (~{TARGET_SENSITIVITY:.0%} sensitivity, threshold from out-of-fold)")
    thr_m = threshold_at_sensitivity(y_oof, oof, TARGET_SENSITIVITY)
    thr_n = threshold_at_sensitivity(y_oof, news_oof, TARGET_SENSITIVITY)
    op_m = operating_point(y_te, p_te, thr_m)
    op_n = operating_point(y_te, news_te, thr_n)
    print_operating_point("XGBoost", op_m)
    print()
    print_operating_point("NEWS2", op_n)
    if op_n["specificity"] < 0.05:
        print("    NOTE: NEWS2 is integer-valued and most pre-onset rows score 0,")
        print("          so 80% sensitivity is only reachable by alerting on everyone.")

    # How much warning the alert actually buys, for true positives.
    fired = (p_te >= thr_m) & (y_te == 1)
    if fired.any():
        lead = test.loc[fired, "time_to_onset"]
        print(f"\n  median lead time on true alerts: {lead.median():.0f} min "
              f"(IQR {lead.quantile(.25):.0f}-{lead.quantile(.75):.0f})")

    # -- subgroups ----------------------------------------------------------
    rule("SUBGROUP AUROC (held-out patients)")
    subgroups = {
        "age < 65": test["age"] < 65,
        "age >= 65": test["age"] >= 65,
        "male": test["sex"] == "M",
        "female": test["sex"] == "F",
        "ESI 4-5": test["esi"] >= 4,
        "ESI 1-3": test["esi"] <= 3,
    }
    print(f"{'subgroup':14}{'rows':>9}{'pos rows':>10}{'events':>8}{'prev':>8}{'AUROC':>8}{'NEWS2':>8}")
    thin = []
    for name, mask in subgroups.items():
        m = mask.to_numpy()
        yy = y_te[m]
        # Distinct deterioration events, i.e. distinct patients contributing
        # positive rows -- the real sample size behind the estimate.
        events = test.loc[m & (test["y"] == 1).to_numpy(), "subject_id"].nunique()
        if len(np.unique(yy)) < 2:
            print(f"{name:14}{m.sum():>9,}{int(yy.sum()):>10}{events:>8}{'':>8}{'n/a':>8}{'n/a':>8}")
            continue
        a, _ = auroc_auprc(yy, p_te[m])
        b, _ = auroc_auprc(yy, news_te[m])
        flag = "" if events >= MIN_EVENTS_FOR_SUBGROUP else "  <- too few events"
        if flag:
            thin.append(name)
        print(f"{name:14}{m.sum():>9,}{int(yy.sum()):>10}{events:>8}{yy.mean():>8.2%}"
              f"{a:>8.3f}{b:>8.3f}{flag}")
    if thin:
        print(f"\n  {', '.join(thin)}: fewer than {MIN_EVENTS_FOR_SUBGROUP} "
              f"distinct events -- treat as indicative only.")

    # -- what the model is using -------------------------------------------
    rule("TOP FEATURES (gain)")
    gain = model.get_booster().get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:15]
    for k, v in top:
        print(f"  {k:28}{v:8.1f}")
    change_share = sum(
        v for k, v in gain.items() if any(tag in k for tag in ("_d", "_slope", "_std"))
    ) / max(sum(gain.values()), 1e-9)
    print(f"\n  share of total gain from change/dispersion features: {change_share:.1%}")

    # -- persist ------------------------------------------------------------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(model_path))
    meta_path.write_text(
        json.dumps(
            {
                "features": feats,
                "horizon_minutes": HORIZON_MIN,
                "target": "deterioration_onset_within_horizon",
                "trained_on": "pre-onset rows only (censored at onset)",
                "grid_minutes": F.GRID_MIN,
                "threshold": thr_m,
                "target_sensitivity": TARGET_SENSITIVITY,
                "test_auroc": m_auroc,
                "test_auprc": m_auprc,
                "news2_auroc": n_auroc,
                "news2_auprc": n_auprc,
                "operating_point": op_m,
                "n_patients": int(len(patients)),
                "seed": int(args.seed),
                "cohort": provenance_meta(args, patients),
            },
            indent=2,
        )
    )
    rule("SAVED")
    print(f"  {model_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
