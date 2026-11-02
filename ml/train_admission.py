"""Train the admission model -- "who will need a bed".

Same feature matrix as the deterioration model, different target and a
different shape of problem:

* The label is a property of the whole stay, not of a moment, so every
  timestep of a stay carries the same label and no censoring applies.
* Admission is common (~45%), not rare, so ``scale_pos_weight`` is near 1 and
  AUROC is a reasonable headline. AUPRC is still reported.
* The useful question is not just "is it accurate" but "how early is it
  accurate", because a bed request at minute 20 is worth far more to flow than
  the same call at minute 300. That table is the point of this script.

Run with ``python -m ml.train_admission`` from the ``watchlist/`` directory.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from ml import features as F
from ml.train import (
    auroc_auprc,
    build_dataset,
    operating_point,
    print_operating_point,
    rule,
    threshold_at_sensitivity,
)
from ml.cohort import (
    add_source_arguments,
    model_paths,
    provenance_meta,
    source_from_args,
)

TARGET_SENSITIVITY = 0.80

# Defaults; the actual destination comes from ml.cohort.model_paths so a
# --source mimic run cannot overwrite the shipped simulator model.
MODEL_PATH = pathlib.Path("ml/admission.json")
META_PATH = pathlib.Path("ml/admission_meta.json")

# Buckets of time-in-department for the "how early" table, in minutes.
EARLINESS_BUCKETS = ((0, 60), (60, 120), (120, 240), (240, 480), (480, 10_000))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-patients", type=int, default=5000)
    ap.add_argument("-s", "--seed", type=int, default=7)
    add_source_arguments(ap)
    args = ap.parse_args()

    rule("DATA")
    model_path, meta_path = model_paths(args, "ml/admission")
    source, provenance = source_from_args(args)
    print(provenance)
    # No censoring: disposition applies to the whole stay.
    data = build_dataset(source, censor_post_onset=False)
    patients = source.patients()

    if "will_admit" not in data.columns:
        raise SystemExit("cohort has no will_admit column -- cannot train admission model")

    data["y_admit"] = data["will_admit"].astype(int)
    feats = F.feature_names()

    print(f"patients           {len(patients)}")
    print(f"scorable rows      {len(data):,}")
    print(f"admitted patients  {int(patients['will_admit'].sum())} "
          f"({patients['will_admit'].mean():.1%})")
    print(f"positive rows      {int(data['y_admit'].sum()):,} ({data['y_admit'].mean():.2%})")

    groups = data["subject_id"].to_numpy()
    tr_idx, te_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed).split(
            data, groups=groups
        )
    )
    train_all, test = data.iloc[tr_idx], data.iloc[te_idx]
    itr, iva = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed).split(
            train_all, groups=train_all["subject_id"].to_numpy()
        )
    )
    train, val = train_all.iloc[itr], train_all.iloc[iva]

    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"{name:>6}: {len(part):>7,} rows  {part['subject_id'].nunique():>5} patients  "
              f"{part['y_admit'].mean():.1%} admitted")
    print(f"patient overlap train/test: "
          f"{len(set(train_all['subject_id']) & set(test['subject_id']))}")

    X_tr, y_tr = train[feats], train["y_admit"].to_numpy()
    X_va, y_va = val[feats], val["y_admit"].to_numpy()
    X_te, y_te = test[feats], test["y_admit"].to_numpy()

    rule("MODEL")
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"scale_pos_weight   {spw:.2f}   (admission is common, so this is near 1)")

    model = XGBClassifier(
        n_estimators=1200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=40,
        reg_lambda=2.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=50,
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print(f"best iteration     {model.best_iteration}")

    p_te = model.predict_proba(X_te)[:, 1]
    p_va = model.predict_proba(X_va)[:, 1]

    # Baseline: triage acuity alone. ESI is what a charge nurse already has,
    # so beating it is the bar for this model earning a place on the board.
    esi_te = -test["esi"].to_numpy(dtype=float)  # negate: lower ESI = sicker

    rule("DISCRIMINATION (held-out patients)")
    m_auroc, m_auprc = auroc_auprc(y_te, p_te)
    e_auroc, e_auprc = auroc_auprc(y_te, esi_te)
    print(f"{'':22}{'AUROC':>8}{'AUPRC':>8}")
    print(f"{'XGBoost':22}{m_auroc:>8.3f}{m_auprc:>8.3f}")
    print(f"{'ESI alone':22}{e_auroc:>8.3f}{e_auprc:>8.3f}")
    print(f"{'chance':22}{0.5:>8.3f}{y_te.mean():>8.3f}")
    print(f"\nlift over ESI: AUROC {m_auroc - e_auroc:+.3f}, AUPRC {m_auprc - e_auprc:+.3f}")

    rule(f"OPERATING POINT (~{TARGET_SENSITIVITY:.0%} sensitivity, threshold from val)")
    thr = threshold_at_sensitivity(y_va, p_va, TARGET_SENSITIVITY)
    op = operating_point(y_te, p_te, thr)
    print_operating_point("XGBoost", op)

    # -- how early is the call right? --------------------------------------
    rule("EARLINESS (test rows bucketed by time in department)")
    print(f"{'minutes in ED':>16}{'rows':>9}{'admitted':>10}{'AUROC':>8}{'accuracy':>10}")
    elapsed = test["minutes_elapsed"].to_numpy(dtype=float)
    for lo, hi in EARLINESS_BUCKETS:
        m = (elapsed >= lo) & (elapsed < hi)
        if m.sum() < 50 or len(np.unique(y_te[m])) < 2:
            continue
        a = roc_auc_score(y_te[m], p_te[m])
        acc = ((p_te[m] >= thr) == (y_te[m] == 1)).mean()
        label = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
        print(f"{label:>16}{m.sum():>9,}{y_te[m].mean():>10.1%}{a:>8.3f}{acc:>10.1%}")

    # -- per-patient view: the board shows one number per patient, not per row
    rule("PER-PATIENT (first score after 60 minutes in department)")
    tp = test.copy()
    tp["p"] = p_te
    early = tp[tp["minutes_elapsed"] >= 60].sort_values("minutes_elapsed").groupby(
        "subject_id", as_index=False
    ).first()
    if len(early) > 50 and early["y_admit"].nunique() > 1:
        a, ap_ = auroc_auprc(early["y_admit"].to_numpy(), early["p"].to_numpy())
        print(f"  patients {len(early):,}   AUROC {a:.3f}   AUPRC {ap_:.3f}   "
              f"admitted {early['y_admit'].mean():.1%}")

    # -- leakage ablation ---------------------------------------------------
    #
    # In this simulator length of stay is drawn FROM will_admit, so
    # "how long have they been here" is a partial read of the label. Real EDs
    # have the same correlation but not deterministically, so the honest
    # headline is the fixed-timepoint per-patient number above, and the gap
    # below is how much of the row-level score is simulator artifact.
    rule("ABLATION: without minutes_elapsed")
    feats_no_time = [f for f in feats if f != "minutes_elapsed"]
    abl = XGBClassifier(
        n_estimators=1200, max_depth=4, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.5, min_child_weight=40, reg_lambda=2.0,
        scale_pos_weight=spw, objective="binary:logistic", eval_metric="auc",
        early_stopping_rounds=50, tree_method="hist",
        random_state=args.seed, n_jobs=-1,
    )
    abl.fit(train[feats_no_time], y_tr,
            eval_set=[(val[feats_no_time], y_va)], verbose=False)
    p_abl = abl.predict_proba(test[feats_no_time])[:, 1]
    a_auroc, a_auprc = auroc_auprc(y_te, p_abl)
    print(f"{'':22}{'AUROC':>8}{'AUPRC':>8}")
    print(f"{'with minutes_elapsed':22}{m_auroc:>8.3f}{m_auprc:>8.3f}")
    print(f"{'without':22}{a_auroc:>8.3f}{a_auprc:>8.3f}")
    print(f"{'attributable to it':22}{m_auroc - a_auroc:>+8.3f}{m_auprc - a_auprc:>+8.3f}")

    rule("TOP FEATURES (gain)")
    gain = model.get_booster().get_score(importance_type="gain")
    for k, v in sorted(gain.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k:28}{v:8.1f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(model_path))
    meta_path.write_text(
        json.dumps(
            {
                "features": feats,
                "target": "will_admit",
                "trained_on": "all scorable rows (no censoring)",
                "horizon_minutes": None,
                "grid_minutes": F.GRID_MIN,
                "threshold": thr,
                "target_sensitivity": TARGET_SENSITIVITY,
                "test_auroc": m_auroc,
                "test_auprc": m_auprc,
                "esi_baseline_auroc": e_auroc,
                "ablation_no_minutes_elapsed_auroc": a_auroc,
                "operating_point": op,
                "n_patients": int(len(source.patients())),
                "seed": int(args.seed),
                "cohort": provenance_meta(args, source.patients()),
            },
            indent=2,
        )
    )
    rule("SAVED")
    print(f"  {model_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
