"""One place both training scripts get their cohort from.

``--source sim`` is the simulator, unchanged and still the default. ``--source
mimic --mimic-root PATH`` is a real extract. The scripts themselves do not
branch on it: they receive something with ``patients()`` and ``chartevents()``
and carry on.

The provenance string this returns is printed at the top of every training run
and stored in the model's sidecar metadata, because a model file that does not
say which cohort trained it is a model file nobody should deploy.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

import pandas as pd


def add_source_arguments(ap: argparse.ArgumentParser) -> None:
    """Attach the cohort-selection flags to a training script's parser."""
    g = ap.add_argument_group("cohort source")
    g.add_argument(
        "--source", choices=("sim", "mimic"), default="sim",
        help="synthetic simulator (default) or a real MIMIC-IV extract",
    )
    g.add_argument(
        "--mimic-root", metavar="PATH",
        help="directory holding the MIMIC-IV modules (hosp/, icu/, ed/)",
    )
    g.add_argument(
        "--mimic-cohort", choices=("auto", "ed", "icu"), default="auto",
        help="'ed' needs MIMIC-IV-ED and is the real product; 'icu' is a "
             "degraded fallback with no triage acuity (see api/mimic.py)",
    )
    g.add_argument(
        "--mimic-limit", type=int, metavar="N",
        help="use only the first N stays by arrival -- for iterating quickly",
    )
    g.add_argument(
        "--out-stem", metavar="PATH",
        help="write the model to PATH.json and PATH_meta.json. Default: "
             "ml/<name>.json for --source sim, ml/<name>_mimic.json for mimic, "
             "so a real-data run never overwrites the shipped simulator models",
    )
    g.add_argument(
        "--outcome", default="icu_transfer,vasopressor", metavar="LIST",
        help="comma-separated deterioration components: icu_transfer, "
             "vasopressor, rapid_response, ventilation, death "
             "(default: icu_transfer,vasopressor)",
    )


def model_paths(args: argparse.Namespace, stem: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Where this run's model and sidecar go.

    A MIMIC run writes to ``<stem>_mimic.json`` and a simulator run to
    ``<stem>.json``, so the two cohorts coexist instead of overwriting each
    other. That is not tidiness -- the shipped models are simulator-trained and
    the API loads them by name, so a training run pointed at a new extract
    would otherwise silently replace the thing the demo is serving. Override
    with ``--out-stem`` when you want a specific name (e.g. per outcome
    definition).
    """
    base = pathlib.Path(args.out_stem) if getattr(args, "out_stem", None) else pathlib.Path(stem)
    if not getattr(args, "out_stem", None) and args.source == "mimic":
        base = base.with_name(base.name + "_mimic")
    return base.with_suffix(".json"), base.with_name(base.name + "_meta").with_suffix(".json")


def _outcome_spec(spec_arg: str):
    from ml.outcomes import RRT_ITEMIDS, OutcomeSpec

    names = {s.strip() for s in spec_arg.split(",") if s.strip()}
    valid = {"icu_transfer", "vasopressor", "rapid_response", "ventilation", "death"}
    unknown = names - valid
    if unknown:
        raise SystemExit(f"unknown --outcome component(s): {sorted(unknown)}; "
                         f"choose from {sorted(valid)}")
    if "rapid_response" in names and not RRT_ITEMIDS:
        print("  WARNING: rapid_response requested, but MIMIC-IV records no RRT")
        print("           activations and ml.outcomes.RRT_ITEMIDS is empty. That")
        print("           component will contribute nothing. Do not describe the")
        print("           label as including RRT.")
    return OutcomeSpec(**{name: (name in names) for name in valid})


def source_from_args(args: argparse.Namespace) -> tuple[Any, str]:
    """Build the cohort source. Returns ``(source, provenance_text)``."""
    if args.source == "sim":
        from sim.simulator import SimulatedSource

        src = SimulatedSource(n_patients=args.n_patients, seed=args.seed)
        return src, (
            f"cohort            synthetic simulator "
            f"(n={args.n_patients}, seed={args.seed})\n"
            f"                  numbers from this cohort describe the simulator, "
            f"not any real ED"
        )

    if not args.mimic_root:
        raise SystemExit("--source mimic requires --mimic-root PATH")

    from api.mimic import MimicCohortSource

    src = MimicCohortSource(
        root=args.mimic_root,
        cohort=args.mimic_cohort,
        limit_stays=args.mimic_limit,
        outcomes=_outcome_spec(args.outcome),
        verbose=True,
    )
    src.load()  # builds now, so the QC and label reports print before training
    return src, src.provenance()


def provenance_meta(args: argparse.Namespace, patients: pd.DataFrame) -> dict[str, Any]:
    """The cohort description that goes into the model's sidecar JSON."""
    meta: dict[str, Any] = {
        "source": args.source,
        "n_stays": int(len(patients)),
        "n_subjects": int(patients["subject_id"].nunique()),
        "deteriorating_rate": float(patients["deteriorates"].mean()),
    }
    if args.source == "sim":
        meta["simulator_seed"] = int(args.seed)
        meta["caveat"] = (
            "Synthetic cohort. Performance is a property of the simulator's "
            "prodrome assumption, not evidence about real patients."
        )
    else:
        meta["mimic_root"] = str(args.mimic_root)
        meta["mimic_cohort"] = args.mimic_cohort
        meta["outcome_definition"] = args.outcome
        meta["limit_stays"] = args.mimic_limit
        meta["caveat"] = (
            "Real MIMIC-IV cohort. The deterioration label is a proxy built "
            "from ICU transfer and vasopressor initiation; see ml/outcomes.py "
            "and docs/MIMIC.md for what it does and does not capture."
        )
    return meta
