"""The seam between the replay engine and whatever is producing patients.

The engine never imports the simulator. It takes a :class:`CohortSource`, which
hands back two frames in MIMIC-IV shape. Swapping the simulator for a real
PhysioNet extract means writing one class here and changing one line in
``main.py``.
"""

from __future__ import annotations

import pandas as pd
from typing import Protocol, runtime_checkable

# Columns the engine actually reads off the patients frame. Anything else is
# passed through untouched.
REQUIRED_PATIENT_COLUMNS = ("subject_id", "stay_id", "age", "sex", "esi")

# Present in simulated cohorts, absent in real ones. Used only to annotate the
# demo board with hindsight; never fed to the model, never used for ranking.
TRUTH_COLUMNS = ("deteriorates", "onset_minutes", "will_admit")

REQUIRED_EVENT_COLUMNS = ("stay_id", "charttime", "itemid", "valuenum")


@runtime_checkable
class CohortSource(Protocol):
    """Anything that can hand the engine a cohort.

    ``load()`` returns ``(patients, chartevents)``:

    * ``patients``    -- one row per ED stay, at least
                         :data:`REQUIRED_PATIENT_COLUMNS`.
    * ``chartevents`` -- long-format observations, at least
                         :data:`REQUIRED_EVENT_COLUMNS`, with ``itemid`` values
                         that ``ml.features.MIMIC_ITEMID_MAP`` understands.
    """

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def validate_cohort(patients: pd.DataFrame, chartevents: pd.DataFrame) -> None:
    """Fail loudly at startup rather than mysteriously on the first tick."""
    missing_p = [c for c in REQUIRED_PATIENT_COLUMNS if c not in patients.columns]
    if missing_p:
        raise ValueError(f"patients frame is missing columns: {missing_p}")

    missing_e = [c for c in REQUIRED_EVENT_COLUMNS if c not in chartevents.columns]
    if missing_e:
        raise ValueError(f"chartevents frame is missing columns: {missing_e}")

    if not pd.api.types.is_datetime64_any_dtype(chartevents["charttime"]):
        raise TypeError("chartevents.charttime must be datetime64")

    orphans = set(chartevents["stay_id"].unique()) - set(patients["stay_id"].unique())
    if orphans:
        raise ValueError(f"{len(orphans)} stay_id(s) in chartevents have no patient row")


class SimulatedCohortSource:
    """Default source: the synthetic generator in ``sim/``."""

    def __init__(self, n_patients: int, seed: int) -> None:
        self.n_patients = n_patients
        self.seed = seed

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        from sim.simulator import EDSimulator  # imported lazily: engine never needs it

        cohort = EDSimulator(n_patients=self.n_patients, seed=self.seed).generate()
        return cohort.patients, cohort.chartevents


def MimicCohortSource(*args, **kwargs):  # noqa: N802 - kept as a constructor name
    """Real MIMIC-IV extract. Implemented in :mod:`api.mimic`.

    Imported lazily and re-exported here so this module stays the one place
    you look for "where do patients come from", without dragging the MIMIC
    reader (and its table discovery) into every process that only wants the
    simulator.

        from api.sources import MimicCohortSource
        engine = build_engine(MimicCohortSource("/data/mimic-iv", cohort="ed"))

    See :class:`api.mimic.MimicCohortSource` for the arguments, the cohort
    modes, and the caveats.
    """
    from api.mimic import MimicCohortSource as _Impl

    return _Impl(*args, **kwargs)
