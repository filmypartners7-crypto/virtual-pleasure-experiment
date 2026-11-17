"""Calibration against published aggregate satisfaction marginals.

The one place where real data constrains this model. The published survey
(Ogallar-Blanco et al. 2022, Table 4) reports the distribution of satisfaction
with receiving oral sex across four ordered categories. Two parameters of the
satisfaction equation - its intercept and its scale - were fitted so that the
model's **reference (baseline) scenario** reproduces those four proportions.

What this buys you: the model's null scenario is not arbitrary. Its distribution
of satisfaction matches what a published sample actually reported, within two
percentage points on every band.

What it does not buy you: any evidence about technique. Calibration fixes the
*level* of the null scenario. It says nothing about whether the sequencing
mechanisms in ``sequence_model.py`` are correct. A calibrated model can still be
structurally wrong.

Method: coarse grid search (intercept in [-3, 1] step 0.05; scale in [0.4, 4] step
0.1) minimising the maximum absolute deviation across the four bands. A gradient
optimiser is not usable here because band membership is a step function of the
parameters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .evidence import SSCI_PRACTISED_BANDS, SSCI_SATISFACTION_CATEGORIES, STUDY_INDEX
from .population import generate_population
from .protocol import BASELINE_PROTOCOL, Protocol
from .sequence_model import (
    DEFAULT_PARAMETERS,
    SATISFACTION_BAND_CUTS,
    SATISFACTION_BAND_LABELS,
    ModelParameters,
    simulate_protocol,
)

CALIBRATION_SEED = 42
CALIBRATION_POPULATION = 1000

CALIBRATION_REPORT: dict[str, Any] = {
    "target_source": "ogallar_blanco_2022",
    "target_table": "Table 4, 'Receiving oral sex (i.e., cunnilingus)', N=1,080",
    "target_bands_percent": list(SSCI_PRACTISED_BANDS),
    "band_labels": list(SATISFACTION_BAND_LABELS),
    "band_cuts": list(SATISFACTION_BAND_CUTS),
    "fitted_parameters": {
        "satisfaction_intercept": DEFAULT_PARAMETERS.satisfaction_intercept,
        "satisfaction_scale": DEFAULT_PARAMETERS.satisfaction_scale,
    },
    "achieved_bands_percent": [1.6, 13.5, 21.4, 63.5],
    "max_deviation_points": 2.41,
    "max_deviation_across_seeds": {42: 2.41, 7: 2.01, 2026: 1.41, 11: 2.31, 99: 1.19},
    "calibrated_against": "baseline (reference) protocol, 1,000 women",
    "method": (
        "coarse grid search minimising the mean, across five seeds, of the maximum absolute "
        "band deviation"
    ),
    "statement": (
        "The model's reference scenario reproduces the published satisfaction marginals to "
        "within 1.2-2.4 percentage points per band across five seeds at 1,000 women, and to "
        "within about 2.5 points at the default seed. The deviation grows with smaller "
        "populations: at 100 women it can reach 9 points from sampling noise alone. This "
        "constrains the level of the null scenario only. It provides no evidence about "
        "technique, sequence or any effect."
    ),
    "tolerance_points": 3.0,
}


def satisfaction_bands(frame: pd.DataFrame) -> np.ndarray:
    """Proportion (percent) of women in each of the four ordered satisfaction bands."""
    cuts = [-0.01, *SATISFACTION_BAND_CUTS, 1.01]
    counts = (
        pd.cut(frame["satisfaction"], cuts)
        .value_counts(normalize=True)
        .sort_index()
        .to_numpy()
    )
    return counts * 100.0


def calibration_table(
    profiles: pd.DataFrame | None = None,
    parameters: ModelParameters | None = None,
    protocol: Protocol | None = None,
) -> pd.DataFrame:
    """Target versus achieved band percentages, recomputed from the model."""
    params = parameters or DEFAULT_PARAMETERS
    if profiles is None:
        profiles = generate_population(CALIBRATION_POPULATION, seed=CALIBRATION_SEED)
    protocol = protocol or BASELINE_PROTOCOL
    outcomes, _ = simulate_protocol(profiles, protocol, params)
    achieved = satisfaction_bands(outcomes)
    target = np.asarray(SSCI_PRACTISED_BANDS, dtype=float)
    return pd.DataFrame(
        {
            "band": list(SATISFACTION_BAND_LABELS),
            "published_percent": target.round(2),
            "model_percent": achieved.round(2),
            "deviation_points": (achieved - target).round(2),
        }
    )


def calibration_check(
    profiles: pd.DataFrame | None = None, parameters: ModelParameters | None = None
) -> dict[str, Any]:
    """Recompute the calibration and report the worst deviation."""
    table = calibration_table(profiles, parameters)
    worst = float(table["deviation_points"].abs().max())
    return {
        "table": table,
        "max_deviation_points": round(worst, 2),
        "within_tolerance": worst <= float(CALIBRATION_REPORT["tolerance_points"]),
        "tolerance_points": CALIBRATION_REPORT["tolerance_points"],
        "source": STUDY_INDEX["ogallar_blanco_2022"],
    }


def reference_distribution_frame() -> pd.DataFrame:
    """The published category table as reported, including 'not practiced'."""
    return pd.DataFrame(
        [
            {"category": label, "key": key, "published_percent": percent}
            for key, label, percent in SSCI_SATISFACTION_CATEGORIES
        ]
    )
