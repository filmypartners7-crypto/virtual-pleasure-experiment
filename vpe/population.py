"""Synthetic population generator for the Virtual Pleasure Experiment.

The generator produces *couples*. Each couple has one synthetic male profile and
one synthetic female profile, both restricted to adults (18+). Most profile
characteristics use hypothetical distributions. One ordinal satisfaction category
is resampled from published aggregate survey percentages; no participant-level
records are included or reproduced.

Every variable is bounded in [0, 1] except the age category (a label) and the
relationship duration (years). The distributions are declared in
:data:`POPULATION_DISTRIBUTIONS` so the dashboard and the README can display them
next to the data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .research import sample_reference_profiles

# --------------------------------------------------------------------------- #
# Age categories - adults only
# --------------------------------------------------------------------------- #

AGE_CATEGORIES: list[str] = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_LOWER_BOUND: dict[str, int] = {
    "18-24": 18,
    "25-34": 25,
    "35-44": 35,
    "45-54": 45,
    "55-64": 55,
    "65+": 65,
}
# Hypothetical sampling weights. These are NOT population prevalence claims.
AGE_CATEGORY_WEIGHTS: list[float] = [0.14, 0.30, 0.24, 0.17, 0.10, 0.05]

MINIMUM_ADULT_AGE = 18


# --------------------------------------------------------------------------- #
# Documented distributions
# --------------------------------------------------------------------------- #

POPULATION_DISTRIBUTIONS: dict[str, dict[str, Any]] = {
    "male_age_category": {
        "group": "male profile",
        "distribution": "Categorical",
        "support": ", ".join(AGE_CATEGORIES),
        "meaning": "Adult age band of the synthetic male partner (18+ only).",
        "note": "Weights are a demonstration choice, not prevalence data.",
    },
    "female_age_category": {
        "group": "female profile",
        "distribution": "Categorical",
        "support": ", ".join(AGE_CATEGORIES),
        "meaning": "Adult age band of the synthetic female partner (18+ only).",
        "note": "Weights are a demonstration choice, not prevalence data.",
    },
    "male_experience_level": {
        "group": "male profile",
        "distribution": "Beta(2.2, 2.2)",
        "support": "[0, 1]",
        "meaning": "How much relevant experience the synthetic profile carries.",
        "note": "Beta parameters are arbitrary and give a broad, symmetric spread.",
    },
    "male_communication_responsiveness": {
        "group": "male profile",
        "distribution": "Beta(2.6, 2.0)",
        "support": "[0, 1]",
        "meaning": "Willingness and ability to respond to a partner's verbal cues.",
        "note": "Simulated only.",
    },
    "male_adherence_variability": {
        "group": "male profile",
        "distribution": "Beta(2.0, 3.0)",
        "support": "[0, 1]",
        "meaning": "Inconsistency in following a structured approach (higher = less consistent).",
        "note": "Simulated only. Enters the model as erosion of declared adherence.",
    },
    "male_feedback_responsiveness": {
        "group": "male profile",
        "distribution": "Beta(2.4, 2.0)",
        "support": "[0, 1]",
        "meaning": "Ability to notice and act on partner feedback in the moment.",
        "note": "Simulated only.",
    },
    "female_biological_response_variability": {
        "group": "female profile",
        "distribution": "Beta(2.5, 2.5)",
        "support": "[0, 1]",
        "meaning": "Individual variability in physiological response tendency.",
        "note": "A model input only; it is not a clinical measure.",
    },
    "female_preference_specificity": {
        "group": "female profile",
        "distribution": "Beta(2.2, 2.0)",
        "support": "[0, 1]",
        "meaning": "How narrow and specific this profile's preferences are.",
        "note": "Higher values make adaptation matter more in the model.",
    },
    "female_psychological_context": {
        "group": "female profile",
        "distribution": "Beta(3.0, 2.0)",
        "support": "[0, 1]",
        "meaning": "Contextual / psychological state brought into the scenario (stress, mood, safety).",
        "note": "A single lumped variable: a simplification of a very complex reality.",
    },
    "female_communication_preference": {
        "group": "female profile",
        "distribution": "Beta(2.5, 2.2)",
        "support": "[0, 1]",
        "meaning": "How readily this profile communicates preferences and limits.",
        "note": "Interacts with male feedback responsiveness.",
    },
    "female_comfort_sensitivity": {
        "group": "female profile",
        "distribution": "Beta(2.0, 2.6)",
        "support": "[0, 1]",
        "meaning": "How strongly this profile reacts to (in)adequate comfort conditions.",
        "note": "Enters comfort negatively and discomfort risk positively.",
    },
    "relationship_duration_years": {
        "group": "couple",
        "distribution": "Log-normal(mean_log=1.6, sd_log=0.7), clipped to [0, 50]",
        "support": "[0, 50]",
        "meaning": "Synthetic duration of the couple's relationship, in years.",
        "note": "Not currently used in the response equations; provided for subgroup exploration.",
    },
    "empirical_reference_category": {
        "group": "female profile (empirical reference anchor)",
        "distribution": "Categorical resampling of published aggregate percentages",
        "support": "Not practiced; not satisfactory; somewhat; quite; very satisfactory",
        "meaning": "Synthetic reproduction of the published category margins for receiving oral sex (cunnilingus).",
        "note": "Derived from N=1,080 Spanish women aged 18-50 in a voluntary non-probability online sample; not globally representative and not participant-level data.",
    },
    "empirical_reference_latent_utility": {
        "group": "female profile (synthetic ordinal-model variable)",
        "distribution": "Logistic latent value sampled conditionally within source-derived ordinal category cut-points",
        "support": "Unbounded real values; NaN for not-practiced category",
        "meaning": "Mathematical latent value used only to explore a hypothetical ordinal shift scenario.",
        "note": "Not measured in the published survey. This latent variable is a modelling device, not empirical participant data.",
    },
}

PROFILE_COLUMNS: list[str] = [
    "male_experience_level",
    "male_communication_responsiveness",
    "male_adherence_variability",
    "male_feedback_responsiveness",
    "female_biological_response_variability",
    "female_preference_specificity",
    "female_psychological_context",
    "female_communication_preference",
    "female_comfort_sensitivity",
]

BOUNDED_PROFILE_COLUMNS: list[str] = PROFILE_COLUMNS


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


def _validate_size(n_couples: int) -> int:
    if isinstance(n_couples, bool) or not isinstance(n_couples, (int, np.integer)):
        raise ValueError("population size must be an integer")
    n_couples = int(n_couples)
    if n_couples < 2:
        raise ValueError("population size must be at least 2 couples")
    if n_couples > 100_000:
        raise ValueError("population size above 100000 is not supported by this demonstration")
    return n_couples


def _beta(rng: np.random.Generator, n: int, a: float, b: float) -> np.ndarray:
    """Beta draws, clipped into [0, 1] for numerical safety."""
    return np.clip(rng.beta(a, b, size=n), 0.0, 1.0)


def generate_population(n_couples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate ``n_couples`` synthetic adult couples.

    Parameters
    ----------
    n_couples:
        Number of couples. Default 1000, as required by the specification.
    seed:
        Seed for the reproducible random number generator.

    Returns
    -------
    pandas.DataFrame
        One row per couple. Column ``couple_id`` is a stable identifier used to
        pair the two experimental conditions.
    """
    n = _validate_size(n_couples)
    rng = np.random.default_rng(seed)

    frame = pd.DataFrame(
        {
            "couple_id": [f"C{i:05d}" for i in range(1, n + 1)],
            "male_age_category": rng.choice(AGE_CATEGORIES, size=n, p=AGE_CATEGORY_WEIGHTS),
            "female_age_category": rng.choice(AGE_CATEGORIES, size=n, p=AGE_CATEGORY_WEIGHTS),
            "male_experience_level": _beta(rng, n, 2.2, 2.2),
            "male_communication_responsiveness": _beta(rng, n, 2.6, 2.0),
            "male_adherence_variability": _beta(rng, n, 2.0, 3.0),
            "male_feedback_responsiveness": _beta(rng, n, 2.4, 2.0),
            "female_biological_response_variability": _beta(rng, n, 2.5, 2.5),
            "female_preference_specificity": _beta(rng, n, 2.2, 2.0),
            "female_psychological_context": _beta(rng, n, 3.0, 2.0),
            "female_communication_preference": _beta(rng, n, 2.5, 2.2),
            "female_comfort_sensitivity": _beta(rng, n, 2.0, 2.6),
        }
    )

    durations = rng.lognormal(mean=1.6, sigma=0.7, size=n)
    frame["relationship_duration_years"] = np.clip(durations, 0.0, 50.0)

    # Add a synthetic ordinal response category sampled from published aggregate
    # percentages. This is not a copy of the original respondents or microdata.
    reference = sample_reference_profiles(n, rng)
    frame = pd.concat([frame.reset_index(drop=True), reference], axis=1)

    frame["male_age_minimum"] = frame["male_age_category"].map(AGE_LOWER_BOUND)
    frame["female_age_minimum"] = frame["female_age_category"].map(AGE_LOWER_BOUND)

    # Hard invariant: nobody in this population is a minor.
    if frame[["male_age_minimum", "female_age_minimum"]].to_numpy().min() < MINIMUM_ADULT_AGE:
        raise RuntimeError("population generator produced a non-adult profile")

    return frame


def population_variability(profiles: pd.DataFrame) -> pd.DataFrame:
    """Describe the spread of the bounded profile variables."""
    rows = []
    for column in BOUNDED_PROFILE_COLUMNS:
        values = profiles[column].to_numpy(dtype=float)
        rows.append(
            {
                "variable": column,
                "mean": round(float(values.mean()), 4),
                "sd": round(float(values.std(ddof=1)), 4),
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
            }
        )
    return pd.DataFrame(rows)


def distributions_table() -> pd.DataFrame:
    """The documented distribution table, for display and export."""
    return pd.DataFrame(
        [
            {
                "variable": name,
                "profile": spec["group"],
                "distribution": spec["distribution"],
                "support": spec["support"],
                "meaning": spec["meaning"],
                "note": spec["note"],
            }
            for name, spec in POPULATION_DISTRIBUTIONS.items()
        ]
    )
