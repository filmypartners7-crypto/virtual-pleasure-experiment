"""Synthetic population of adult women for VPE 0.3.0.

1,000 fictional women by default. Each is generated from a latent preference
*archetype* plus individual variation, so the population contains genuinely
different responders rather than one generic profile with noise on top.

Evidence status: the existence of large preference heterogeneity is
research-supported (Herbenick et al. 2018). The specific archetypes, their
weights and their parameter values are a modelling device - they are **not**
prevalence estimates and no source claims that 22% of women are one type.

Every woman is an adult (18+) by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MINIMUM_ADULT_AGE = 18
AGE_BANDS: tuple[str, ...] = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
AGE_BAND_WEIGHTS: tuple[float, ...] = (0.16, 0.30, 0.24, 0.16, 0.09, 0.05)
AGE_BAND_MINIMUM: dict[str, int] = {
    "18-24": 18, "25-34": 25, "35-44": 35, "45-54": 45, "55-64": 55, "65+": 65,
}


@dataclass(frozen=True)
class Archetype:
    """A latent responder type. A modelling device, not a taxonomy of real women."""

    key: str
    label: str
    weight: float
    preferred_intensity: float
    preferred_directness: float
    warmup_need_min: float
    overstim_threshold: float
    rhythm_dependence: float
    comfort_sensitivity: float
    communication_openness: float
    description: str


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        key="direct_firm",
        label="Prefers direct and firm",
        weight=0.22,
        preferred_intensity=0.72,
        preferred_directness=0.85,
        warmup_need_min=2.0,
        overstim_threshold=0.85,
        rhythm_dependence=0.50,
        comfort_sensitivity=0.35,
        communication_openness=0.60,
        description="Wants direct glans contact early and higher intensity; tolerates a lot before overstimulation.",
    ),
    Archetype(
        key="indirect_gentle",
        label="Prefers indirect and gentle",
        weight=0.20,
        preferred_intensity=0.42,
        preferred_directness=0.45,
        warmup_need_min=4.0,
        overstim_threshold=0.58,
        rhythm_dependence=0.60,
        comfort_sensitivity=0.60,
        communication_openness=0.55,
        description="Prefers work through the hood at lower intensity; direct glans contact too early is aversive.",
    ),
    Archetype(
        key="sustained_rhythm",
        label="Needs sustained rhythm",
        weight=0.20,
        preferred_intensity=0.58,
        preferred_directness=0.65,
        warmup_need_min=3.0,
        overstim_threshold=0.75,
        rhythm_dependence=0.92,
        comfort_sensitivity=0.45,
        communication_openness=0.50,
        description="Moderate preferences but strongly penalised by constant variation; needs one rhythm held.",
    ),
    Archetype(
        key="broad_stimulation",
        label="Prefers broad, diffuse stimulation",
        weight=0.18,
        preferred_intensity=0.55,
        preferred_directness=0.35,
        warmup_need_min=5.0,
        overstim_threshold=0.70,
        rhythm_dependence=0.50,
        comfort_sensitivity=0.50,
        communication_openness=0.50,
        description="Responds to labial and perineal contact; narrow, pointed stimulation is not what works.",
    ),
    Archetype(
        key="highly_sensitive",
        label="Highly sensitive",
        weight=0.20,
        preferred_intensity=0.32,
        preferred_directness=0.38,
        warmup_need_min=6.0,
        overstim_threshold=0.42,
        rhythm_dependence=0.70,
        comfort_sensitivity=0.82,
        communication_openness=0.65,
        description="Low overstimulation threshold and a long warm-up need; intensity escalates into discomfort quickly.",
    ),
)

ARCHETYPE_INDEX: dict[str, Archetype] = {a.key: a for a in ARCHETYPES}

POPULATION_VARIABLES: dict[str, dict[str, str]] = {
    "age_band": {"distribution": "Categorical", "meaning": "Adult age band (18+ only)."},
    "archetype": {"distribution": "Categorical", "meaning": "Latent preference archetype (modelling device)."},
    "preferred_intensity": {"distribution": "Archetype mean + N(0, 0.08), clipped [0.05, 0.95]", "meaning": "Intensity at which stimulation feels best."},
    "preferred_directness": {"distribution": "Archetype mean + N(0, 0.10), clipped [0.05, 0.95]", "meaning": "How directly she wants the glans addressed."},
    "intensity_tolerance": {"distribution": "Beta-derived, 0.12-0.40", "meaning": "Width of the acceptable intensity band."},
    "warmup_need_min": {"distribution": "Archetype mean + N(0, 1.2) minutes, clipped [0, 12]", "meaning": "Minutes of indirect stimulation before direct contact is comfortable."},
    "overstim_threshold": {"distribution": "Archetype mean + N(0, 0.08), clipped [0.15, 0.98]", "meaning": "Point where more intensity turns aversive."},
    "rhythm_dependence": {"distribution": "Archetype mean + N(0, 0.10), clipped [0, 1]", "meaning": "Benefit from a held rhythm."},
    "comfort_sensitivity": {"distribution": "Archetype mean + N(0, 0.10), clipped [0, 1]", "meaning": "Sensitivity to poor comfort conditions."},
    "communication_openness": {"distribution": "Archetype mean + N(0, 0.12), clipped [0, 1]", "meaning": "How readily she gives feedback during the encounter."},
    "experience_level": {"distribution": "Beta(2.2, 2.2)", "meaning": "Prior experience carried into the encounter."},
    "obligation_sensitivity": {"distribution": "Beta(2.4, 2.0)", "meaning": "How strongly pressure or obligation reduces comfort."},
}


def _validate_size(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError("population size must be an integer")
    n = int(n)
    if n < 2:
        raise ValueError("population size must be at least 2")
    if n > 100_000:
        raise ValueError("population size above 100000 is not supported")
    return n


def generate_population(n_women: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate ``n_women`` synthetic adult women with heterogeneous responders."""
    n = _validate_size(n_women)
    rng = np.random.default_rng(seed)

    weights = np.array([a.weight for a in ARCHETYPES], dtype=float)
    weights = weights / weights.sum()

    # Stratified allocation rather than a multinomial draw: each synthetic
    # population contains the archetype proportions almost exactly, so the
    # calibration target does not drift with sampling noise in the mix.
    raw_counts = weights * n
    counts = np.floor(raw_counts).astype(int)
    remainder = n - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        for i in range(remainder):
            counts[order[i % len(counts)]] += 1
    keys = [a.key for a in ARCHETYPES]
    archetype_keys = np.array(
        [key for key, count in zip(keys, counts) for _ in range(int(count))], dtype=object
    )
    rng.shuffle(archetype_keys)
    archetype_keys = archetype_keys.astype(str)

    def per_woman(param: str, sd: float, low: float, high: float) -> np.ndarray:
        means = np.array([getattr(ARCHETYPE_INDEX[key], param) for key in archetype_keys], dtype=float)
        return np.clip(means + rng.normal(0.0, sd, size=n), low, high)

    frame = pd.DataFrame(
        {
            "woman_id": [f"W{i:05d}" for i in range(1, n + 1)],
            "age_band": rng.choice(AGE_BANDS, size=n, p=AGE_BAND_WEIGHTS),
            "archetype": archetype_keys,
            "preferred_intensity": per_woman("preferred_intensity", 0.08, 0.05, 0.95),
            "preferred_directness": per_woman("preferred_directness", 0.10, 0.05, 0.95),
            "intensity_tolerance": np.clip(rng.beta(3.0, 4.0, size=n) * 0.6 + 0.10, 0.12, 0.40),
            "warmup_need_min": per_woman("warmup_need_min", 1.2, 0.0, 12.0),
            "overstim_threshold": per_woman("overstim_threshold", 0.08, 0.15, 0.98),
            "rhythm_dependence": per_woman("rhythm_dependence", 0.10, 0.0, 1.0),
            "comfort_sensitivity": per_woman("comfort_sensitivity", 0.10, 0.0, 1.0),
            "communication_openness": per_woman("communication_openness", 0.12, 0.0, 1.0),
            "experience_level": np.clip(rng.beta(2.2, 2.2, size=n), 0.0, 1.0),
            "obligation_sensitivity": np.clip(rng.beta(2.4, 2.0, size=n), 0.0, 1.0),
        }
    )
    frame["archetype_label"] = frame["archetype"].map(lambda key: ARCHETYPE_INDEX[key].label)
    frame["age_minimum"] = frame["age_band"].map(AGE_BAND_MINIMUM)

    if frame["age_minimum"].min() < MINIMUM_ADULT_AGE:
        raise RuntimeError("population generator produced a non-adult profile")
    return frame


NUMERIC_PROFILE_COLUMNS: tuple[str, ...] = (
    "preferred_intensity",
    "preferred_directness",
    "intensity_tolerance",
    "warmup_need_min",
    "overstim_threshold",
    "rhythm_dependence",
    "comfort_sensitivity",
    "communication_openness",
    "experience_level",
    "obligation_sensitivity",
)


def archetypes_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "archetype": a.key,
                "label": a.label,
                "weight": a.weight,
                "preferred_intensity": a.preferred_intensity,
                "preferred_directness": a.preferred_directness,
                "warmup_need_min": a.warmup_need_min,
                "overstim_threshold": a.overstim_threshold,
                "rhythm_dependence": a.rhythm_dependence,
                "comfort_sensitivity": a.comfort_sensitivity,
                "description": a.description,
            }
            for a in ARCHETYPES
        ]
    )


def population_variability(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_PROFILE_COLUMNS:
        values = profiles[column].to_numpy(dtype=float)
        rows.append(
            {
                "variable": column,
                "mean": round(float(values.mean()), 3),
                "sd": round(float(values.std(ddof=1)), 3),
                "min": round(float(values.min()), 3),
                "max": round(float(values.max()), 3),
            }
        )
    return pd.DataFrame(rows)


def population_variables_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": name, "distribution": spec["distribution"], "meaning": spec["meaning"]}
            for name, spec in POPULATION_VARIABLES.items()
        ]
    )
