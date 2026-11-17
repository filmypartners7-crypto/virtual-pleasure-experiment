"""Statistical analysis for synthetic paired experiments.

Everything in this module operates on *model output*. The intervals computed here
are bootstrap percentile intervals **conditional on the model**; they are never
called confidence intervals, and they do not include the uncertainty created by
the model's assumptions (which is larger and is not quantified here).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .model_config import COEFFICIENT_SPECS, ModelConfig, clamp_parameter
from .protocol import Protocol
from .response_model import make_noise, outcome_columns, simulate_outcomes

INTERVAL_LABEL = "bootstrap percentile interval (conditional on the model)"

DEFAULT_OUTCOMES: list[str] = [
    "subjective_arousal",
    "pleasure",
    "comfort",
    "satisfaction",
    "discomfort_risk",
]


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #


def summarise(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Mean, median, standard deviation and quantiles for each outcome column."""
    columns = list(columns) if columns is not None else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c in DEFAULT_OUTCOMES
    ]
    rows = []
    for column in columns:
        values = df[column].to_numpy(dtype=float)
        rows.append(
            {
                "outcome": column,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def paired_differences(
    baseline: pd.DataFrame,
    guide: pd.DataFrame,
    columns: Sequence[str] | None = None,
    id_column: str = "couple_id",
) -> pd.DataFrame:
    """Build a paired table: baseline value, guide value and difference per couple.

    Raises
    ------
    ValueError
        If the two conditions do not describe exactly the same couples.
    """
    columns = list(columns) if columns is not None else list(DEFAULT_OUTCOMES)
    base_ids = baseline[id_column].tolist()
    guide_ids = guide[id_column].tolist()
    if base_ids != guide_ids:
        raise ValueError(
            "baseline and guide conditions must contain exactly the same couples in the "
            "same order (paired design)"
        )

    out = pd.DataFrame({id_column: base_ids})
    for column in columns:
        base_values = baseline[column].to_numpy(dtype=float)
        guide_values = guide[column].to_numpy(dtype=float)
        out[f"baseline_{column}"] = base_values
        out[f"guide_{column}"] = guide_values
        out[f"diff_{column}"] = guide_values - base_values
    return out


def condition_summary(
    baseline: pd.DataFrame,
    guide: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Side-by-side mean/median/sd per condition plus the mean paired difference."""
    columns = list(columns) if columns is not None else list(DEFAULT_OUTCOMES)
    rows = []
    for column in columns:
        base_values = baseline[column].to_numpy(dtype=float)
        guide_values = guide[column].to_numpy(dtype=float)
        diff = guide_values - base_values
        rows.append(
            {
                "outcome": column,
                "baseline_mean": float(base_values.mean()),
                "baseline_median": float(np.median(base_values)),
                "baseline_sd": float(base_values.std(ddof=1)),
                "guide_mean": float(guide_values.mean()),
                "guide_median": float(np.median(guide_values)),
                "guide_sd": float(guide_values.std(ddof=1)),
                "mean_paired_diff": float(diff.mean()),
                "median_paired_diff": float(np.median(diff)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Bootstrap uncertainty
# --------------------------------------------------------------------------- #


def bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap interval for the *mean* of ``values``.

    This is a property of the simulation given its assumptions. It is not a
    confidence interval for anything in the real world.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100 for a stable percentile interval")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean": float(values.mean()),
        "low": float(low),
        "high": float(high),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
    }


def paired_interval_table(
    paired: pd.DataFrame,
    columns: Sequence[str] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap intervals for the mean paired difference of each outcome."""
    columns = list(columns) if columns is not None else list(DEFAULT_OUTCOMES)
    rows = []
    for column in columns:
        diff_column = f"diff_{column}"
        stats = bootstrap_interval(paired[diff_column], n_boot=n_boot, alpha=alpha, seed=seed)
        rows.append(
            {
                "outcome": column,
                "mean_paired_diff": stats["mean"],
                "interval_low": stats["low"],
                "interval_high": stats["high"],
                "interval_type": INTERVAL_LABEL,
                "excludes_zero": bool(stats["low"] > 0 or stats["high"] < 0),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sensitivity analysis
# --------------------------------------------------------------------------- #


def sensitivity_analysis(
    profiles: pd.DataFrame,
    protocol: Protocol,
    config: ModelConfig,
    seed: int,
    outcomes: Sequence[str] | None = None,
    relative_delta: float = 0.20,
    parameters: Iterable[str] | None = None,
) -> pd.DataFrame:
    """How much each coefficient moves the mean outcome, one at a time.

    Each coefficient is moved by +/- ``relative_delta`` (clipped to its documented
    range) and the model is re-run with the *same* noise draws, so the reported
    change is caused by the coefficient alone.

    This is a property of the equations. It is not causal evidence about people.
    """
    if not (0.0 < relative_delta < 1.0):
        raise ValueError("relative_delta must lie strictly between 0 and 1")

    names = list(parameters) if parameters is not None else list(config.numeric_items())
    outcomes = list(outcomes) if outcomes is not None else outcome_columns(config)

    rng = np.random.default_rng(seed)
    noise = make_noise(len(profiles), rng, include_orgasm=config.include_orgasm_estimate)

    base = simulate_outcomes(profiles, protocol, config, noise)
    base_means = {outcome: float(base[outcome].mean()) for outcome in outcomes}

    rows = []
    for name in names:
        if name not in COEFFICIENT_SPECS:
            raise ValueError(f"unknown model parameter: {name}")
        current = float(getattr(config, name))
        for direction in (-1.0, 1.0):
            candidate = current * (1.0 + relative_delta * direction)
            if current == 0.0:
                candidate = relative_delta * direction
            candidate = clamp_parameter(name, candidate)
            if np.isclose(candidate, current):
                continue
            perturbed_config = config.with_overrides(**{name: candidate})
            perturbed = simulate_outcomes(profiles, protocol, perturbed_config, noise)
            row: dict[str, float | str] = {
                "parameter": name,
                "direction": "lower" if direction < 0 else "higher",
                "base_value": current,
                "perturbed_value": candidate,
            }
            for outcome in outcomes:
                row[f"delta_{outcome}"] = float(perturbed[outcome].mean()) - base_means[outcome]
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.attrs["base_means"] = base_means
    frame.attrs["relative_delta"] = relative_delta
    frame.attrs["note"] = (
        "Sensitivity of the model to its own assumptions. Not causal evidence about people."
    )
    return frame


# --------------------------------------------------------------------------- #
# Subgroups
# --------------------------------------------------------------------------- #


def subgroup_summary(
    paired: pd.DataFrame,
    profiles: pd.DataFrame,
    by: str,
    outcome: str = "pleasure",
    bins: int = 3,
    labels: Sequence[str] = ("low", "middle", "high"),
) -> pd.DataFrame:
    """Compare the paired difference across quantile bands of a profile variable."""
    if by not in profiles.columns:
        raise ValueError(f"unknown profile variable: {by}")
    if f"diff_{outcome}" not in paired.columns:
        raise ValueError(f"unknown outcome in paired table: {outcome}")
    if len(paired) != len(profiles):
        raise ValueError("paired table and profile table must have the same number of rows")

    values = profiles[by]
    if pd.api.types.is_numeric_dtype(values):
        bands = pd.qcut(values, q=bins, labels=list(labels)[:bins], duplicates="drop")
        band_labels = [str(b) for b in bands]
    else:
        band_labels = values.astype(str).tolist()

    frame = pd.DataFrame(
        {
            "subgroup": band_labels,
            f"diff_{outcome}": paired[f"diff_{outcome}"].to_numpy(dtype=float),
            f"baseline_{outcome}": paired[f"baseline_{outcome}"].to_numpy(dtype=float),
            f"guide_{outcome}": paired[f"guide_{outcome}"].to_numpy(dtype=float),
        }
    )
    grouped = (
        frame.groupby("subgroup", observed=True)
        .agg(
            n=(f"diff_{outcome}", "size"),
            mean_paired_diff=(f"diff_{outcome}", "mean"),
            sd_paired_diff=(f"diff_{outcome}", "std"),
            baseline_mean=(f"baseline_{outcome}", "mean"),
            guide_mean=(f"guide_{outcome}", "mean"),
        )
        .reset_index()
    )
    grouped["sd_paired_diff"] = grouped["sd_paired_diff"].fillna(0.0)
    grouped.insert(1, "grouped_by", by)
    return grouped
