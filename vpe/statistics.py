"""Statistics for protocol comparisons on a shared population.

Every interval here is a bootstrap percentile interval **conditional on the
model**. It describes simulation noise across the 1,000 synthetic women. It is
not a confidence interval for anything in the world, and it does not include
uncertainty in the model's assumptions - which is larger and is explored
separately by the sensitivity analysis.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .sequence_model import OUTCOME_COLUMNS

INTERVAL_LABEL = "bootstrap percentile interval (conditional on the model)"


def summarise(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Mean, median, SD and quantiles per outcome."""
    columns = list(columns) if columns is not None else list(OUTCOME_COLUMNS)
    rows = []
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        rows.append(
            {
                "outcome": column,
                "n": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "q05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "q95": float(np.quantile(values, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def paired_differences(
    condition_a: pd.DataFrame,
    condition_b: pd.DataFrame,
    columns: Sequence[str] | None = None,
    id_column: str = "woman_id",
) -> pd.DataFrame:
    """B - A per woman. Both conditions must describe the same women in order."""
    columns = list(columns) if columns is not None else list(OUTCOME_COLUMNS)
    if condition_a[id_column].tolist() != condition_b[id_column].tolist():
        raise ValueError(
            "both conditions must contain the same women in the same order (paired design)"
        )
    out = pd.DataFrame({id_column: condition_a[id_column].tolist()})
    for column in columns:
        a = condition_a[column].to_numpy(dtype=float)
        b = condition_b[column].to_numpy(dtype=float)
        out[f"a_{column}"] = a
        out[f"b_{column}"] = b
        out[f"diff_{column}"] = b - a
    return out


def bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
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
    columns = list(columns) if columns is not None else list(OUTCOME_COLUMNS)
    rows = []
    for column in columns:
        stats = bootstrap_interval(paired[f"diff_{column}"], n_boot=n_boot, alpha=alpha, seed=seed)
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


def condition_summary(
    condition_a: pd.DataFrame,
    condition_b: pd.DataFrame,
    label_a: str = "A",
    label_b: str = "B",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Side-by-side means plus the mean paired difference."""
    columns = list(columns) if columns is not None else list(OUTCOME_COLUMNS)
    rows = []
    for column in columns:
        a = condition_a[column].to_numpy(dtype=float)
        b = condition_b[column].to_numpy(dtype=float)
        rows.append(
            {
                "outcome": column,
                f"{label_a}_mean": float(a.mean()),
                f"{label_a}_median": float(np.median(a)),
                f"{label_a}_sd": float(a.std(ddof=1)),
                f"{label_b}_mean": float(b.mean()),
                f"{label_b}_median": float(np.median(b)),
                f"{label_b}_sd": float(b.std(ddof=1)),
                "mean_paired_diff": float((b - a).mean()),
                "share_improved_percent": float((b > a).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def subgroup_summary(
    paired: pd.DataFrame,
    profiles: pd.DataFrame,
    by: str,
    outcome: str = "pleasure",
) -> pd.DataFrame:
    """Compare the paired difference across groups of the synthetic population."""
    if by not in profiles.columns:
        raise ValueError(f"unknown grouping variable: {by}")
    if f"diff_{outcome}" not in paired.columns:
        raise ValueError(f"unknown outcome: {outcome}")
    if len(paired) != len(profiles):
        raise ValueError("paired table and profiles must have the same number of rows")

    frame = pd.DataFrame(
        {
            "subgroup": profiles[by].astype(str).to_numpy(),
            "diff": paired[f"diff_{outcome}"].to_numpy(dtype=float),
            "a": paired[f"a_{outcome}"].to_numpy(dtype=float),
            "b": paired[f"b_{outcome}"].to_numpy(dtype=float),
        }
    )
    grouped = (
        frame.groupby("subgroup")
        .agg(
            n=("diff", "size"),
            mean_diff=("diff", "mean"),
            sd_diff=("diff", "std"),
            condition_a_mean=("a", "mean"),
            condition_b_mean=("b", "mean"),
            share_improved_percent=("diff", lambda s: float((s > 0).mean() * 100.0)),
        )
        .reset_index()
    )
    grouped["sd_diff"] = grouped["sd_diff"].fillna(0.0)
    grouped.insert(1, "grouped_by", by)
    return grouped.sort_values("mean_diff", ascending=False).reset_index(drop=True)


def sensitivity_analysis(
    profiles: pd.DataFrame,
    protocol,
    parameters,
    outcomes: Sequence[str] | None = None,
    relative_delta: float = 0.20,
    parameters_to_perturb: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One-at-a-time perturbation of the model's coefficients.

    A property of the equations. Not causal evidence about people.
    """
    from .sequence_model import simulate_protocol

    if not (0.0 < relative_delta < 1.0):
        raise ValueError("relative_delta must lie strictly between 0 and 1")

    outcomes = list(outcomes) if outcomes is not None else ["pleasure", "comfort", "discomfort", "satisfaction"]
    names = (
        list(parameters_to_perturb)
        if parameters_to_perturb is not None
        else list(parameters.numeric_items())
    )

    base, _ = simulate_protocol(profiles, protocol, parameters)
    base_means = {outcome: float(base[outcome].mean()) for outcome in outcomes}

    rows = []
    for name in names:
        current = float(getattr(parameters, name))
        if name == "dt_min":
            continue
        for direction in (-1.0, 1.0):
            candidate = current * (1.0 + relative_delta * direction)
            if current == 0.0:
                candidate = relative_delta * direction
            if np.isclose(candidate, current):
                continue
            try:
                perturbed_params = parameters.with_overrides(**{name: candidate})
            except (TypeError, ValueError):
                continue
            perturbed, _ = simulate_protocol(profiles, protocol, perturbed_params)
            row = {
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
    frame.attrs["note"] = (
        "One-at-a-time sensitivity of the model to its own assumptions. "
        "Not causal evidence about people."
    )
    return frame
