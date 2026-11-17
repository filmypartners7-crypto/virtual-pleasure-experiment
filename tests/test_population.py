"""Synthetic population: 1,000 adults, genuinely heterogeneous, reproducible."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vpe.population import (
    ARCHETYPES,
    MINIMUM_ADULT_AGE,
    NUMERIC_PROFILE_COLUMNS,
    generate_population,
    population_variability,
)


def test_default_population_is_one_thousand_women() -> None:
    profiles = generate_population()
    assert len(profiles) == 1000
    assert profiles["woman_id"].is_unique


def test_every_woman_is_an_adult() -> None:
    profiles = generate_population(2000, seed=7)
    assert profiles["age_minimum"].min() >= MINIMUM_ADULT_AGE


def test_population_contains_multiple_archetypes() -> None:
    profiles = generate_population(1000, seed=3)
    counts = profiles["archetype"].value_counts()
    assert len(counts) == len(ARCHETYPES)
    assert counts.min() > 40


def test_responders_are_not_identical() -> None:
    profiles = generate_population(1000, seed=4)
    for column in NUMERIC_PROFILE_COLUMNS:
        assert profiles[column].std() > 0.01, column
    variability = population_variability(profiles)
    assert (variability["sd"] > 0.01).all()


def test_archetypes_differ_from_each_other() -> None:
    profiles = generate_population(4000, seed=5)
    means = profiles.groupby("archetype")["preferred_intensity"].mean()
    assert means.max() - means.min() > 0.2


def test_same_seed_reproduces_the_population() -> None:
    pd.testing.assert_frame_equal(generate_population(500, seed=11), generate_population(500, seed=11))


def test_different_seeds_differ() -> None:
    a = generate_population(500, seed=1)
    b = generate_population(500, seed=2)
    assert not np.allclose(a["preferred_intensity"].to_numpy(), b["preferred_intensity"].to_numpy())


def test_parameters_stay_in_range() -> None:
    profiles = generate_population(1000, seed=9)
    for column in ("preferred_intensity", "preferred_directness", "overstim_threshold"):
        assert profiles[column].between(0.0, 1.0).all(), column
    assert profiles["warmup_need_min"].between(0.0, 12.0).all()


def test_no_personal_identifiers() -> None:
    profiles = generate_population(200, seed=2)
    forbidden = {"name", "email", "phone", "address", "ssn"}
    assert not forbidden & {c.lower() for c in profiles.columns}


@pytest.mark.parametrize("bad", [0, 1, -4, 100_001])
def test_invalid_sizes_rejected(bad: int) -> None:
    with pytest.raises(ValueError):
        generate_population(bad)


@pytest.mark.parametrize("bad", [1.5, "1000", None, True])
def test_non_integer_sizes_rejected(bad) -> None:
    with pytest.raises(ValueError):
        generate_population(bad)
