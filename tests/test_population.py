"""Tests for the synthetic population generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vpe.population import (
    AGE_CATEGORIES,
    AGE_LOWER_BOUND,
    BOUNDED_PROFILE_COLUMNS,
    MINIMUM_ADULT_AGE,
    generate_population,
    population_variability,
)


def test_default_population_has_exactly_1000_couples() -> None:
    profiles = generate_population()
    assert len(profiles) == 1000
    assert profiles["couple_id"].is_unique


def test_every_participant_is_an_adult() -> None:
    profiles = generate_population(2000, seed=7)
    assert set(profiles["male_age_category"]).issubset(set(AGE_CATEGORIES))
    assert set(profiles["female_age_category"]).issubset(set(AGE_CATEGORIES))
    assert profiles["male_age_minimum"].min() >= MINIMUM_ADULT_AGE
    assert profiles["female_age_minimum"].min() >= MINIMUM_ADULT_AGE
    # every declared band really is an adult band
    assert min(AGE_LOWER_BOUND.values()) >= MINIMUM_ADULT_AGE


def test_same_seed_reproduces_the_same_population() -> None:
    first = generate_population(500, seed=123)
    second = generate_population(500, seed=123)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_produce_different_populations() -> None:
    first = generate_population(500, seed=1)
    second = generate_population(500, seed=2)
    assert not np.allclose(
        first["female_comfort_sensitivity"].to_numpy(),
        second["female_comfort_sensitivity"].to_numpy(),
    )
    assert first["couple_id"].tolist() == second["couple_id"].tolist()


def test_profile_variables_stay_within_bounds() -> None:
    profiles = generate_population(1000, seed=99)
    for column in BOUNDED_PROFILE_COLUMNS:
        values = profiles[column].to_numpy(dtype=float)
        assert values.min() >= 0.0, column
        assert values.max() <= 1.0, column
    durations = profiles["relationship_duration_years"].to_numpy(dtype=float)
    assert durations.min() >= 0.0
    assert durations.max() <= 50.0


def test_population_preserves_individual_variation() -> None:
    profiles = generate_population(1000, seed=5)
    variability = population_variability(profiles)
    assert (variability["sd"] > 0.05).all()
    assert variability["min"].max() < 1.0
    assert variability["max"].min() > 0.0


def test_no_personal_identifiers_are_generated() -> None:
    profiles = generate_population(200, seed=3)
    forbidden = {"name", "first_name", "last_name", "email", "phone", "address", "ssn"}
    assert not forbidden & {column.lower() for column in profiles.columns}


@pytest.mark.parametrize("bad_size", [0, 1, -5, 100_001])
def test_invalid_population_sizes_are_rejected(bad_size: int) -> None:
    with pytest.raises(ValueError):
        generate_population(bad_size)


@pytest.mark.parametrize("bad_size", [1.5, "1000", None, True])
def test_non_integer_population_sizes_are_rejected(bad_size: object) -> None:
    with pytest.raises(ValueError):
        generate_population(bad_size)  # type: ignore[arg-type]
