"""Tests for the response engine."""

from __future__ import annotations

import numpy as np
import pytest

from vpe.model_config import COEFFICIENT_SPECS, ModelConfig, default_config
from vpe.population import generate_population
from vpe.protocol import Protocol, default_guide_protocol
from vpe.response_model import (
    ORGASM_COLUMN,
    OUTCOME_COLUMNS,
    equations_text,
    make_noise,
    outcome_columns,
    simulate_outcomes,
)


@pytest.fixture(scope="module")
def profiles():
    return generate_population(300, seed=11)


@pytest.fixture(scope="module")
def noise():
    return make_noise(300, np.random.default_rng(2024))


def test_outcomes_stay_within_their_bounds(profiles, noise) -> None:
    outcomes = simulate_outcomes(profiles, default_guide_protocol(), ModelConfig(), noise)
    for column in OUTCOME_COLUMNS:
        values = outcomes[column].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert values.min() > 0.0
        assert values.max() < 1.0


def test_bounds_hold_with_extreme_coefficients(profiles) -> None:
    extreme = ModelConfig(
        arousal_intercept=3.0,
        pleasure_intercept=3.0,
        comfort_intercept=-3.0,
        satisfaction_intercept=3.0,
        discomfort_intercept=2.0,
        w_pleasure_arousal=3.0,
        w_pleasure_comfort=3.0,
    )
    outcomes = simulate_outcomes(profiles, default_guide_protocol(), extreme)
    for column in OUTCOME_COLUMNS:
        values = outcomes[column].to_numpy(dtype=float)
        assert values.min() >= 0.0
        assert values.max() <= 1.0


def test_model_is_deterministic_given_the_same_noise(profiles, noise) -> None:
    first = simulate_outcomes(profiles, default_guide_protocol(), ModelConfig(), noise)
    second = simulate_outcomes(profiles, default_guide_protocol(), ModelConfig(), noise)
    assert first.equals(second)


def test_a_better_protocol_raises_expected_outcomes(profiles) -> None:
    noise = make_noise(300, np.random.default_rng(5))
    weak = Protocol(
        consistency=0.1,
        feedback_responsiveness=0.1,
        adaptation=0.1,
        communication=0.1,
        comfort_checks=0.1,
        boundary_respect=0.2,
        adherence=0.1,
    )
    strong = Protocol(
        consistency=0.95,
        feedback_responsiveness=0.95,
        adaptation=0.95,
        communication=0.95,
        comfort_checks=0.95,
        boundary_respect=0.98,
        adherence=0.95,
    )
    weak_outcomes = simulate_outcomes(profiles, weak, ModelConfig(), noise)
    strong_outcomes = simulate_outcomes(profiles, strong, ModelConfig(), noise)

    for column in ["pleasure", "comfort", "satisfaction", "subjective_arousal"]:
        assert strong_outcomes[column].mean() > weak_outcomes[column].mean(), column
    assert strong_outcomes["discomfort_risk"].mean() < weak_outcomes["discomfort_risk"].mean()


def test_arousal_is_not_treated_as_pleasure(profiles, noise) -> None:
    outcomes = simulate_outcomes(profiles, default_guide_protocol(), ModelConfig(), noise)
    arousal = outcomes["subjective_arousal"].to_numpy(dtype=float)
    pleasure = outcomes["pleasure"].to_numpy(dtype=float)
    assert not np.allclose(arousal, pleasure)
    correlation = float(np.corrcoef(arousal, pleasure)[0, 1])
    assert 0.3 < correlation < 0.99, correlation
    assert default_config().w_pleasure_arousal < default_config().w_pleasure_preference_match


def test_discomfort_is_a_separate_mechanism(profiles, noise) -> None:
    outcomes = simulate_outcomes(profiles, default_guide_protocol(), ModelConfig(), noise)
    complement = 1.0 - outcomes["comfort"].to_numpy(dtype=float)
    discomfort = outcomes["discomfort_risk"].to_numpy(dtype=float)
    assert not np.allclose(complement, discomfort)
    assert float(np.corrcoef(complement, discomfort)[0, 1]) < 0.95


def test_zero_noise_removes_random_variation(profiles) -> None:
    silent = {channel: np.zeros(300) for channel in ["arousal", "pleasure", "comfort", "satisfaction", "discomfort"]}
    outcomes = simulate_outcomes(profiles, default_guide_protocol(), ModelConfig(), silent)
    same_profiles = profiles.copy()
    outcomes_repeat = simulate_outcomes(same_profiles, default_guide_protocol(), ModelConfig(), silent)
    assert outcomes["pleasure"].equals(outcomes_repeat["pleasure"])


def test_intermediate_terms_are_exposed_and_bounded(profiles) -> None:
    outcomes = simulate_outcomes(profiles, default_guide_protocol(), ModelConfig())
    for column in [
        "adherence_effective",
        "consistency_effective",
        "feedback_loop",
        "communication_effective",
        "preference_match",
        "boundary_effective",
        "comfort_checks_effective",
        "safety_practices",
    ]:
        values = outcomes[column].to_numpy(dtype=float)
        assert values.min() >= 0.0
        assert values.max() <= 1.0


def test_orgasm_estimate_is_opt_in_and_labelled(profiles, noise) -> None:
    off = simulate_outcomes(profiles, default_guide_protocol(), ModelConfig(), noise)
    assert ORGASM_COLUMN not in off.columns
    assert ORGASM_COLUMN not in outcome_columns(ModelConfig())

    orgasm_noise = make_noise(300, np.random.default_rng(2024), include_orgasm=True)
    on_config = ModelConfig(include_orgasm_estimate=True)
    on = simulate_outcomes(profiles, default_guide_protocol(), on_config, orgasm_noise)
    assert ORGASM_COLUMN in on.columns
    assert ORGASM_COLUMN in outcome_columns(on_config)
    assert "hypothetical" in ORGASM_COLUMN
    values = on[ORGASM_COLUMN].to_numpy(dtype=float)
    assert values.min() > 0.0 and values.max() < 1.0


def test_wrong_noise_length_is_rejected(profiles) -> None:
    short_noise = {channel: np.zeros(10) for channel in ["arousal", "pleasure", "comfort", "satisfaction", "discomfort"]}
    with pytest.raises(ValueError):
        simulate_outcomes(profiles, default_guide_protocol(), ModelConfig(), short_noise)


@pytest.mark.parametrize(
    "overrides",
    [
        {"consistency": 1.5},
        {"comfort_checks": -0.1},
        {"boundary_respect": "high"},
        {"adherence": None},
    ],
)
def test_invalid_protocol_values_are_rejected(overrides) -> None:
    with pytest.raises(ValueError):
        Protocol(**overrides)


def test_out_of_range_coefficients_are_rejected() -> None:
    with pytest.raises(ValueError):
        ModelConfig(w_pleasure_arousal=99.0)
    with pytest.raises(ValueError):
        ModelConfig(w_comfort_comfort_sensitivity=2.0)


def test_equations_text_documents_every_coefficient() -> None:
    config = ModelConfig(include_orgasm_estimate=True)
    text = equations_text(config)
    assert "sigma(x) = 1 / (1 + exp(-x))" in text
    for name in COEFFICIENT_SPECS:
        assert name in text, f"{name} is not named in the equation text"
        assert str(getattr(config, name)) in text, f"value of {name} is not shown"
