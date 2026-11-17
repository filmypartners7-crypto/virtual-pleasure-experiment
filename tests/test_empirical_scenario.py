from __future__ import annotations

import numpy as np
import pytest

from vpe.empirical_scenario import run_empirical_satisfaction_scenario
from vpe.population import generate_population
from vpe.protocol import BASELINE_PROTOCOL, PROTOCOL_VARIABLES, Protocol, default_guide_protocol


def test_zero_effect_reproduces_reference_categories_exactly() -> None:
    profiles = generate_population(1000, seed=123)
    frame, meta = run_empirical_satisfaction_scenario(
        profiles, default_guide_protocol(), BASELINE_PROTOCOL, assumed_effect_logit=0.0
    )
    assert (frame["baseline_category"] == frame["guide_scenario_category"]).all()
    assert meta["moved_up_count"] == 0
    assert meta["moved_down_count"] == 0
    assert meta["unchanged_count"] == meta["practiced_count"]


def test_positive_assumption_never_moves_a_profile_down() -> None:
    profiles = generate_population(2000, seed=321)
    strong_guide = Protocol(**{key: 1.0 for key in PROTOCOL_VARIABLES})
    frame, meta = run_empirical_satisfaction_scenario(
        profiles, strong_guide, BASELINE_PROTOCOL, assumed_effect_logit=2.5
    )
    changes = frame.loc[frame["practiced_in_reference"], "ordinal_score_change"]
    assert (changes >= 0).all()
    assert meta["moved_down_count"] == 0


def test_negative_assumption_never_moves_a_profile_up() -> None:
    profiles = generate_population(2000, seed=654)
    weak_guide = Protocol(**{key: 0.0 for key in PROTOCOL_VARIABLES})
    frame, meta = run_empirical_satisfaction_scenario(
        profiles, weak_guide, BASELINE_PROTOCOL, assumed_effect_logit=2.5
    )
    changes = frame.loc[frame["practiced_in_reference"], "ordinal_score_change"]
    assert (changes <= 0).all()
    assert meta["moved_up_count"] == 0


def test_not_practiced_category_is_held_unchanged() -> None:
    profiles = generate_population(1500, seed=789)
    strong_guide = Protocol(**{key: 1.0 for key in PROTOCOL_VARIABLES})
    frame, _ = run_empirical_satisfaction_scenario(
        profiles, strong_guide, BASELINE_PROTOCOL, assumed_effect_logit=3.0
    )
    not_practiced = frame["baseline_category"] == "not_practiced"
    assert (frame.loc[not_practiced, "guide_scenario_category"] == "not_practiced").all()


@pytest.mark.parametrize("effect", [-3.01, 3.01, float("nan"), float("inf")])
def test_invalid_assumed_effect_is_rejected(effect: float) -> None:
    profiles = generate_population(100, seed=9)
    with pytest.raises(ValueError):
        run_empirical_satisfaction_scenario(
            profiles, default_guide_protocol(), BASELINE_PROTOCOL, assumed_effect_logit=effect
        )
