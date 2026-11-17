"""Tests for the paired experiment engine, reproducibility and exports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vpe.disclaimer import SYNTHETIC_DISCLAIMER
from vpe.experiment import build_settings, rerun_with_seed, run_experiment
from vpe.export import (
    read_exported_csv,
    read_exported_json,
    results_to_csv_bytes,
    settings_to_json_bytes,
)
from vpe.model_config import MODEL_VERSION, ModelConfig, default_config
from vpe.population import PROFILE_COLUMNS
from vpe.protocol import BASELINE_PROTOCOL, Protocol, default_guide_protocol


@pytest.fixture(scope="module")
def result():
    return run_experiment(
        population_size=400,
        seed=2026,
        protocol=default_guide_protocol(),
        config=default_config(),
    )


def test_baseline_and_guide_use_identical_paired_profiles(result) -> None:
    assert result.baseline["couple_id"].tolist() == result.guide["couple_id"].tolist()
    assert result.baseline["couple_id"].tolist() == result.profiles["couple_id"].tolist()
    assert len(result.paired) == len(result.profiles) == 400
    # the paired table must be a one-to-one join of the two conditions
    assert result.paired["couple_id"].is_unique


def test_both_conditions_share_the_same_profiles_by_construction() -> None:
    """Re-running with a different protocol must not change the population."""
    first = run_experiment(200, seed=77, protocol=default_guide_protocol())
    second = run_experiment(
        200,
        seed=77,
        protocol=Protocol(
            consistency=0.9,
            feedback_responsiveness=0.9,
            adaptation=0.9,
            communication=0.9,
            comfort_checks=0.9,
            boundary_respect=0.9,
            adherence=0.9,
        ),
    )
    pd.testing.assert_frame_equal(first.profiles, second.profiles)
    assert not np.allclose(
        first.guide["pleasure"].to_numpy(), second.guide["pleasure"].to_numpy()
    )


def test_same_seed_reproduces_identical_results() -> None:
    first = run_experiment(250, seed=11, protocol=default_guide_protocol())
    second = run_experiment(250, seed=11, protocol=default_guide_protocol())
    pd.testing.assert_frame_equal(first.profiles, second.profiles)
    pd.testing.assert_frame_equal(first.baseline, second.baseline)
    pd.testing.assert_frame_equal(first.guide, second.guide)
    assert first.settings["run_fingerprint"] == second.settings["run_fingerprint"]


def test_different_seeds_produce_different_populations_and_results() -> None:
    first = run_experiment(250, seed=11, protocol=default_guide_protocol())
    second = run_experiment(250, seed=12, protocol=default_guide_protocol())
    assert not np.allclose(
        first.profiles["female_psychological_context"].to_numpy(),
        second.profiles["female_psychological_context"].to_numpy(),
    )
    assert first.settings["run_fingerprint"] != second.settings["run_fingerprint"]


def test_outcomes_stay_within_bounds(result) -> None:
    for frame in (result.baseline, result.guide):
        for column in result.outcomes:
            values = frame[column].to_numpy(dtype=float)
            assert values.min() > 0.0
            assert values.max() < 1.0


def test_summary_reports_both_conditions_and_the_paired_difference(result) -> None:
    assert set(result.summary["outcome"]) == set(result.outcomes)
    row = result.summary.set_index("outcome").loc["pleasure"]
    assert row["mean_paired_diff"] == pytest.approx(row["guide_mean"] - row["baseline_mean"])


def test_settings_record_the_full_configuration(result) -> None:
    settings = result.settings
    assert settings["random_seed"] == 2026
    assert settings["population_size"] == 400
    assert settings["model_version"] == MODEL_VERSION
    assert settings["protocol"]["consistency"] == pytest.approx(default_guide_protocol().consistency)
    assert settings["baseline_protocol"]["label"] == "baseline"
    assert settings["baseline_definition"]
    assert "model_config" in settings
    assert len(settings["run_fingerprint"]) == 16
    assert SYNTHETIC_DISCLAIMER in settings["disclaimer"]
    assert "not a control group" in settings["baseline_definition"].lower()


def test_baseline_is_visible_and_fixed() -> None:
    assert BASELINE_PROTOCOL.label == "baseline"
    assert BASELINE_PROTOCOL.values()  # every variable has a value
    settings = build_settings(10, 1, default_guide_protocol(), BASELINE_PROTOCOL, ModelConfig())
    assert settings["baseline_protocol"]["boundary_respect"] == pytest.approx(
        BASELINE_PROTOCOL.boundary_respect
    )


def test_guide_reference_text_is_stored_but_not_used() -> None:
    with_text = run_experiment(
        100,
        seed=5,
        protocol=Protocol(**default_guide_protocol().values(), guide_reference_text="some guide text"),
    )
    without_text = run_experiment(100, seed=5, protocol=default_guide_protocol())
    assert with_text.settings["protocol"]["guide_reference_text"] == "some guide text"
    assert "NOT parsed" in with_text.settings["protocol"]["guide_reference_note"]
    # identical numbers: the free text cannot influence the model
    pd.testing.assert_frame_equal(with_text.guide, without_text.guide)


def test_csv_export_contains_disclaimer_and_configuration(result) -> None:
    payload = results_to_csv_bytes(result)
    text = payload.decode("utf-8")
    assert text.startswith("#")
    assert "SYNTHETIC DATA" in text
    assert f"seed={result.seed}" in text
    assert f"model_version={MODEL_VERSION}" in text

    frame = read_exported_csv(payload)
    assert len(frame) == result.population_size
    for column in ["couple_id", "baseline_pleasure", "guide_pleasure", "diff_pleasure"]:
        assert column in frame.columns
    for column in PROFILE_COLUMNS:
        assert column in frame.columns


def test_json_export_contains_settings_model_version_and_seed(result) -> None:
    payload = settings_to_json_bytes(result)
    settings = read_exported_json(payload)
    assert settings["random_seed"] == result.seed
    assert settings["model_version"] == MODEL_VERSION
    assert settings["population_size"] == result.population_size
    assert settings["protocol"]["adaptation"] == pytest.approx(
        result.settings["protocol"]["adaptation"]
    )
    assert "SYNTHETIC DATA" in settings["disclaimer"]
    assert "equations" in settings
    assert "confidence interval" in settings["interval_labelling"]


def test_orgasm_estimate_is_exported_only_when_enabled() -> None:
    config = default_config(include_orgasm_estimate=True)
    enabled = run_experiment(120, seed=9, protocol=default_guide_protocol(), config=config)
    assert "orgasm_estimate_hypothetical" in enabled.outcomes
    assert "diff_orgasm_estimate_hypothetical" in enabled.paired.columns

    default_run = run_experiment(120, seed=9, protocol=default_guide_protocol())
    assert "orgasm_estimate_hypothetical" not in default_run.outcomes


def test_rerun_with_new_seed_keeps_the_protocol() -> None:
    first = run_experiment(150, seed=3, protocol=default_guide_protocol())
    second = rerun_with_seed(first, 4)
    assert second.seed == 4
    assert second.settings["protocol"]["adaptation"] == pytest.approx(
        first.settings["protocol"]["adaptation"]
    )
    assert second.population_size == first.population_size


@pytest.mark.parametrize("bad_seed", [-1, 2**32, 1.5, "42", None, True])
def test_invalid_seeds_are_rejected(bad_seed) -> None:
    with pytest.raises(ValueError):
        run_experiment(50, seed=bad_seed, protocol=default_guide_protocol())


def test_missing_or_invalid_protocol_is_rejected() -> None:
    with pytest.raises(ValueError):
        run_experiment(50, seed=1, protocol=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        run_experiment(50, seed=1, protocol={"consistency": 0.5})  # type: ignore[arg-type]


def test_empirical_reference_and_guide_scenario_are_included_in_run_and_exports(result) -> None:
    assert "empirical_reference_category" in result.profiles.columns
    assert "empirical_reference_latent_utility" in result.profiles.columns
    assert len(result.empirical_scenario) == result.population_size
    assert result.settings["assumed_effect_logit"] == 0.0
    assert result.settings["empirical_reference"]["reference_distribution_source_id"] == "ssci_2022_receiving_oral_sex"
    assert result.empirical_scenario_metadata["moved_up_count"] == 0
    assert result.empirical_scenario_metadata["moved_down_count"] == 0

    frame = read_exported_csv(results_to_csv_bytes(result))
    for column in [
        "empirical_reference_category",
        "baseline_category",
        "guide_scenario_category",
        "ordinal_score_change",
    ]:
        assert column in frame.columns

    settings = read_exported_json(settings_to_json_bytes(result))
    assert settings["empirical_reference"]["reference_distribution_source"]["year"] == 2022
    assert settings["empirical_scenario_metadata"]["assumed_effect_logit_per_full_score_unit"] == 0.0


def test_assumed_empirical_effect_is_recorded_and_changes_fingerprint() -> None:
    protocol = default_guide_protocol()
    null_effect = run_experiment(250, seed=808, protocol=protocol, assumed_effect_logit=0.0)
    positive_scenario = run_experiment(250, seed=808, protocol=protocol, assumed_effect_logit=2.0)
    assert positive_scenario.settings["assumed_effect_logit"] == 2.0
    assert positive_scenario.fingerprint() != null_effect.fingerprint()
    assert positive_scenario.empirical_scenario_metadata["applied_latent_logit_shift"] > 0
