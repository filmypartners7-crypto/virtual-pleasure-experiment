"""Tests for the statistical layer on small, known datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vpe.model_config import ModelConfig, default_config
from vpe.population import generate_population
from vpe.protocol import default_guide_protocol
from vpe.statistics import (
    INTERVAL_LABEL,
    bootstrap_interval,
    condition_summary,
    paired_differences,
    paired_interval_table,
    sensitivity_analysis,
    subgroup_summary,
    summarise,
)


def _small_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.DataFrame(
        {
            "couple_id": ["C1", "C2", "C3"],
            "pleasure": [0.2, 0.4, 0.6],
            "comfort": [0.5, 0.5, 0.5],
        }
    )
    guide = pd.DataFrame(
        {
            "couple_id": ["C1", "C2", "C3"],
            "pleasure": [0.3, 0.4, 0.5],
            "comfort": [0.6, 0.5, 0.4],
        }
    )
    return baseline, guide


def test_summarise_matches_hand_computed_values() -> None:
    frame = pd.DataFrame({"pleasure": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = summarise(frame, ["pleasure"]).set_index("outcome").loc["pleasure"]
    assert stats["n"] == 5
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["median"] == pytest.approx(3.0)
    assert stats["sd"] == pytest.approx(float(np.std([1, 2, 3, 4, 5], ddof=1)))
    assert stats["q25"] == pytest.approx(2.0)
    assert stats["q75"] == pytest.approx(4.0)


def test_paired_differences_are_computed_per_couple() -> None:
    baseline, guide = _small_pair()
    paired = paired_differences(baseline, guide, ["pleasure", "comfort"])
    assert paired["diff_pleasure"].tolist() == pytest.approx([0.1, 0.0, -0.1])
    assert paired["diff_comfort"].tolist() == pytest.approx([0.1, 0.0, -0.1])
    assert paired["baseline_pleasure"].tolist() == pytest.approx([0.2, 0.4, 0.6])
    assert paired["guide_pleasure"].tolist() == pytest.approx([0.3, 0.4, 0.5])
    assert paired["diff_pleasure"].mean() == pytest.approx(0.0)


def test_paired_differences_reject_mismatched_couples() -> None:
    baseline, guide = _small_pair()
    shuffled = guide.iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(ValueError):
        paired_differences(baseline, shuffled, ["pleasure"])


def test_condition_summary_reports_the_mean_difference() -> None:
    baseline, guide = _small_pair()
    summary = condition_summary(baseline, guide, ["pleasure"]).set_index("outcome")
    assert summary.loc["pleasure", "baseline_mean"] == pytest.approx(0.4)
    assert summary.loc["pleasure", "guide_mean"] == pytest.approx(0.4)
    assert summary.loc["pleasure", "mean_paired_diff"] == pytest.approx(0.0)
    assert summary.loc["pleasure", "median_paired_diff"] == pytest.approx(0.0)


def test_bootstrap_interval_is_reproducible_ordered_and_centred() -> None:
    values = np.array([0.1] * 50 + [0.9] * 50)
    first = bootstrap_interval(values, n_boot=1000, seed=7)
    second = bootstrap_interval(values, n_boot=1000, seed=7)
    assert first == second
    assert first["mean"] == pytest.approx(0.5)
    assert first["low"] < first["mean"] < first["high"]
    assert first["low"] >= 0.0
    assert first["high"] <= 1.0


def test_bootstrap_interval_handles_constant_data() -> None:
    result = bootstrap_interval([0.25] * 30, n_boot=500, seed=1)
    assert result["low"] == pytest.approx(0.25)
    assert result["high"] == pytest.approx(0.25)


def test_bootstrap_interval_widens_with_more_confidence() -> None:
    rng = np.random.default_rng(3)
    values = rng.uniform(0, 1, size=300)
    narrow = bootstrap_interval(values, n_boot=800, alpha=0.10, seed=2)
    wide = bootstrap_interval(values, n_boot=800, alpha=0.01, seed=2)
    assert (wide["high"] - wide["low"]) > (narrow["high"] - narrow["low"])


@pytest.mark.parametrize(
    "kwargs",
    [{"n_boot": 10}, {"alpha": 0.0}, {"alpha": 1.5}],
)
def test_invalid_bootstrap_arguments_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        bootstrap_interval([0.1, 0.2, 0.3], **kwargs)


def test_bootstrap_of_empty_sample_is_rejected() -> None:
    with pytest.raises(ValueError):
        bootstrap_interval([])


def test_paired_interval_table_labels_intervals_correctly() -> None:
    baseline, guide = _small_pair()
    paired = paired_differences(baseline, guide, ["pleasure"])
    table = paired_interval_table(paired, ["pleasure"], n_boot=500, seed=4)
    row = table.set_index("outcome").loc["pleasure"]
    assert row["interval_type"] == INTERVAL_LABEL
    assert "conditional on the model" in row["interval_type"]
    assert "confidence interval" not in row["interval_type"].lower()
    assert row["interval_low"] <= row["mean_paired_diff"] <= row["interval_high"]
    assert row["excludes_zero"] in (True, False)


def test_sensitivity_analysis_covers_each_coefficient_twice() -> None:
    profiles = generate_population(120, seed=8)
    config = ModelConfig()
    frame = sensitivity_analysis(
        profiles,
        default_guide_protocol(),
        config,
        seed=1,
        outcomes=["pleasure"],
        relative_delta=0.2,
    )
    assert set(frame["direction"]) == {"lower", "higher"}
    assert len(frame) == 2 * len(config.numeric_items())
    assert (frame["perturbed_value"] != frame["base_value"]).all()
    assert "delta_pleasure" in frame.columns
    assert frame["delta_pleasure"].abs().max() > 0.0
    assert "not causal evidence" in frame.attrs["note"].lower()


def test_sensitivity_analysis_is_reproducible() -> None:
    profiles = generate_population(80, seed=2)
    first = sensitivity_analysis(profiles, default_guide_protocol(), ModelConfig(), seed=5, outcomes=["pleasure"])
    second = sensitivity_analysis(profiles, default_guide_protocol(), ModelConfig(), seed=5, outcomes=["pleasure"])
    pd.testing.assert_frame_equal(first, second)


def test_sensitivity_analysis_rejects_unknown_parameter() -> None:
    profiles = generate_population(50, seed=1)
    with pytest.raises(ValueError):
        sensitivity_analysis(
            profiles, default_guide_protocol(), ModelConfig(), seed=1, parameters=["not_a_parameter"]
        )


def test_subgroup_summary_partitions_the_population() -> None:
    profiles = generate_population(90, seed=4)
    baseline = pd.DataFrame(
        {"couple_id": profiles["couple_id"], "pleasure": np.full(90, 0.4)}
    )
    guide = pd.DataFrame(
        {"couple_id": profiles["couple_id"], "pleasure": np.full(90, 0.6)}
    )
    paired = paired_differences(baseline, guide, ["pleasure"])
    table = subgroup_summary(paired, profiles, by="female_comfort_sensitivity", outcome="pleasure")

    assert table["n"].sum() == 90
    assert len(table) == 3
    assert table["mean_paired_diff"].to_numpy() == pytest.approx([0.2, 0.2, 0.2])
    assert table["grouped_by"].iloc[0] == "female_comfort_sensitivity"
    assert set(table["subgroup"]) == {"low", "middle", "high"}


def test_subgroup_summary_supports_categorical_grouping() -> None:
    profiles = generate_population(60, seed=6)
    baseline = pd.DataFrame({"couple_id": profiles["couple_id"], "pleasure": np.full(60, 0.3)})
    guide = pd.DataFrame({"couple_id": profiles["couple_id"], "pleasure": np.full(60, 0.5)})
    paired = paired_differences(baseline, guide, ["pleasure"])
    table = subgroup_summary(paired, profiles, by="female_age_category", outcome="pleasure")
    assert table["n"].sum() == 60
    assert table["subgroup"].nunique() >= 2


def test_subgroup_summary_rejects_unknown_columns() -> None:
    profiles = generate_population(30, seed=1)
    baseline = pd.DataFrame({"couple_id": profiles["couple_id"], "pleasure": np.full(30, 0.3)})
    guide = pd.DataFrame({"couple_id": profiles["couple_id"], "pleasure": np.full(30, 0.5)})
    paired = paired_differences(baseline, guide, ["pleasure"])
    with pytest.raises(ValueError):
        subgroup_summary(paired, profiles, by="not_a_column", outcome="pleasure")
    with pytest.raises(ValueError):
        subgroup_summary(paired, profiles, by="female_age_category", outcome="not_an_outcome")


def test_default_config_is_used_when_none_given() -> None:
    assert default_config().w_pleasure_arousal == ModelConfig().w_pleasure_arousal
