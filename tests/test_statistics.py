"""Statistics on small known datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vpe.statistics import (
    INTERVAL_LABEL,
    bootstrap_interval,
    condition_summary,
    paired_differences,
    paired_interval_table,
    subgroup_summary,
    summarise,
)


def _pair():
    a = pd.DataFrame({"woman_id": ["W1", "W2", "W3"], "pleasure": [0.2, 0.4, 0.6]})
    b = pd.DataFrame({"woman_id": ["W1", "W2", "W3"], "pleasure": [0.3, 0.4, 0.5]})
    return a, b


def test_summarise_matches_hand_computation() -> None:
    stats = summarise(pd.DataFrame({"pleasure": [1.0, 2.0, 3.0, 4.0, 5.0]}), ["pleasure"])
    row = stats.set_index("outcome").loc["pleasure"]
    assert row["mean"] == pytest.approx(3.0)
    assert row["median"] == pytest.approx(3.0)
    assert row["sd"] == pytest.approx(float(np.std([1, 2, 3, 4, 5], ddof=1)))
    assert row["q25"] == pytest.approx(2.0)


def test_paired_differences_are_b_minus_a() -> None:
    a, b = _pair()
    paired = paired_differences(a, b, ["pleasure"])
    assert paired["diff_pleasure"].tolist() == pytest.approx([0.1, 0.0, -0.1])
    assert paired["diff_pleasure"].mean() == pytest.approx(0.0)


def test_paired_differences_reject_different_women() -> None:
    a, b = _pair()
    with pytest.raises(ValueError):
        paired_differences(a, b.iloc[[1, 0, 2]].reset_index(drop=True), ["pleasure"])


def test_condition_summary_share_improved() -> None:
    a, b = _pair()
    summary = condition_summary(a, b, "A", "B", ["pleasure"]).set_index("outcome")
    assert summary.loc["pleasure", "mean_paired_diff"] == pytest.approx(0.0)
    assert summary.loc["pleasure", "share_improved_percent"] == pytest.approx(100 / 3)


def test_bootstrap_is_reproducible_and_ordered() -> None:
    values = np.array([0.1] * 50 + [0.9] * 50)
    first = bootstrap_interval(values, n_boot=800, seed=5)
    second = bootstrap_interval(values, n_boot=800, seed=5)
    assert first == second
    assert first["mean"] == pytest.approx(0.5)
    assert first["low"] < first["mean"] < first["high"]


def test_bootstrap_constant_data() -> None:
    result = bootstrap_interval([0.25] * 40, n_boot=300, seed=1)
    assert result["low"] == pytest.approx(0.25)
    assert result["high"] == pytest.approx(0.25)


@pytest.mark.parametrize("kwargs", [{"n_boot": 10}, {"alpha": 0.0}, {"alpha": 2.0}])
def test_bootstrap_rejects_bad_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        bootstrap_interval([0.1, 0.2], **kwargs)


def test_bootstrap_rejects_empty() -> None:
    with pytest.raises(ValueError):
        bootstrap_interval([])


def test_paired_interval_table_label() -> None:
    a, b = _pair()
    table = paired_interval_table(paired_differences(a, b, ["pleasure"]), ["pleasure"], n_boot=400, seed=2)
    row = table.set_index("outcome").loc["pleasure"]
    assert row["interval_type"] == INTERVAL_LABEL
    assert "confidence" not in row["interval_type"].lower()


def test_subgroup_summary_partitions_the_population() -> None:
    profiles = pd.DataFrame({"archetype": ["x"] * 4 + ["y"] * 6})
    paired = pd.DataFrame(
        {
            "diff_pleasure": [0.1] * 4 + [0.2] * 6,
            "a_pleasure": [0.5] * 10,
            "b_pleasure": [0.6] * 4 + [0.7] * 6,
        }
    )
    table = subgroup_summary(paired, profiles, by="archetype", outcome="pleasure")
    assert table["n"].sum() == 10
    assert table.set_index("subgroup").loc["y", "mean_diff"] == pytest.approx(0.2)


def test_subgroup_summary_rejects_unknown_columns() -> None:
    profiles = pd.DataFrame({"archetype": ["x"] * 3})
    paired = pd.DataFrame({"diff_pleasure": [0.1] * 3, "a_pleasure": [0.5] * 3, "b_pleasure": [0.6] * 3})
    with pytest.raises(ValueError):
        subgroup_summary(paired, profiles, by="nope", outcome="pleasure")
    with pytest.raises(ValueError):
        subgroup_summary(paired, profiles, by="archetype", outcome="nope")
