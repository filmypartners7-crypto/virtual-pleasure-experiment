"""Experiment engine: paired design, ablations, ordering, calibration, exports."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vpe.calibration import CALIBRATION_REPORT, calibration_check
from vpe.disclaimer import SYNTHETIC_DISCLAIMER
from vpe.evidence import MODEL_VERSION, SSCI_PRACTISED_BANDS
from vpe.experiment import run_experiment
from vpe.export import (
    read_exported_csv,
    read_exported_json,
    results_to_csv_bytes,
    settings_to_json_bytes,
)
from vpe.population import generate_population
from vpe.protocol import BASELINE_PROTOCOL, Protocol, Step, default_structured_protocol
from vpe.sequence_model import DEFAULT_PARAMETERS


@pytest.fixture(scope="module")
def result():
    return run_experiment(
        population_size=300,
        seed=2026,
        protocol_a=BASELINE_PROTOCOL,
        protocol_b=default_structured_protocol(),
        label_a="Baseline",
        label_b="Guide",
    )


def test_both_conditions_use_the_same_women(result) -> None:
    assert result.condition_a["woman_id"].tolist() == result.condition_b["woman_id"].tolist()
    assert result.condition_a["woman_id"].tolist() == result.profiles["woman_id"].tolist()
    assert result.paired["woman_id"].is_unique
    assert len(result.paired) == 300


def test_same_seed_reproduces_everything() -> None:
    a = run_experiment(200, seed=11, protocol_b=default_structured_protocol())
    b = run_experiment(200, seed=11, protocol_b=default_structured_protocol())
    pd.testing.assert_frame_equal(a.condition_a, b.condition_a)
    pd.testing.assert_frame_equal(a.condition_b, b.condition_b)
    assert a.fingerprint() == b.fingerprint()


def test_different_seed_changes_population_and_fingerprint() -> None:
    a = run_experiment(200, seed=11, protocol_b=default_structured_protocol())
    b = run_experiment(200, seed=12, protocol_b=default_structured_protocol())
    assert not np.allclose(
        a.profiles["preferred_intensity"].to_numpy(), b.profiles["preferred_intensity"].to_numpy()
    )
    assert a.fingerprint() != b.fingerprint()


def test_paired_difference_equals_mean_difference(result) -> None:
    row = result.summary.set_index("outcome").loc["pleasure"]
    assert row["mean_paired_diff"] == pytest.approx(row["Guide_mean"] - row["Baseline_mean"], abs=1e-9)


def test_paced_guide_beats_the_unstructured_baseline(result) -> None:
    summary = result.summary.set_index("outcome")
    assert summary.loc["pleasure", "mean_paired_diff"] > 0
    assert summary.loc["comfort", "mean_paired_diff"] > 0
    assert summary.loc["discomfort", "mean_paired_diff"] < 0


def test_intervals_are_labelled_and_ordered(result) -> None:
    for row in result.intervals.to_dict("records"):
        assert row["interval_type"].startswith("bootstrap percentile interval")
        assert "confidence interval" not in row["interval_type"].lower()
        assert row["interval_low"] <= row["mean_paired_diff"] <= row["interval_high"]


def test_ablation_covers_every_step_of_protocol_b(result) -> None:
    protocol = default_structured_protocol()
    assert len(result.ablations) == len(protocol.steps)
    assert set(result.ablations["removed_key"]) == {step.technique_key for step in protocol.steps}


def test_removing_the_warmup_step_hurts_pleasure(result) -> None:
    warmup_rows = result.ablations[
        result.ablations["removed_key"].isin(["outer_labia", "inner_labia"])
    ]
    assert (warmup_rows["delta_pleasure"] < 0).all()


def test_sequence_variants_include_the_key_comparisons(result) -> None:
    labels = set(result.sequence_variants["variant"])
    assert "as written" in labels
    assert "reversed order" in labels
    assert "direct step moved to the front" in labels
    assert "communication removed" in labels


def test_reversed_order_scores_worse_than_as_written(result) -> None:
    variants = result.sequence_variants.set_index("variant")
    assert variants.loc["reversed order", "pleasure"] < variants.loc["as written", "pleasure"]


def test_subgroups_cover_all_archetypes(result) -> None:
    assert result.subgroups["n"].sum() == 300
    assert result.subgroups["subgroup"].nunique() >= 4
    assert "mean_diff" in result.subgroups.columns


def test_sensitivity_covers_parameters_and_is_signed(result) -> None:
    assert not result.sensitivity.empty
    assert "delta_pleasure" in result.sensitivity.columns
    assert set(result.sensitivity["direction"]) == {"lower", "higher"}


def test_calibration_table_is_reported_for_the_run(result) -> None:
    table = result.calibration
    assert list(table["band"]) == ["not satisfactory", "somewhat", "quite", "very"]
    assert table["published_percent"].sum() == pytest.approx(sum(SSCI_PRACTISED_BANDS), abs=0.1)
    # the deviation is always reported, whatever the population size
    assert table["deviation_points"].notna().all()
    assert table["model_percent"].sum() == pytest.approx(100.0, abs=0.5)


def test_calibration_holds_at_the_calibration_configuration() -> None:
    check = calibration_check()
    assert check["within_tolerance"] is True
    assert check["max_deviation_points"] <= 3.0


def test_settings_record_the_full_configuration(result) -> None:
    settings = result.settings
    assert settings["random_seed"] == 2026
    assert settings["population_size"] == 300
    assert settings["model_version"] == MODEL_VERSION
    assert settings["protocol_a"]["name"] == "baseline"
    assert settings["protocol_b"]["steps"]
    assert settings["calibration"]["target_source"] == "ogallar_blanco_2022"
    assert len(settings["studies"]) == 6
    assert len(settings["run_fingerprint"]) == 16
    assert SYNTHETIC_DISCLAIMER in settings["disclaimer"]
    assert "unvalidated" in settings["evidence_position"].lower()


def test_csv_export_has_disclaimer_and_both_conditions(result) -> None:
    payload = results_to_csv_bytes(result)
    text = payload.decode()
    assert text.startswith("#")
    assert "SYNTHETIC SIMULATION" in text
    frame = read_exported_csv(payload)
    assert len(frame) == 300
    for column in ("woman_id", "a_pleasure", "b_pleasure", "diff_pleasure", "archetype"):
        assert column in frame.columns


def test_json_export_contains_studies_and_calibration(result) -> None:
    payload = settings_to_json_bytes(result)
    record = read_exported_json(payload)
    assert record["model_version"] == MODEL_VERSION
    assert record["random_seed"] == result.seed
    assert len(record["studies"]) == 6
    assert record["calibration"]["max_deviation_points"] <= 3.0
    assert record["results_summary"]
    assert record["ablations"]
    assert "SYNTHETIC" in record["disclaimer"]


def test_invalid_seeds_rejected() -> None:
    for bad in (-1, 2**32, 1.5, "42", None, True):
        with pytest.raises(ValueError):
            run_experiment(50, seed=bad, protocol_b=default_structured_protocol())


def test_non_protocol_rejected() -> None:
    with pytest.raises(ValueError):
        run_experiment(50, seed=1, protocol_a={"steps": []})  # type: ignore[arg-type]


def test_single_step_protocols_run() -> None:
    single = Protocol(name="single", steps=(Step("clitoral_hood", 5.0),))
    outcome = run_experiment(100, seed=3, protocol_a=single, protocol_b=single, run_ablations=True)
    assert outcome.ablations.empty          # nothing to ablate
    assert outcome.population_size == 100


def test_calibration_check_function() -> None:
    check = calibration_check()
    assert check["source"].id == "ogallar_blanco_2022"
    assert CALIBRATION_REPORT["method"].startswith("coarse grid search")
    assert CALIBRATION_REPORT["max_deviation_across_seeds"]
    assert CALIBRATION_REPORT["tolerance_points"] == 3.0
