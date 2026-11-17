from __future__ import annotations

import numpy as np
import pytest

from vpe.research import (
    MODEL_REFERENCE_ID,
    RESEARCH_STUDIES,
    SSCI_RECEIVING_ORAL_SEX,
    SSCI_SATISFACTION_FACTORS,
    reference_distribution_table,
    research_metadata,
    sample_reference_profiles,
    satisfaction_factors_table,
)


def test_published_reference_distribution_sums_to_100() -> None:
    assert sum(row["percent"] for row in SSCI_RECEIVING_ORAL_SEX) == pytest.approx(100.0)
    table = reference_distribution_table()
    assert table["percent"].sum() == pytest.approx(100.0)
    assert "Not practiced" in table["category"].tolist()


def test_synthetic_resampling_approximates_published_aggregate_margins() -> None:
    frame = sample_reference_profiles(100_000, np.random.default_rng(813))
    observed = frame["empirical_reference_category"].value_counts(normalize=True) * 100
    for row in SSCI_RECEIVING_ORAL_SEX:
        assert observed[row["key"]] == pytest.approx(row["percent"], abs=0.6)


def test_reference_resampling_is_reproducible() -> None:
    first = sample_reference_profiles(500, np.random.default_rng(14))
    second = sample_reference_profiles(500, np.random.default_rng(14))
    assert first.equals(second)


def test_research_metadata_contains_provenance_and_caveats() -> None:
    meta = research_metadata()
    assert meta["reference_distribution_source_id"] == MODEL_REFERENCE_ID
    assert meta["reference_distribution_source"]["year"] == 2022
    assert "non-probability" in meta["reference_distribution_source"]["sample"]
    assert "No raw" in meta["data_handling"]
    assert len(RESEARCH_STUDIES) >= 4
    assert len(satisfaction_factors_table()) >= 5


# --------------------------------------------------------------------------- #
# Citation lock
# --------------------------------------------------------------------------- #
# These values are transcribed from a published paper. If anyone edits the
# embedded numbers, these tests fail - that is the point. Transcribed and
# verified against the published tables on 2026-11-17.
#
# Source: Ogallar-Blanco, Lara-Moreno & Godoy-Izquierdo (2022), IJERPH 19(16),
# 10232, doi:10.3390/ijerph191610232. Open access: PMC9408228.

PUBLISHED_TABLE_4_RECEIVING_ORAL_SEX = {
    "not_practiced": 5.3,
    "not_satisfactory": 3.1,
    "somewhat_satisfactory": 10.5,
    "quite_satisfactory": 21.9,
    "very_satisfactory": 59.2,
}

PUBLISHED_TABLE_3_AFFECTS_A_LOT = {
    "Partner attraction": 72.5,
    "Not feeling obliged to do unwanted activity": 72.1,
    "Confidence with sexual partner": 66.8,
    "Feeling comfortable with own body": 63.3,
    "Feeling relaxed and comfortable": 51.5,
    "Partner's actions / techniques": 53.7,
    "Communication about likes, dislikes and boundaries": 39.3,
}


def test_table_4_transcription_is_exact() -> None:
    embedded = {row["key"]: row["percent"] for row in SSCI_RECEIVING_ORAL_SEX}
    assert embedded == PUBLISHED_TABLE_4_RECEIVING_ORAL_SEX


def test_table_4_reproduces_the_published_mean() -> None:
    """The paper reports 3.27 +/- 1.11 on a 0-4 scale, scoring 'not practiced' as 0."""
    weights = [row["percent"] / 100.0 for row in SSCI_RECEIVING_ORAL_SEX]
    scores = [0.0 if row["score"] is None else float(row["score"]) for row in SSCI_RECEIVING_ORAL_SEX]
    weighted_mean = sum(w * s for w, s in zip(weights, scores))
    assert weighted_mean == pytest.approx(3.27, abs=0.005)


def test_table_3_transcription_is_exact() -> None:
    embedded = {row["factor"]: row["percent_affects_a_lot"] for row in SSCI_SATISFACTION_FACTORS}
    assert embedded == PUBLISHED_TABLE_3_AFFECTS_A_LOT


def test_citation_metadata_is_complete() -> None:
    source = RESEARCH_STUDIES[0]
    assert source["doi"] == "10.3390/ijerph191610232"
    assert source["journal"].startswith("International Journal of Environmental Research and Public Health")
    assert "1,080" in source["sample"]
    assert "18-50" in source["sample"]
    for study in RESEARCH_STUDIES:
        for field in ("id", "title", "authors", "year", "journal", "doi", "sample", "limitations", "use_in_vpe"):
            assert study.get(field), f"{study['id']} is missing {field}"

