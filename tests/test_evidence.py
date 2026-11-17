"""The evidence database must stay honest: real citations, declared gaps."""

from __future__ import annotations

import pytest

from vpe.evidence import (
    CALIBRATION_RESULT,
    EVIDENCE_LEVELS,
    PARAMETER_EVIDENCE,
    SSCI_PRACTISED_BANDS,
    SSCI_SATISFACTION_CATEGORIES,
    STUDIES,
    STUDY_INDEX,
    evidence_coverage,
    parameter_evidence_frame,
    studies_frame,
    studies_for,
    unverified_citations,
)


def test_six_sources_are_registered() -> None:
    assert len(STUDIES) == 6
    assert len(STUDY_INDEX) == 6


def test_every_citation_is_verified_and_complete() -> None:
    assert list(unverified_citations()) == []
    for study in STUDIES:
        for field in ("id", "authors", "title", "journal", "doi", "url", "sample", "limitations"):
            assert getattr(study, field), f"{study.id} is missing {field}"
        assert study.doi.startswith("10."), study.id
        assert study.findings and study.supports and study.does_not_support


def test_known_dois_are_transcribed_exactly() -> None:
    assert STUDY_INDEX["ogallar_blanco_2022"].doi == "10.3390/ijerph191610232"
    assert STUDY_INDEX["richters_2006"].doi == "10.1080/00224490609552320"
    assert STUDY_INDEX["herbenick_2018"].doi == "10.1080/0092623X.2017.1346530"
    assert STUDY_INDEX["frederick_2018"].doi == "10.1007/s10508-017-0939-z"
    assert STUDY_INDEX["backstrom_2012"].doi == "10.1080/00224499.2011.585523"
    assert STUDY_INDEX["rosen_2000_fsfi"].doi == "10.1080/009262300278597"


def test_sample_sizes_match_the_published_records() -> None:
    assert "1,080" in STUDY_INDEX["ogallar_blanco_2022"].sample
    assert "19,307" in STUDY_INDEX["richters_2006"].sample
    assert "1,055" in STUDY_INDEX["herbenick_2018"].sample
    assert "52,588" in STUDY_INDEX["frederick_2018"].sample
    assert "43" in STUDY_INDEX["backstrom_2012"].sample


def test_no_source_claims_to_estimate_a_technique_effect() -> None:
    """The single most important honesty invariant in the project."""
    for study in STUDIES:
        items = study.does_not_support
        if isinstance(items, str):
            items = (items,)
        joined = " ".join(items).lower()
        assert "technique" in joined or "effect" in joined or "coefficient" in joined, study.id


def test_technique_effects_are_declared_unvalidated() -> None:
    assert PARAMETER_EVIDENCE["technique_effects"]["evidence"] == "unvalidated_relationship"
    assert PARAMETER_EVIDENCE["sequence_and_ordering"]["evidence"] == "unvalidated_relationship"


def test_every_parameter_has_an_evidence_level() -> None:
    for name, spec in PARAMETER_EVIDENCE.items():
        assert spec["evidence"] in EVIDENCE_LEVELS, name
        if spec["study"]:
            assert spec["study"] in STUDY_INDEX, name


def test_published_marginals_sum_to_one_hundred() -> None:
    assert sum(p for _, _, p in SSCI_SATISFACTION_CATEGORIES) == pytest.approx(100.0)


def test_calibration_target_is_renormalised_correctly() -> None:
    # conditional on practising: 94.7% of the sample
    assert sum(SSCI_PRACTISED_BANDS) == pytest.approx(100.0, abs=0.05)
    assert SSCI_PRACTISED_BANDS[0] == pytest.approx(3.1 / 94.7 * 100, abs=0.01)


def test_calibration_metadata_is_populated() -> None:
    assert CALIBRATION_RESULT["target_source"] == "ogallar_blanco_2022"
    assert CALIBRATION_RESULT["target_table"].startswith("Table 4")


def test_frames_render() -> None:
    studies = studies_frame()
    assert len(studies) == 6
    assert "does_not_support" in studies.columns
    params = parameter_evidence_frame()
    assert {"parameter", "evidence_status", "source"} <= set(params.columns)
    coverage = evidence_coverage()
    assert coverage["parameters"].sum() == len(PARAMETER_EVIDENCE)


def test_studies_for_returns_the_cited_source() -> None:
    assert studies_for("preference_heterogeneity")[0].id == "herbenick_2018"
    assert studies_for("satisfaction_calibration_target")[0].id == "ogallar_blanco_2022"
    assert studies_for("technique_effects") == ()
