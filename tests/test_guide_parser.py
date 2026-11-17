"""The parser must be predictable, and must not invent steps from negations."""
from __future__ import annotations

import pytest

from vpe.guide_parser import (
    extract_duration_minutes,
    extract_intensity_modifier,
    find_techniques,
    parse_guide,
    split_blocks,
)

GUIDE = """1. Approach - whole vulva first, no clitoral contact. Start with broad flat-tongue strokes over the outer labia for several minutes.
2. Inner labia and vaginal opening. Trace the inner labia, then the perineum.
3. Indirect clitoral contact. Work over the clitoral hood, not the glans.
4. Direct glans contact once aroused. Flat tongue first, then pointed tongue circles.
5. Check in every few minutes: more, less, or the same?
6. Gentle suction only as escalation, never as the opener.
7. A pause or stop is not a negotiation."""


def test_parses_the_expected_sequence_in_order() -> None:
    result = parse_guide(GUIDE)
    keys = [step.technique_key for step in result.steps]
    assert keys[:4] == ["outer_labia", "inner_labia", "vaginal_opening", "perineum"]
    assert "clitoral_hood" in keys
    assert keys.index("clitoral_hood") < keys.index("clitoral_glans")
    assert "check_in" in keys and "pause" in keys


def test_negated_mentions_are_not_turned_into_steps() -> None:
    """'not the glans' must not create a glans step; 'no clitoral contact' must not either."""
    result = parse_guide("Work over the clitoral hood, not the glans.")
    keys = [step.technique_key for step in result.steps]
    assert "clitoral_glans" not in keys
    assert "clitoral_hood" in keys

    result = parse_guide("Start with the outer labia. No direct clitoral contact yet.")
    assert "clitoral_glans" not in [s.technique_key for s in result.steps]


def test_no_duplicate_technique_within_one_block() -> None:
    result = parse_guide("Trace the inner labia, then the perineum, then the inner labia again.")
    keys = [step.technique_key for step in result.steps]
    assert keys.count("inner_labia") == 1


def test_explicit_duration_is_used() -> None:
    minutes, explicit = extract_duration_minutes("keep going for 3 minutes")
    assert explicit is True
    assert minutes == pytest.approx(3.0)


def test_seconds_are_converted() -> None:
    minutes, explicit = extract_duration_minutes("about 30 seconds here")
    assert explicit is True
    assert minutes == pytest.approx(0.5)


def test_word_glans_is_not_read_as_seconds() -> None:
    """Regression: 'glans' once matched 'an' + 's' and produced a 1-second step."""
    minutes, explicit = extract_duration_minutes("direct glans contact")
    assert explicit is False
    assert minutes is None


def test_as_is_not_read_as_seconds() -> None:
    minutes, explicit = extract_duration_minutes("gentle suction only as escalation")
    assert explicit is False


def test_missing_duration_falls_back_to_the_default() -> None:
    result = parse_guide("Work the outer labia slowly.", default_minutes=2.5)
    assert result.steps[0].duration_min == pytest.approx(2.5)
    assert any("default" in warning for warning in result.warnings)


def test_intensity_modifiers_nudge_the_value() -> None:
    gentle = parse_guide("Use gentle light strokes on the clitoral hood.")
    firm = parse_guide("Use firm pressure on the clitoral hood.")
    assert gentle.steps[0].intensity < firm.steps[0].intensity


def test_unrecognised_text_is_reported_not_invented() -> None:
    result = parse_guide("Hold what works. Keep the rhythm steady for a minute.")
    assert result.steps == []
    assert result.unclassified_blocks


def test_empty_text_is_handled() -> None:
    result = parse_guide("   ")
    assert result.steps == []
    assert result.warnings


def test_all_durations_are_valid_for_a_protocol() -> None:
    result = parse_guide(GUIDE)
    for step in result.steps:
        assert 0.1 <= step.duration_min <= 15.0
    assert result.total_minutes <= 45.0


def test_block_splitting_handles_numbered_and_blank_line_input() -> None:
    numbered = split_blocks("1. one\n2. two\n3. three")
    assert len(numbered) == 3
    blank = split_blocks("first para\n\nsecond para")
    assert len(blank) == 2


def test_ambiguous_labia_is_flagged() -> None:
    result = parse_guide("Use flat strokes on the labia for two minutes.")
    assert result.steps[0].technique_key in {"outer_labia", "inner_labia"}
    assert any("ambiguous" in warning.lower() for warning in result.warnings)


def test_find_techniques_returns_ordered_mentions() -> None:
    mentions = find_techniques("outer labia, then clitoral hood, then glans")
    assert [m[1] for m in mentions] == ["outer_labia", "clitoral_hood", "clitoral_glans"]
