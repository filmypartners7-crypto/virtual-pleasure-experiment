"""Protocol objects: validation and the transformations the engine relies on."""
from __future__ import annotations

import pytest

from vpe.protocol import BASELINE_PROTOCOL, Protocol, Step, default_structured_protocol


def test_step_validation() -> None:
    with pytest.raises(ValueError):
        Step("not_a_technique", 2.0)
    with pytest.raises(ValueError):
        Step("outer_labia", 0.0)
    with pytest.raises(ValueError):
        Step("outer_labia", 99.0)
    with pytest.raises(ValueError):
        Step("outer_labia", 2.0, intensity=1.5)


def test_protocol_requires_steps_and_a_name() -> None:
    with pytest.raises(ValueError):
        Protocol(name="empty", steps=())
    with pytest.raises(ValueError):
        Protocol(name="  ", steps=(Step("outer_labia", 1.0),))


def test_duration_and_check_in_fraction() -> None:
    protocol = Protocol(
        name="p",
        steps=(Step("outer_labia", 3.0), Step("check_in", 1.0), Step("clitoral_glans", 6.0)),
    )
    assert protocol.duration_min == pytest.approx(10.0)
    assert protocol.check_in_fraction == pytest.approx(0.1)


def test_disable_step() -> None:
    protocol = default_structured_protocol()
    reduced = protocol.with_step_disabled(0)
    assert len(reduced.steps) == len(protocol.steps) - 1
    assert reduced.steps[0] == protocol.steps[1]


def test_cannot_remove_the_only_step() -> None:
    single = Protocol(name="one", steps=(Step("outer_labia", 1.0),))
    with pytest.raises(ValueError):
        single.with_step_disabled(0)


def test_reorder_requires_a_permutation() -> None:
    protocol = default_structured_protocol()
    with pytest.raises(ValueError):
        protocol.reordered([0, 0, 1])
    reversed_protocol = protocol.reordered(list(reversed(range(len(protocol.steps)))))
    assert reversed_protocol.steps[0] == protocol.steps[-1]


def test_replace_step_keeps_duration() -> None:
    protocol = Protocol(name="p", steps=(Step("clitoral_glans", 4.0), Step("outer_labia", 2.0)))
    swapped = protocol.with_step_replaced(0, "clitoral_hood")
    assert swapped.steps[0].technique_key == "clitoral_hood"
    assert swapped.steps[0].duration_min == pytest.approx(4.0)


def test_from_rows_round_trip() -> None:
    rows = [
        {"technique_key": "outer_labia", "minutes": 3.0, "intensity": 0.3},
        {"technique_key": "check_in", "minutes": 0.5, "intensity": None},
    ]
    protocol = Protocol.from_rows(rows, name="built")
    assert [s.technique_key for s in protocol.steps] == ["outer_labia", "check_in"]
    assert protocol.steps[1].intensity is None


def test_from_rows_rejects_unknown_technique() -> None:
    with pytest.raises(ValueError):
        Protocol.from_rows([{"technique_key": "nonsense", "minutes": 2.0}], name="bad")


def test_duration_scaling() -> None:
    protocol = Protocol(name="p", steps=(Step("outer_labia", 2.0),))
    doubled = protocol.with_duration_scaled(2.0)
    assert doubled.duration_min == pytest.approx(4.0)
    with pytest.raises(ValueError):
        protocol.with_duration_scaled(50.0)


def test_baseline_is_documented_and_fixed() -> None:
    assert BASELINE_PROTOCOL.name == "baseline"
    assert len(BASELINE_PROTOCOL.steps) >= 3
    assert BASELINE_PROTOCOL.check_in_fraction == 0.0


def test_to_dict_records_the_sequence() -> None:
    payload = default_structured_protocol().to_dict()
    assert payload["steps"]
    assert payload["signature"] == default_structured_protocol().signature()
    assert payload["duration_min"] == pytest.approx(default_structured_protocol().duration_min, abs=0.01)
