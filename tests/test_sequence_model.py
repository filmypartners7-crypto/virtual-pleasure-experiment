"""The sequence model: bounded, deterministic, and ordering must actually matter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vpe.population import generate_population
from vpe.protocol import BASELINE_PROTOCOL, Protocol, Step, default_structured_protocol
from vpe.sequence_model import (
    DEFAULT_PARAMETERS,
    OUTCOME_COLUMNS,
    ModelParameters,
    build_schedule,
    simulate_many,
    simulate_protocol,
)


@pytest.fixture(scope="module")
def profiles():
    return generate_population(400, seed=21)


def test_outcomes_are_bounded(profiles) -> None:
    outcomes, _ = simulate_protocol(profiles, default_structured_protocol())
    for column in OUTCOME_COLUMNS:
        values = outcomes[column].to_numpy(dtype=float)
        assert np.isfinite(values).all()
        assert values.min() > 0.0
        assert values.max() < 1.0


def test_simulation_is_deterministic(profiles) -> None:
    a, _ = simulate_protocol(profiles, default_structured_protocol())
    b, _ = simulate_protocol(profiles, default_structured_protocol())
    pd.testing.assert_frame_equal(a, b)


def test_warmup_before_direct_contact_beats_direct_first(profiles) -> None:
    """The core ordering claim, tested directly."""
    paced = Protocol(
        name="paced",
        steps=(Step("outer_labia", 4.0), Step("inner_labia", 3.0), Step("clitoral_hood", 4.0), Step("clitoral_glans", 5.0)),
    )
    direct_first = Protocol(
        name="direct first",
        steps=(Step("clitoral_glans", 5.0), Step("outer_labia", 4.0), Step("inner_labia", 3.0), Step("clitoral_hood", 4.0)),
    )
    paced_out, _ = simulate_protocol(profiles, paced)
    direct_out, _ = simulate_protocol(profiles, direct_first)
    assert paced_out["pleasure"].mean() > direct_out["pleasure"].mean()
    assert paced_out["discomfort"].mean() < direct_out["discomfort"].mean()


def test_reversed_order_is_worse_than_as_written(profiles) -> None:
    protocol = default_structured_protocol()
    forward, _ = simulate_protocol(profiles, protocol)
    reversed_protocol = protocol.reordered(list(reversed(range(len(protocol.steps)))))
    backward, _ = simulate_protocol(profiles, reversed_protocol)
    assert forward["pleasure"].mean() > backward["pleasure"].mean()


def test_removing_warmup_increases_discomfort(profiles) -> None:
    full = default_structured_protocol()
    keep = [i for i, s in enumerate(full.steps) if s.technique.directness >= 0.5]
    stripped = Protocol(name="no warm-up", steps=tuple(full.steps[i] for i in keep))
    full_out, _ = simulate_protocol(profiles, full)
    strip_out, _ = simulate_protocol(profiles, stripped)
    assert strip_out["discomfort"].mean() > full_out["discomfort"].mean()
    assert strip_out["pleasure"].mean() < full_out["pleasure"].mean()


def test_communication_raises_comfort(profiles) -> None:
    without = Protocol(name="no checks", steps=(Step("outer_labia", 4.0), Step("clitoral_glans", 5.0)))
    with_checks = Protocol(
        name="with checks",
        steps=(
            Step("outer_labia", 4.0), Step("check_in", 0.5),
            Step("clitoral_glans", 5.0), Step("check_in", 0.5),
        ),
    )
    a, _ = simulate_protocol(profiles, without)
    b, _ = simulate_protocol(profiles, with_checks)
    assert b["comfort"].mean() > a["comfort"].mean()


def test_overstimulation_punishes_excessive_intensity(profiles) -> None:
    gentle = Protocol(name="gentle", steps=(Step("clitoral_hood", 6.0, intensity=0.30),))
    harsh = Protocol(name="harsh", steps=(Step("clitoral_glans", 6.0, intensity=1.00),))
    g, _ = simulate_protocol(profiles, gentle)
    h, _ = simulate_protocol(profiles, harsh)
    assert h["discomfort"].mean() > g["discomfort"].mean()


def test_individuals_respond_differently(profiles) -> None:
    outcomes, _ = simulate_protocol(profiles, default_structured_protocol())
    assert outcomes["pleasure"].std() > 0.03
    merged = profiles.merge(outcomes, on="woman_id")
    spread = merged.groupby("archetype")["pleasure"].mean()
    assert spread.max() - spread.min() > 0.05


def test_duration_affects_outcome(profiles) -> None:
    short = Protocol(name="short", steps=(Step("clitoral_hood", 2.0),))
    long = Protocol(name="long", steps=(Step("clitoral_hood", 8.0),))
    s, _ = simulate_protocol(profiles, short)
    l, _ = simulate_protocol(profiles, long)
    assert l["pleasure"].mean() > s["pleasure"].mean()


def test_trajectory_is_returned_on_request(profiles) -> None:
    _, trajectory = simulate_protocol(profiles, default_structured_protocol(), keep_trajectory=True)
    assert trajectory is not None
    assert len(trajectory) > 5
    assert trajectory["mean_arousal"].between(0.0, 1.0).all()


def test_schedule_matches_the_protocol_length() -> None:
    protocol = Protocol(name="p", steps=(Step("outer_labia", 2.0), Step("clitoral_glans", 3.0)))
    indices, intensities, verbal = build_schedule(protocol, dt=1.0)
    assert len(indices) == 5
    assert len(intensities) == 5
    assert not verbal.any()


def test_out_of_range_parameters_are_rejected(profiles) -> None:
    with pytest.raises(ValueError):
        simulate_protocol(profiles, default_structured_protocol(), ModelParameters(arousal_gain=-1.0))


def test_simulate_many_stacks_conditions(profiles) -> None:
    frames = simulate_many(
        profiles,
        {"baseline": BASELINE_PROTOCOL, "paced": default_structured_protocol()},
    )
    assert set(frames["condition"]) == {"baseline", "paced"}
    assert len(frames) == 2 * len(profiles)


def test_empty_population_is_rejected() -> None:
    with pytest.raises(ValueError):
        simulate_protocol(generate_population(2, seed=1).iloc[0:0], BASELINE_PROTOCOL)
