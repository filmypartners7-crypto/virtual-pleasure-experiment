"""Protocols: an ordered list of steps.

A protocol is now a *sequence*, not seven sliders. That is what makes ordering,
ablation and substitution experiments possible: you can disable a step, move it,
or swap a technique, and re-run against the same women.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import pandas as pd

from .techniques import TECHNIQUE_INDEX, get as get_technique

MAX_STEP_MINUTES = 15.0


@dataclass(frozen=True)
class Step:
    """One step of a protocol: a technique, for a duration, at an intensity."""

    technique_key: str
    duration_min: float = 2.0
    intensity: float | None = None      # None -> the technique's default intensity
    note: str = ""

    def __post_init__(self) -> None:
        if self.technique_key not in TECHNIQUE_INDEX:
            raise ValueError(f"unknown technique: {self.technique_key}")
        if not isinstance(self.duration_min, (int, float)) or isinstance(self.duration_min, bool):
            raise ValueError("step duration must be a number of minutes")
        if not (1.0 / 60.0 <= float(self.duration_min) <= MAX_STEP_MINUTES):
            raise ValueError(
                f"step duration must lie between 1 second and {MAX_STEP_MINUTES} minutes"
            )
        if self.intensity is not None:
            if not isinstance(self.intensity, (int, float)) or isinstance(self.intensity, bool):
                raise ValueError("step intensity must be a number or None")
            if not (0.0 <= float(self.intensity) <= 1.0):
                raise ValueError("step intensity must lie in [0, 1]")

    @property
    def technique(self):
        return get_technique(self.technique_key)

    def resolved_intensity(self) -> float:
        if self.technique.verbal:
            return 0.0
        return float(self.technique.intensity if self.intensity is None else self.intensity)

    def as_row(self, index: int) -> dict[str, Any]:
        return {
            "step": index,
            "technique_key": self.technique_key,
            "technique": self.technique.label,
            "minutes": round(float(self.duration_min), 2),
            "intensity": round(self.resolved_intensity(), 2),
            "note": self.note,
        }


@dataclass(frozen=True)
class Protocol:
    name: str
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("protocol name must be a non-empty string")
        if not self.steps:
            raise ValueError("a protocol must contain at least one step")
        if not all(isinstance(step, Step) for step in self.steps):
            raise ValueError("protocol steps must be Step instances")

    # -- properties ------------------------------------------------------ #

    @property
    def duration_min(self) -> float:
        return float(sum(step.duration_min for step in self.steps))

    @property
    def verbal_minutes(self) -> float:
        return float(sum(step.duration_min for step in self.steps if step.technique.verbal))

    @property
    def check_in_fraction(self) -> float:
        return self.verbal_minutes / self.duration_min if self.duration_min else 0.0

    @property
    def max_directness(self) -> float:
        return max((step.technique.directness for step in self.steps), default=0.0)

    def signature(self) -> str:
        return ">".join(step.technique_key for step in self.steps)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([step.as_row(i) for i, step in enumerate(self.steps, start=1)])

    # -- transformations used by the experiment engine ------------------- #

    def with_step_disabled(self, index: int, name_suffix: str = " (step removed)") -> "Protocol":
        """Return a copy with the step at ``index`` (0-based) removed."""
        if not (0 <= index < len(self.steps)):
            raise IndexError("step index out of range")
        if len(self.steps) == 1:
            raise ValueError("cannot remove the only step of a protocol")
        remaining = tuple(step for i, step in enumerate(self.steps) if i != index)
        return replace(self, name=f"{self.name}{name_suffix}", steps=remaining)

    def reordered(self, order: Sequence[int], name_suffix: str = " (reordered)") -> "Protocol":
        """Return a copy whose steps follow ``order`` (a permutation of indices)."""
        if sorted(order) != list(range(len(self.steps))):
            raise ValueError("order must be a permutation of the step indices")
        return replace(
            self,
            name=f"{self.name}{name_suffix}",
            steps=tuple(self.steps[i] for i in order),
        )

    def with_step_replaced(
        self, index: int, technique_key: str, name_suffix: str = " (step substituted)"
    ) -> "Protocol":
        if not (0 <= index < len(self.steps)):
            raise IndexError("step index out of range")
        if technique_key not in TECHNIQUE_INDEX:
            raise ValueError(f"unknown technique: {technique_key}")
        steps = list(self.steps)
        old = steps[index]
        steps[index] = Step(
            technique_key=technique_key,
            duration_min=old.duration_min,
            intensity=None,
            note=f"substituted for {old.technique_key}",
        )
        return replace(self, name=f"{self.name}{name_suffix}", steps=tuple(steps))

    def with_duration_scaled(self, factor: float, name_suffix: str = "") -> "Protocol":
        if not (0.1 <= factor <= 10.0):
            raise ValueError("duration factor must lie between 0.1 and 10")
        return replace(
            self,
            name=f"{self.name}{name_suffix}",
            steps=tuple(
                Step(
                    technique_key=step.technique_key,
                    duration_min=max(1.0 / 60.0, min(MAX_STEP_MINUTES, step.duration_min * factor)),
                    intensity=step.intensity,
                    note=step.note,
                )
                for step in self.steps
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_min": round(self.duration_min, 2),
            "signature": self.signature(),
            "steps": [
                {
                    "technique_key": step.technique_key,
                    "technique": step.technique.label,
                    "minutes": round(float(step.duration_min), 2),
                    "intensity": round(step.resolved_intensity(), 2),
                    "note": step.note,
                }
                for step in self.steps
            ],
        }

    # -- constructors ---------------------------------------------------- #

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]], name: str) -> "Protocol":
        """Build a protocol from the app's editable step table."""
        steps = []
        for row in rows:
            technique_key = row.get("technique_key") or row.get("technique")
            if technique_key is None:
                raise ValueError("each row needs a technique_key")
            duration = row.get("minutes", row.get("duration_min", 2.0))
            intensity = row.get("intensity")
            if intensity is not None and pd.isna(intensity):
                intensity = None
            steps.append(
                Step(
                    technique_key=str(technique_key),
                    duration_min=float(duration),
                    intensity=None if intensity is None else float(intensity),
                )
            )
        return cls(name=name, steps=tuple(steps))


# --------------------------------------------------------------------------- #
# The baseline condition
# --------------------------------------------------------------------------- #

BASELINE_DEFINITION = (
    "BASELINE CONDITION (modelling convention, not a control group). A short, "
    "unstructured interaction: brief non-genital contact, a little labial work, then "
    "straight to direct glans stimulation with no check-ins and no pacing discipline. "
    "It is a deliberate contrast to a structured guide and it is visible in full in the "
    "app. Because it is a convention rather than a measurement, it sets much of the "
    "size of any difference you observe: a harsher baseline makes any guide look better."
)

BASELINE_PROTOCOL = Protocol(
    name="baseline",
    steps=(
        Step("thighs_kissing", 1.0, note="brief priming"),
        Step("outer_labia", 2.0),
        Step("inner_labia", 1.5),
        Step("clitoral_glans", 4.0, note="straight to direct contact, no hood transition"),
    ),
)

REFERENCE_PROTOCOL = BASELINE_PROTOCOL


def default_structured_protocol() -> Protocol:
    """A paced, consent-forward starting point for the user's condition."""
    return Protocol(
        name="structured guide",
        steps=(
            Step("thighs_kissing", 2.0),
            Step("outer_labia", 3.0),
            Step("inner_labia", 3.0),
            Step("perineum", 1.5),
            Step("check_in", 0.5),
            Step("clitoral_hood", 4.0),
            Step("flat_tongue", 4.0),
            Step("check_in", 0.5),
            Step("clitoral_glans", 5.0),
            Step("suction", 2.0),
            Step("check_in", 0.5),
        ),
    )
