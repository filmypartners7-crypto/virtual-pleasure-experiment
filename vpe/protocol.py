"""Protocol representation for the Virtual Pleasure Experiment.

A protocol is a small set of structured variables the experimenter sets with
sliders. The optional free-text guide is stored for reference only: it is never
parsed, summarised or interpreted by the application. The experimenter is the
one who decides which slider value corresponds to which part of their guide.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

PROTOCOL_VARIABLES: list[str] = [
    "consistency",
    "feedback_responsiveness",
    "adaptation",
    "communication",
    "comfort_checks",
    "boundary_respect",
    "adherence",
]

PROTOCOL_VARIABLE_INFO: dict[str, dict[str, str]] = {
    "consistency": {
        "label": "Consistency of the approach",
        "help": (
            "How reliably the approach is applied across the encounter. Enters as "
            "effective consistency, which is discounted by the profile's adherence variability."
        ),
    },
    "feedback_responsiveness": {
        "label": "Responsiveness to partner feedback",
        "help": (
            "How far the approach changes in response to cues from the partner. Averaged with "
            "the male profile's feedback responsiveness to form the feedback loop."
        ),
    },
    "adaptation": {
        "label": "Adaptation to individual preferences",
        "help": (
            "How far the approach is tailored to this person. Drives the preference-match term, "
            "which is stronger for profiles with specific preferences."
        ),
    },
    "communication": {
        "label": "Communication",
        "help": (
            "How much explicit verbal communication the approach builds in. Combined with the "
            "feedback loop to form effective communication."
        ),
    },
    "comfort_checks": {
        "label": "Comfort checks",
        "help": (
            "How often the approach pauses to check comfort. Raises comfort and lowers "
            "discomfort risk."
        ),
    },
    "boundary_respect": {
        "label": "Respect for pauses and boundaries",
        "help": (
            "How strictly the approach honours pauses, refusals and limits. The strongest "
            "single term in the comfort equation."
        ),
    },
    "adherence": {
        "label": "Protocol adherence",
        "help": (
            "How closely the declared approach is actually followed. Discounted by the male "
            "profile's adherence variability before it enters any equation."
        ),
    },
}


def _check_unit_interval(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"protocol variable '{name}' must be a number between 0 and 1")
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"protocol variable '{name}' must lie in [0, 1], got {value}")
    return value


@dataclass(frozen=True)
class Protocol:
    """A structured experimental protocol, all variables on a 0-1 scale."""

    consistency: float = 0.50
    feedback_responsiveness: float = 0.50
    adaptation: float = 0.50
    communication: float = 0.50
    comfort_checks: float = 0.50
    boundary_respect: float = 0.50
    adherence: float = 0.50
    label: str = "user_configured"
    guide_reference_text: str = ""

    def __post_init__(self) -> None:
        for name in PROTOCOL_VARIABLES:
            object.__setattr__(self, name, _check_unit_interval(name, getattr(self, name)))
        if not isinstance(self.label, str):
            raise ValueError("protocol label must be a string")
        if not isinstance(self.guide_reference_text, str):
            raise ValueError("guide reference text must be a string")

    def values(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in PROTOCOL_VARIABLES}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["guide_reference_text"] = self.guide_reference_text
        data["guide_reference_note"] = (
            "Reference text supplied by the experimenter. It is NOT parsed or interpreted by "
            "the application; the experimenter maps the guide to the structured variables."
        )
        return data

    def as_frame(self) -> pd.DataFrame:
        rows = []
        for name in PROTOCOL_VARIABLES:
            info = PROTOCOL_VARIABLE_INFO[name]
            rows.append(
                {
                    "variable": name,
                    "label": info["label"],
                    "value": round(float(getattr(self, name)), 3),
                    "how it enters the model": info["help"],
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The baseline condition
# --------------------------------------------------------------------------- #

BASELINE_DEFINITION = (
    "BASELINE CONDITION (modelling convention, not a control group). A fixed, "
    "unstructured 'ordinary interaction' reference: no deliberate structure, moderate "
    "communication, occasional comfort checks and no systematic adaptation to the "
    "individual. The exact values are visible in the protocol table and are identical "
    "for every couple. Because the baseline is a convention rather than a measurement, "
    "it largely determines the size of any baseline-versus-guide difference: choosing "
    "lower baseline values makes almost any structured protocol look better."
)

BASELINE_PROTOCOL = Protocol(
    consistency=0.40,
    feedback_responsiveness=0.35,
    adaptation=0.30,
    communication=0.40,
    comfort_checks=0.30,
    boundary_respect=0.55,
    adherence=0.40,
    label="baseline",
    guide_reference_text="",
)


def default_guide_protocol() -> Protocol:
    """A neutral starting point for the user-configured condition."""
    return Protocol(
        consistency=0.65,
        feedback_responsiveness=0.70,
        adaptation=0.70,
        communication=0.70,
        comfort_checks=0.65,
        boundary_respect=0.85,
        adherence=0.65,
        label="user_configured",
    )
