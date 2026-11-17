"""Central configuration for the Virtual Pleasure Experiment (VPE).

Every number that can influence a simulated outcome lives in this module so that
the whole model can be inspected, printed, exported and changed in one place.

Two rules govern this file:

1.  **No invented citations.**  Each parameter carries an ``evidence`` label from
    :data:`EVIDENCE_LEVELS`.  In version 0.2.0 every continuous response-model coefficient is either a
    ``hypothetical_assumption`` or an ``unvalidated_relationship``.  A label of
    "research-supported concept" describes the *qualitative idea* only (for
    example: comfort and consent practices plausibly matter) - never the number.
2.  **No hidden constants.**  Structural constants (how strongly adherence
    variability erodes a protocol, for example) are also declared here with a
    documented range instead of being buried inside a formula.

Nothing here is fitted to empirical data.  Changing a coefficient changes the
simulation, not the world.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import pandas as pd

MODEL_VERSION = "0.2.0"

# --------------------------------------------------------------------------- #
# Evidence taxonomy (section 11 of the specification)
# --------------------------------------------------------------------------- #

EVIDENCE_LEVELS: dict[str, dict[str, str]] = {
    "research_supported_concept": {
        "code": "1",
        "label": "Research-supported concept",
        "meaning": (
            "The qualitative idea that this variable matters is discussed in the "
            "sexological / relationship literature. The numeric value used here is "
            "still a hypothetical placeholder and is not taken from any study."
        ),
    },
    "research_informed_range": {
        "code": "2",
        "label": "Research-informed range",
        "meaning": (
            "The plausible range of the parameter is loosely informed by published "
            "work, but the default value is chosen for demonstration and has not "
            "been fitted to data."
        ),
    },
    "hypothetical_assumption": {
        "code": "3",
        "label": "Hypothetical assumption",
        "meaning": (
            "Chosen so the demonstration behaves sensibly. No empirical source "
            "exists for this number."
        ),
    },
    "unvalidated_relationship": {
        "code": "4",
        "label": "Unvalidated model relationship",
        "meaning": (
            "A structural link between two variables invented for this model. It "
            "has never been tested and may simply be wrong."
        ),
    },
}


# --------------------------------------------------------------------------- #
# Parameter specifications
# --------------------------------------------------------------------------- #
# Each entry: label, allowed range (low, high), description, evidence level.

COEFFICIENT_SPECS: dict[str, dict[str, Any]] = {
    # ---------------- subjective arousal (latent scale) ----------------
    "arousal_intercept": {
        "label": "Arousal intercept",
        "low": -3.0,
        "high": 3.0,
        "description": "Baseline log-odds of subjective arousal when all inputs are at their reference value.",
        "evidence": "hypothetical_assumption",
    },
    "w_arousal_consistency": {
        "label": "Arousal <- consistency of approach",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of effective consistency of the approach on the arousal logit.",
        "evidence": "hypothetical_assumption",
    },
    "w_arousal_adaptation": {
        "label": "Arousal <- preference match",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of how well the protocol matches the individual's preferences.",
        "evidence": "hypothetical_assumption",
    },
    "w_arousal_communication": {
        "label": "Arousal <- communication loop",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the couple's effective communication / feedback loop.",
        "evidence": "hypothetical_assumption",
    },
    "w_arousal_bio_variability": {
        "label": "Arousal <- biological response variability",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the synthetic female profile's biological response variability (centred).",
        "evidence": "hypothetical_assumption",
    },
    "w_arousal_psych_context": {
        "label": "Arousal <- psychological context",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the synthetic female profile's psychological / contextual state (centred).",
        "evidence": "hypothetical_assumption",
    },
    "w_arousal_comfort": {
        "label": "Arousal <- safety practices",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of effective boundary respect and comfort checks on arousal.",
        "evidence": "research_supported_concept",
    },
    "arousal_noise_sd": {
        "label": "Arousal unexplained variation (sd)",
        "low": 0.0,
        "high": 2.0,
        "description": "Standard deviation of random variation added to the arousal logit.",
        "evidence": "hypothetical_assumption",
    },
    # ---------------- subjective pleasure ----------------
    "pleasure_intercept": {
        "label": "Pleasure intercept",
        "low": -3.0,
        "high": 3.0,
        "description": "Baseline log-odds of the pleasure estimate when all inputs are at reference.",
        "evidence": "hypothetical_assumption",
    },
    "w_pleasure_arousal": {
        "label": "Pleasure <- subjective arousal",
        "low": 0.0,
        "high": 3.0,
        "description": (
            "Weight of the *subjective arousal estimate* on pleasure. Deliberately "
            "moderate: the model does not treat arousal as pleasure."
        ),
        "evidence": "research_supported_concept",
    },
    "w_pleasure_comfort": {
        "label": "Pleasure <- comfort",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the comfort estimate on pleasure.",
        "evidence": "research_supported_concept",
    },
    "w_pleasure_preference_match": {
        "label": "Pleasure <- preference match",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of protocol-individual preference match on pleasure.",
        "evidence": "research_supported_concept",
    },
    "w_pleasure_communication": {
        "label": "Pleasure <- communication loop",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the communication loop on pleasure.",
        "evidence": "hypothetical_assumption",
    },
    "w_pleasure_consistency": {
        "label": "Pleasure <- consistency",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of effective consistency on pleasure.",
        "evidence": "hypothetical_assumption",
    },
    "w_pleasure_psych_context": {
        "label": "Pleasure <- psychological context",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the psychological / contextual profile variable on pleasure (centred).",
        "evidence": "hypothetical_assumption",
    },
    "w_pleasure_biological": {
        "label": "Pleasure <- biological variability",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the biological response variability profile variable on pleasure (centred).",
        "evidence": "hypothetical_assumption",
    },
    "pleasure_noise_sd": {
        "label": "Pleasure unexplained variation (sd)",
        "low": 0.0,
        "high": 2.0,
        "description": "Standard deviation of random variation added to the pleasure logit.",
        "evidence": "hypothetical_assumption",
    },
    # ---------------- comfort ----------------
    "comfort_intercept": {
        "label": "Comfort intercept",
        "low": -3.0,
        "high": 3.0,
        "description": "Baseline log-odds of the comfort estimate when all inputs are at reference.",
        "evidence": "hypothetical_assumption",
    },
    "w_comfort_boundary_respect": {
        "label": "Comfort <- boundary respect",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of effective respect for pauses and boundaries on comfort.",
        "evidence": "research_supported_concept",
    },
    "w_comfort_comfort_checks": {
        "label": "Comfort <- comfort checks",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of explicit comfort checks on comfort.",
        "evidence": "research_supported_concept",
    },
    "w_comfort_communication": {
        "label": "Comfort <- communication loop",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the communication loop on comfort.",
        "evidence": "research_supported_concept",
    },
    "w_comfort_psych_context": {
        "label": "Comfort <- psychological context",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the psychological / contextual profile variable on comfort (centred).",
        "evidence": "hypothetical_assumption",
    },
    "w_comfort_comfort_sensitivity": {
        "label": "Comfort <- comfort sensitivity",
        "low": -3.0,
        "high": 0.0,
        "description": (
            "Negative weight: a profile with higher comfort sensitivity reports lower "
            "comfort at the same protocol quality."
        ),
        "evidence": "hypothetical_assumption",
    },
    "w_comfort_adherence": {
        "label": "Comfort <- adherence",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of effective protocol adherence on comfort.",
        "evidence": "hypothetical_assumption",
    },
    "comfort_noise_sd": {
        "label": "Comfort unexplained variation (sd)",
        "low": 0.0,
        "high": 2.0,
        "description": "Standard deviation of random variation added to the comfort logit.",
        "evidence": "hypothetical_assumption",
    },
    # ---------------- satisfaction ----------------
    "satisfaction_intercept": {
        "label": "Satisfaction intercept",
        "low": -3.0,
        "high": 3.0,
        "description": "Baseline log-odds of the satisfaction estimate when all inputs are at reference.",
        "evidence": "hypothetical_assumption",
    },
    "w_satisfaction_pleasure": {
        "label": "Satisfaction <- pleasure",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the pleasure estimate on satisfaction.",
        "evidence": "research_supported_concept",
    },
    "w_satisfaction_comfort": {
        "label": "Satisfaction <- comfort",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the comfort estimate on satisfaction.",
        "evidence": "research_supported_concept",
    },
    "w_satisfaction_communication": {
        "label": "Satisfaction <- communication loop",
        "low": 0.0,
        "high": 3.0,
        "description": "Weight of the communication loop on satisfaction.",
        "evidence": "hypothetical_assumption",
    },
    "w_satisfaction_expectation_gap": {
        "label": "Satisfaction <- unmet specificity",
        "low": -3.0,
        "high": 0.0,
        "description": (
            "Negative weight applied to unmet preference specificity "
            "(specificity x (1 - match)): a strongly specific profile whose "
            "preferences are not met is less satisfied."
        ),
        "evidence": "hypothetical_assumption",
    },
    "satisfaction_noise_sd": {
        "label": "Satisfaction unexplained variation (sd)",
        "low": 0.0,
        "high": 2.0,
        "description": "Standard deviation of random variation added to the satisfaction logit.",
        "evidence": "hypothetical_assumption",
    },
    # ---------------- discomfort (separate mechanism) ----------------
    "discomfort_intercept": {
        "label": "Discomfort intercept",
        "low": -5.0,
        "high": 2.0,
        "description": "Baseline log-odds of the discomfort risk indicator.",
        "evidence": "hypothetical_assumption",
    },
    "w_discomfort_boundary_lapse": {
        "label": "Discomfort <- boundary lapse",
        "low": 0.0,
        "high": 5.0,
        "description": "Weight of (1 - effective boundary respect) on discomfort risk.",
        "evidence": "research_supported_concept",
    },
    "w_discomfort_comfort_check_gap": {
        "label": "Discomfort <- absent comfort checks",
        "low": 0.0,
        "high": 5.0,
        "description": "Weight of (1 - effective comfort checks) on discomfort risk.",
        "evidence": "research_supported_concept",
    },
    "w_discomfort_comfort_sensitivity": {
        "label": "Discomfort <- comfort sensitivity",
        "low": 0.0,
        "high": 5.0,
        "description": "Weight of the profile's comfort sensitivity (centred) on discomfort risk.",
        "evidence": "hypothetical_assumption",
    },
    "w_discomfort_communication_gap": {
        "label": "Discomfort <- communication gap",
        "low": 0.0,
        "high": 5.0,
        "description": "Weight of (1 - communication loop) on discomfort risk.",
        "evidence": "hypothetical_assumption",
    },
    "w_discomfort_adherence_instability": {
        "label": "Discomfort <- instability",
        "low": 0.0,
        "high": 5.0,
        "description": "Weight of (1 - effective consistency) on discomfort risk.",
        "evidence": "hypothetical_assumption",
    },
    "discomfort_noise_sd": {
        "label": "Discomfort unexplained variation (sd)",
        "low": 0.0,
        "high": 2.0,
        "description": "Standard deviation of random variation added to the discomfort logit.",
        "evidence": "hypothetical_assumption",
    },
    # ---------------- optional orgasm-related estimate ----------------
    "orgasm_intercept": {
        "label": "Orgasm estimate intercept",
        "low": -5.0,
        "high": 2.0,
        "description": (
            "Baseline log-odds of the optional, clearly hypothetical orgasm-related "
            "estimate. Disabled by default."
        ),
        "evidence": "unvalidated_relationship",
    },
    "w_orgasm_arousal": {
        "label": "Orgasm estimate <- arousal",
        "low": 0.0,
        "high": 5.0,
        "description": "Weight of the arousal estimate on the optional orgasm-related estimate.",
        "evidence": "unvalidated_relationship",
    },
    "w_orgasm_pleasure": {
        "label": "Orgasm estimate <- pleasure",
        "low": 0.0,
        "high": 5.0,
        "description": "Weight of the pleasure estimate on the optional orgasm-related estimate.",
        "evidence": "unvalidated_relationship",
    },
    "w_orgasm_comfort": {
        "label": "Orgasm estimate <- comfort",
        "low": 0.0,
        "high": 5.0,
        "description": "Weight of the comfort estimate on the optional orgasm-related estimate.",
        "evidence": "unvalidated_relationship",
    },
    "orgasm_noise_sd": {
        "label": "Orgasm estimate unexplained variation (sd)",
        "low": 0.0,
        "high": 2.0,
        "description": "Standard deviation of random variation in the optional orgasm-related estimate.",
        "evidence": "unvalidated_relationship",
    },
    # ---------------- structural constants ----------------
    "adherence_erosion_weight": {
        "label": "Adherence erosion",
        "low": 0.0,
        "high": 1.0,
        "description": (
            "How strongly a male profile's adherence variability erodes the declared "
            "protocol adherence: adherence_eff = adherence - w * variability * (1 - adherence)."
        ),
        "evidence": "unvalidated_relationship",
    },
    "consistency_adherence_floor": {
        "label": "Consistency floor",
        "low": 0.0,
        "high": 1.0,
        "description": "Floor term in consistency_eff = consistency * (floor + (1-floor) * adherence_eff).",
        "evidence": "unvalidated_relationship",
    },
    "feedback_protocol_weight": {
        "label": "Feedback loop averaging weight",
        "low": 0.0,
        "high": 1.0,
        "description": "Weight combining protocol responsiveness and the male profile's responsiveness into one loop.",
        "evidence": "unvalidated_relationship",
    },
    "communication_protocol_weight": {
        "label": "Communication averaging weight",
        "low": 0.0,
        "high": 1.0,
        "description": "Weight combining the protocol communication level and the feedback loop into effective communication.",
        "evidence": "unvalidated_relationship",
    },
    "preference_match_floor": {
        "label": "Preference-match floor",
        "low": 0.0,
        "high": 1.0,
        "description": "Floor term in the preference-match equation (adaptation x (floor + (1-floor) x specificity)).",
        "evidence": "unvalidated_relationship",
    },
    "boundary_erosion_weight": {
        "label": "Boundary erosion",
        "low": 0.0,
        "high": 1.0,
        "description": "How strongly adherence variability erodes declared boundary respect.",
        "evidence": "unvalidated_relationship",
    },
    "comfort_check_boundary_floor": {
        "label": "Comfort-check floor",
        "low": 0.0,
        "high": 1.0,
        "description": "Floor term linking comfort checks to effective boundary respect.",
        "evidence": "unvalidated_relationship",
    },
    "interaction_adaptation_pref_specificity": {
        "label": "Interaction: adaptation x specificity",
        "low": 0.0,
        "high": 3.0,
        "description": (
            "Extra effect of adaptation for profiles with strong, specific preferences "
            "(interaction term, centred on 0.5)."
        ),
        "evidence": "unvalidated_relationship",
    },
    "interaction_male_feedback_female_communication": {
        "label": "Interaction: male feedback x female communication",
        "low": 0.0,
        "high": 3.0,
        "description": (
            "Interaction: male feedback responsiveness pays off more with a partner who "
            "communicates preferences readily (centred on 0.5)."
        ),
        "evidence": "unvalidated_relationship",
    },
}


# --------------------------------------------------------------------------- #
# The configuration object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelConfig:
    """All numeric assumptions of the response model, in one immutable object."""

    # arousal
    arousal_intercept: float = -0.75
    w_arousal_consistency: float = 1.10
    w_arousal_adaptation: float = 0.95
    w_arousal_communication: float = 0.60
    w_arousal_bio_variability: float = 0.85
    w_arousal_psych_context: float = 0.70
    w_arousal_comfort: float = 0.50
    arousal_noise_sd: float = 0.55

    # pleasure
    pleasure_intercept: float = -1.15
    w_pleasure_arousal: float = 0.75
    w_pleasure_comfort: float = 0.90
    w_pleasure_preference_match: float = 1.00
    w_pleasure_communication: float = 0.50
    w_pleasure_consistency: float = 0.40
    w_pleasure_psych_context: float = 0.60
    w_pleasure_biological: float = 0.45
    pleasure_noise_sd: float = 0.50

    # comfort
    comfort_intercept: float = -0.55
    w_comfort_boundary_respect: float = 1.30
    w_comfort_comfort_checks: float = 1.00
    w_comfort_communication: float = 0.80
    w_comfort_psych_context: float = 0.70
    w_comfort_comfort_sensitivity: float = -1.10
    w_comfort_adherence: float = 0.45
    comfort_noise_sd: float = 0.45

    # satisfaction
    satisfaction_intercept: float = -0.80
    w_satisfaction_pleasure: float = 1.00
    w_satisfaction_comfort: float = 0.70
    w_satisfaction_communication: float = 0.50
    w_satisfaction_expectation_gap: float = -0.60
    satisfaction_noise_sd: float = 0.45

    # discomfort
    discomfort_intercept: float = -2.80
    w_discomfort_boundary_lapse: float = 2.00
    w_discomfort_comfort_check_gap: float = 1.00
    w_discomfort_comfort_sensitivity: float = 1.20
    w_discomfort_communication_gap: float = 0.90
    w_discomfort_adherence_instability: float = 0.80
    discomfort_noise_sd: float = 0.50

    # optional orgasm-related estimate (off by default)
    orgasm_intercept: float = -1.60
    w_orgasm_arousal: float = 1.20
    w_orgasm_pleasure: float = 0.80
    w_orgasm_comfort: float = 0.50
    orgasm_noise_sd: float = 0.60

    # structural constants
    adherence_erosion_weight: float = 0.50
    consistency_adherence_floor: float = 0.60
    feedback_protocol_weight: float = 0.50
    communication_protocol_weight: float = 0.50
    preference_match_floor: float = 0.55
    boundary_erosion_weight: float = 0.30
    comfort_check_boundary_floor: float = 0.50
    interaction_adaptation_pref_specificity: float = 0.80
    interaction_male_feedback_female_communication: float = 0.60

    # switches
    include_orgasm_estimate: bool = False

    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        for name, value in self.numeric_items().items():
            low = COEFFICIENT_SPECS[name]["low"]
            high = COEFFICIENT_SPECS[name]["high"]
            if not (low <= float(value) <= high):
                raise ValueError(
                    f"{name}={value} is outside the documented range [{low}, {high}]. "
                    "Values outside these bounds would make the demonstration "
                    "uninterpretable."
                )

    def numeric_items(self) -> dict[str, float]:
        """Return every numeric coefficient as a plain dict."""
        out: dict[str, float] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in COEFFICIENT_SPECS:
                out[field.name] = float(value)
        return out

    def with_overrides(self, **overrides: float) -> "ModelConfig":
        """Return a copy with some coefficients replaced (bounds-checked)."""
        known = {field.name for field in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown model parameters: {sorted(unknown)}")
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_version"] = MODEL_VERSION
        return data

    def evidence_table(self) -> pd.DataFrame:
        """A readable table of every coefficient, its range and its evidence status."""
        rows = []
        for name, value in self.numeric_items().items():
            spec = COEFFICIENT_SPECS[name]
            level = EVIDENCE_LEVELS[spec["evidence"]]
            rows.append(
                {
                    "parameter": name,
                    "value": round(value, 4),
                    "allowed_range": f"[{spec['low']}, {spec['high']}]",
                    "evidence_code": level["code"],
                    "evidence_status": level["label"],
                    "description": spec["description"],
                }
            )
        return pd.DataFrame(rows)

    def evidence_summary(self) -> pd.DataFrame:
        """Count of parameters per evidence level."""
        rows = []
        for key, level in EVIDENCE_LEVELS.items():
            count = sum(
                1 for name in self.numeric_items() if COEFFICIENT_SPECS[name]["evidence"] == key
            )
            rows.append(
                {
                    "code": level["code"],
                    "evidence_status": level["label"],
                    "parameters": count,
                    "meaning": level["meaning"],
                }
            )
        return pd.DataFrame(rows)


def default_config(include_orgasm_estimate: bool = False) -> ModelConfig:
    """The documented default configuration (all values hypothetical)."""
    return ModelConfig(include_orgasm_estimate=include_orgasm_estimate)


def clamp_parameter(name: str, value: float) -> float:
    """Clamp a value into the documented range for that parameter."""
    spec = COEFFICIENT_SPECS[name]
    return float(min(max(value, spec["low"]), spec["high"]))
