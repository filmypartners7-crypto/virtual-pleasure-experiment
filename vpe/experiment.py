"""The experiment engine.

Everything here answers one of the questions the tool exists for, and always on
the *same* 1,000 synthetic women:

* **A vs B** - two protocols, paired per woman.
* **Ablation** - remove one step at a time and see what that step was doing.
* **Sequence variants** - the same steps in a different order, including a
  no-warm-up variant and a reversed variant.
* **Substitution** - swap one technique for another at the same point.
* **Subgroups** - which archetypes gained, which lost.
* **Sensitivity** - how much the answer depends on the coefficients.

The comparisons are paired by construction: the population is generated once and
reused, so a difference cannot come from a different random draw.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .calibration import CALIBRATION_REPORT, calibration_table
from .disclaimer import KEY_LIMITATIONS, SYNTHETIC_DISCLAIMER, evidence_statement
from .evidence import (
    MODEL_VERSION,
    SSCI_PRACTISED_BANDS,
    evidence_summary_text,
    parameter_evidence_frame,
    studies_frame,
)
from .population import generate_population
from .protocol import BASELINE_PROTOCOL, Protocol
from .sequence_model import (
    DEFAULT_PARAMETERS,
    OUTCOME_COLUMNS,
    ModelParameters,
    simulate_protocol,
)
from .statistics import (
    condition_summary,
    paired_differences,
    paired_interval_table,
    subgroup_summary,
    summarise,
)

COMPARISON_OUTCOMES: tuple[str, ...] = OUTCOME_COLUMNS


@dataclass
class ExperimentResult:
    settings: dict[str, Any]
    profiles: pd.DataFrame
    condition_a: pd.DataFrame
    condition_b: pd.DataFrame
    paired: pd.DataFrame
    summary: pd.DataFrame
    intervals: pd.DataFrame
    summary_a: pd.DataFrame
    summary_b: pd.DataFrame
    subgroups: pd.DataFrame
    ablations: pd.DataFrame
    sequence_variants: pd.DataFrame
    substitutions: pd.DataFrame
    sensitivity: pd.DataFrame
    calibration: pd.DataFrame
    trajectory_a: pd.DataFrame | None
    trajectory_b: pd.DataFrame | None
    label_a: str
    label_b: str
    outcomes: list[str] = field(default_factory=list)

    @property
    def population_size(self) -> int:
        return len(self.profiles)

    @property
    def seed(self) -> int:
        return int(self.settings["random_seed"])

    def fingerprint(self) -> str:
        return str(self.settings["run_fingerprint"])


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("random seed must be an integer")
    seed = int(seed)
    if not (0 <= seed <= 2**32 - 1):
        raise ValueError("random seed must lie between 0 and 4294967295")
    return seed


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Component studies
# --------------------------------------------------------------------------- #

def ablation_study(
    profiles: pd.DataFrame,
    protocol: Protocol,
    parameters: ModelParameters,
) -> pd.DataFrame:
    """Remove one step at a time; report the change in each mean outcome."""
    full, _ = simulate_protocol(profiles, protocol, parameters)
    baseline_means = {outcome: float(full[outcome].mean()) for outcome in COMPARISON_OUTCOMES}

    rows = []
    for index, step in enumerate(protocol.steps):
        if len(protocol.steps) < 2:
            break
        try:
            reduced = protocol.with_step_disabled(index, name_suffix="")
        except ValueError:
            continue
        outcomes, _ = simulate_protocol(profiles, reduced, parameters)
        row = {
            "removed_step": index + 1,
            "removed_technique": step.technique.label,
            "removed_key": step.technique_key,
            "minutes_removed": round(float(step.duration_min), 2),
        }
        for outcome in COMPARISON_OUTCOMES:
            row[f"delta_{outcome}"] = float(outcomes[outcome].mean()) - baseline_means[outcome]
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["abs_effect"] = frame[[f"delta_{o}" for o in COMPARISON_OUTCOMES]].abs().max(axis=1)
        frame = frame.sort_values("abs_effect", ascending=False).reset_index(drop=True)
    return frame


def sequence_variants(
    profiles: pd.DataFrame,
    protocol: Protocol,
    parameters: ModelParameters,
) -> pd.DataFrame:
    """Same steps, different order - does ordering matter?"""
    variants: list[tuple[str, Protocol]] = []

    as_written, _ = simulate_protocol(profiles, protocol, parameters)
    base = {outcome: float(as_written[outcome].mean()) for outcome in COMPARISON_OUTCOMES}

    if len(protocol.steps) >= 2:
        variants.append(("as written", protocol))
        variants.append(("reversed order", protocol.reordered(list(reversed(range(len(protocol.steps)))))))

        # no warm-up: put the most direct step first
        most_direct = max(
            range(len(protocol.steps)), key=lambda i: protocol.steps[i].technique.directness
        )
        order = [most_direct] + [i for i in range(len(protocol.steps)) if i != most_direct]
        variants.append(("direct step moved to the front", protocol.reordered(order)))

        # no communication: drop every verbal step
        non_verbal = [i for i, s in enumerate(protocol.steps) if not s.technique.verbal]
        if 0 < len(non_verbal) < len(protocol.steps):
            variants.append(
                (
                    "communication removed",
                    Protocol(name="communication removed", steps=tuple(protocol.steps[i] for i in non_verbal)),
                )
            )

        # no warm-up at all: drop every low-directness step
        warmup_idx = [i for i, s in enumerate(protocol.steps) if s.technique.directness < 0.5]
        if 0 < len(warmup_idx) < len(protocol.steps):
            keep = [i for i in range(len(protocol.steps)) if i not in warmup_idx]
            variants.append(
                (
                    "warm-up steps removed",
                    Protocol(name="warm-up removed", steps=tuple(protocol.steps[i] for i in keep)),
                )
            )

    rows = []
    for label, variant in variants:
        outcomes, _ = simulate_protocol(profiles, variant, parameters)
        row = {"variant": label, "minutes": round(variant.duration_min, 1)}
        for outcome in COMPARISON_OUTCOMES:
            row[outcome] = float(outcomes[outcome].mean())
            row[f"delta_{outcome}"] = row[outcome] - base[outcome]
        rows.append(row)
    return pd.DataFrame(rows)


def substitution_study(
    profiles: pd.DataFrame,
    protocol: Protocol,
    parameters: ModelParameters,
    replacements: dict[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Swap a technique at the same position; report what changes."""
    default_map: dict[str, Sequence[str]] = {
        "clitoral_glans": ("clitoral_hood", "flat_tongue"),
        "suction": ("flat_tongue",),
        "clitoral_hood": ("clitoral_glans",),
        "flat_tongue": ("pointed_tongue",),
        "fingers_internal": ("fingers_external",),
    }
    replacements = replacements or default_map

    full, _ = simulate_protocol(profiles, protocol, parameters)
    base = {outcome: float(full[outcome].mean()) for outcome in COMPARISON_OUTCOMES}

    rows = []
    for index, step in enumerate(protocol.steps):
        for alternative in replacements.get(step.technique_key, ()):  # type: ignore[arg-type]
            try:
                swapped = protocol.with_step_replaced(index, alternative, name_suffix="")
            except (ValueError, IndexError):
                continue
            outcomes, _ = simulate_protocol(profiles, swapped, parameters)
            row = {
                "step": index + 1,
                "original_technique": step.technique.label,
                "replacement": swapped.steps[index].technique.label,
            }
            for outcome in COMPARISON_OUTCOMES:
                row[f"delta_{outcome}"] = float(outcomes[outcome].mean()) - base[outcome]
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Settings record
# --------------------------------------------------------------------------- #

def build_settings(
    population_size: int,
    seed: int,
    protocol_a: Protocol,
    protocol_b: Protocol,
    parameters: ModelParameters,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "application": "Virtual Pleasure Experiment (VPE)",
        "model_version": MODEL_VERSION,
        "population_size": int(population_size),
        "random_seed": int(seed),
        "design": (
            "paired sequence simulation: the same synthetic women are exposed to both "
            "protocols; differences are computed within woman"
        ),
        "label_a": label_a,
        "label_b": label_b,
        "protocol_a": protocol_a.to_dict(),
        "protocol_b": protocol_b.to_dict(),
        "model_parameters": parameters.to_dict(),
        "parameter_evidence": parameter_evidence_frame().to_dict(orient="records"),
        "evidence_position": evidence_statement(),
        "evidence_summary": evidence_summary_text(),
        "studies": studies_frame().to_dict(orient="records"),
        "calibration": CALIBRATION_REPORT,
        "calibration_target_bands": list(SSCI_PRACTISED_BANDS),
        "limitations": KEY_LIMITATIONS,
        "interval_labelling": (
            "bootstrap percentile intervals conditional on the model; never confidence "
            "intervals, never clinical"
        ),
        "disclaimer": SYNTHETIC_DISCLAIMER,
    }
    payload["run_fingerprint"] = _fingerprint(payload)
    return payload


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def run_experiment(
    population_size: int = 1000,
    seed: int = 42,
    protocol_a: Protocol | None = None,
    protocol_b: Protocol | None = None,
    parameters: ModelParameters | None = None,
    label_a: str = "Protocol A",
    label_b: str = "Protocol B",
    run_ablations: bool = True,
    run_variants: bool = True,
    run_substitutions: bool = True,
    run_sensitivity: bool = True,
) -> ExperimentResult:
    """Run a full paired experiment, including the component studies."""
    from .statistics import sensitivity_analysis

    seed = _validate_seed(seed)
    parameters = parameters or DEFAULT_PARAMETERS
    protocol_a = protocol_a or BASELINE_PROTOCOL
    protocol_b = protocol_b or protocol_a
    if not isinstance(protocol_a, Protocol) or not isinstance(protocol_b, Protocol):
        raise ValueError("protocol_a and protocol_b must be Protocol instances")

    profiles = generate_population(population_size, seed=seed)

    outcomes_a, trajectory_a = simulate_protocol(profiles, protocol_a, parameters, keep_trajectory=True)
    outcomes_b, trajectory_b = simulate_protocol(profiles, protocol_b, parameters, keep_trajectory=True)

    paired = paired_differences(outcomes_a, outcomes_b, COMPARISON_OUTCOMES)
    summary = condition_summary(outcomes_a, outcomes_b, label_a, label_b, COMPARISON_OUTCOMES)
    intervals = paired_interval_table(paired, COMPARISON_OUTCOMES, seed=seed)

    subgroups = subgroup_summary(paired, profiles, by="archetype", outcome="pleasure")

    ablations = ablation_study(profiles, protocol_b, parameters) if run_ablations else pd.DataFrame()
    variants = sequence_variants(profiles, protocol_b, parameters) if run_variants else pd.DataFrame()
    substitutions = (
        substitution_study(profiles, protocol_b, parameters) if run_substitutions else pd.DataFrame()
    )
    sensitivity = (
        sensitivity_analysis(profiles, protocol_b, parameters, outcomes=list(COMPARISON_OUTCOMES))
        if run_sensitivity
        else pd.DataFrame()
    )

    calibration = calibration_table(profiles, parameters)

    settings = build_settings(
        population_size, seed, protocol_a, protocol_b, parameters, label_a, label_b
    )

    return ExperimentResult(
        settings=settings,
        profiles=profiles,
        condition_a=outcomes_a,
        condition_b=outcomes_b,
        paired=paired,
        summary=summary,
        intervals=intervals,
        summary_a=summarise(outcomes_a),
        summary_b=summarise(outcomes_b),
        subgroups=subgroups,
        ablations=ablations,
        sequence_variants=variants,
        substitutions=substitutions,
        sensitivity=sensitivity,
        calibration=calibration,
        trajectory_a=trajectory_a,
        trajectory_b=trajectory_b,
        label_a=label_a,
        label_b=label_b,
        outcomes=list(COMPARISON_OUTCOMES),
    )
