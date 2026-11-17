"""The paired experiment engine.

Design: one synthetic population, two conditions.

* **Baseline** - a fixed, visible modelling convention (see
  :data:`vpe.protocol.BASELINE_DEFINITION`). Not a control group.
* **Guide** - the user-configured protocol.

Both conditions run on the *same* couples with the *same* random noise draws
(common random numbers), so a paired difference reflects the protocol change and
its interactions with profile variables rather than noise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .disclaimer import EVIDENCE_STATUS_STATEMENT, KEY_ASSUMPTIONS, SYNTHETIC_DISCLAIMER
from .model_config import MODEL_VERSION, ModelConfig, default_config
from .empirical_scenario import run_empirical_satisfaction_scenario
from .research import research_metadata
from .population import generate_population
from .protocol import BASELINE_DEFINITION, BASELINE_PROTOCOL, Protocol
from .response_model import (
    assumptions_summary,
    equations_text,
    make_noise,
    outcome_columns,
    simulate_outcomes,
)
from .statistics import condition_summary, paired_differences, summarise


@dataclass(frozen=True)
class ExperimentResult:
    """Everything produced by one run of the experiment."""

    settings: dict[str, Any]
    profiles: pd.DataFrame
    baseline: pd.DataFrame
    guide: pd.DataFrame
    paired: pd.DataFrame
    summary: pd.DataFrame
    baseline_summary: pd.DataFrame
    guide_summary: pd.DataFrame
    empirical_scenario: pd.DataFrame
    empirical_scenario_metadata: dict[str, Any]
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
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("random seed must lie between 0 and 4294967295")
    return seed


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_settings(
    population_size: int,
    seed: int,
    protocol: Protocol,
    baseline_protocol: Protocol,
    config: ModelConfig,
    assumed_effect_logit: float = 0.0,
    empirical_scenario_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The complete, exportable record of one experiment's configuration."""
    payload: dict[str, Any] = {
        "application": "Virtual Pleasure Experiment (VPE)",
        "model_version": MODEL_VERSION,
        "population_size": int(population_size),
        "random_seed": int(seed),
        "assumed_effect_logit": float(assumed_effect_logit),
        "design": "paired synthetic model comparison plus a separate ordinal scenario anchored to published aggregate satisfaction percentages",
        "empirical_reference": research_metadata(),
        "empirical_scenario_metadata": empirical_scenario_metadata or {},
        "data_source_statement": (
            "Baseline ordinal satisfaction categories are synthetic resamples of rounded published aggregate percentages. "
            "The guide effect is hypothetical and is not estimated by the source study."
        ),
        "protocol": protocol.to_dict(),
        "baseline_protocol": baseline_protocol.to_dict(),
        "baseline_definition": BASELINE_DEFINITION,
        "model_config": config.to_dict(),
        "outcome_columns": outcome_columns(config),
        "coefficient_evidence": config.evidence_summary().to_dict(orient="records"),
        "model_assumptions": assumptions_summary(config),
        "documented_assumptions": KEY_ASSUMPTIONS,
        "evidence_status_statement": EVIDENCE_STATUS_STATEMENT,
        "equations": equations_text(config),
        "disclaimer": SYNTHETIC_DISCLAIMER,
        "interval_labelling": "bootstrap percentile intervals, conditional on the model; never clinical confidence intervals",
    }
    payload["run_fingerprint"] = _fingerprint(payload)
    return payload


def run_experiment(
    population_size: int = 1000,
    seed: int = 42,
    protocol: Protocol | None = None,
    config: ModelConfig | None = None,
    baseline_protocol: Protocol | None = None,
    assumed_effect_logit: float = 0.0,
) -> ExperimentResult:
    """Generate a population and evaluate both conditions on it.

    Parameters
    ----------
    population_size:
        Number of synthetic couples (default 1000).
    seed:
        Random seed for the population and the noise draws.
    protocol:
        The user-configured guide condition.
    config:
        Model coefficients (defaults to the documented hypothetical values).
    baseline_protocol:
        Overrides the baseline convention. Exposed for transparency only.
    """
    seed = _validate_seed(seed)
    config = config or default_config()
    if protocol is None:
        raise ValueError("a user-configured protocol is required for the guide condition")
    if not isinstance(protocol, Protocol):
        raise ValueError("protocol must be a vpe.protocol.Protocol instance")
    baseline_protocol = baseline_protocol or BASELINE_PROTOCOL
    if not isinstance(baseline_protocol, Protocol):
        raise ValueError("baseline_protocol must be a vpe.protocol.Protocol instance")

    # Independent, reproducible streams: one for the population, one for the noise.
    seed_sequence = np.random.SeedSequence(seed)
    population_entropy, noise_entropy = seed_sequence.spawn(2)
    population_seed = int(population_entropy.generate_state(1)[0])
    noise_seed = int(noise_entropy.generate_state(1)[0])

    profiles = generate_population(population_size, seed=population_seed)

    noise = make_noise(
        len(profiles),
        np.random.default_rng(noise_seed),
        include_orgasm=config.include_orgasm_estimate,
    )

    baseline_outcomes = simulate_outcomes(profiles, baseline_protocol, config, noise)
    guide_outcomes = simulate_outcomes(profiles, protocol, config, noise)

    outcomes = outcome_columns(config)
    paired = paired_differences(baseline_outcomes, guide_outcomes, outcomes)
    summary = condition_summary(baseline_outcomes, guide_outcomes, outcomes)

    empirical_scenario, empirical_metadata = run_empirical_satisfaction_scenario(
        profiles, protocol, baseline_protocol, assumed_effect_logit=assumed_effect_logit
    )
    settings = build_settings(
        population_size, seed, protocol, baseline_protocol, config,
        assumed_effect_logit=assumed_effect_logit,
        empirical_scenario_metadata=empirical_metadata,
    )

    return ExperimentResult(
        settings=settings,
        profiles=profiles,
        baseline=baseline_outcomes,
        guide=guide_outcomes,
        paired=paired,
        summary=summary,
        baseline_summary=summarise(baseline_outcomes, outcomes),
        guide_summary=summarise(guide_outcomes, outcomes),
        empirical_scenario=empirical_scenario,
        empirical_scenario_metadata=empirical_metadata,
        outcomes=outcomes,
    )


def rerun_with_seed(result: ExperimentResult, new_seed: int) -> ExperimentResult:
    """Re-run the same experiment with a different random seed."""
    protocol = Protocol(**{k: v for k, v in result.settings["protocol"].items() if k in Protocol.__dataclass_fields__})
    config_dict = {
        k: v for k, v in result.settings["model_config"].items() if k in ModelConfig.__dataclass_fields__
    }
    baseline_dict = {
        k: v for k, v in result.settings["baseline_protocol"].items()
        if k in Protocol.__dataclass_fields__
    }
    return run_experiment(
        population_size=result.population_size,
        seed=new_seed,
        protocol=protocol,
        baseline_protocol=Protocol(**baseline_dict),
        config=ModelConfig(**config_dict),
        assumed_effect_logit=float(result.settings.get("assumed_effect_logit", 0.0)),
    )
