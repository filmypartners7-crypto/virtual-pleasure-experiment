"""Virtual Pleasure Experiment (VPE).

A small, transparent computational demonstration: a synthetic population of adult
couples, a configurable protocol, a hypothetical response model and a paired
comparison. Everything it produces is simulated. Read ``vpe.disclaimer`` before
interpreting any number it prints.
"""

from __future__ import annotations

from .disclaimer import (
    CANNOT_ESTABLISH,
    CAN_ESTABLISH,
    EVIDENCE_STATUS_STATEMENT,
    KEY_ASSUMPTIONS,
    SHORT_DISCLAIMER,
    SYNTHETIC_DISCLAIMER,
)
from .experiment import ExperimentResult, build_settings, rerun_with_seed, run_experiment
from .empirical_scenario import run_empirical_satisfaction_scenario, empirical_scenario_distribution
from .research import RESEARCH_STUDIES, research_metadata, reference_distribution_table, satisfaction_factors_table
from .model_config import (
    COEFFICIENT_SPECS,
    EVIDENCE_LEVELS,
    MODEL_VERSION,
    ModelConfig,
    default_config,
)
from .population import (
    AGE_CATEGORIES,
    POPULATION_DISTRIBUTIONS,
    PROFILE_COLUMNS,
    distributions_table,
    generate_population,
    population_variability,
)
from .protocol import (
    BASELINE_DEFINITION,
    BASELINE_PROTOCOL,
    PROTOCOL_VARIABLE_INFO,
    PROTOCOL_VARIABLES,
    Protocol,
    default_guide_protocol,
)
from .response_model import (
    ORGASM_COLUMN,
    OUTCOME_COLUMNS,
    assumptions_summary,
    equations_text,
    outcome_columns,
    simulate_outcomes,
)

__version__ = MODEL_VERSION

__all__ = [
    "AGE_CATEGORIES",
    "BASELINE_DEFINITION",
    "BASELINE_PROTOCOL",
    "CANNOT_ESTABLISH",
    "CAN_ESTABLISH",
    "COEFFICIENT_SPECS",
    "EVIDENCE_LEVELS",
    "EVIDENCE_STATUS_STATEMENT",
    "ExperimentResult",
    "RESEARCH_STUDIES",
    "KEY_ASSUMPTIONS",
    "MODEL_VERSION",
    "ModelConfig",
    "ORGASM_COLUMN",
    "OUTCOME_COLUMNS",
    "POPULATION_DISTRIBUTIONS",
    "PROFILE_COLUMNS",
    "PROTOCOL_VARIABLES",
    "PROTOCOL_VARIABLE_INFO",
    "Protocol",
    "SHORT_DISCLAIMER",
    "SYNTHETIC_DISCLAIMER",
    "assumptions_summary",
    "build_settings",
    "default_config",
    "default_guide_protocol",
    "distributions_table",
    "equations_text",
    "generate_population",
    "empirical_scenario_distribution",
    "reference_distribution_table",
    "research_metadata",
    "run_empirical_satisfaction_scenario",
    "satisfaction_factors_table",
    "outcome_columns",
    "population_variability",
    "rerun_with_seed",
    "run_experiment",
    "simulate_outcomes",
]
