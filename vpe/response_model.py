"""The response engine: a transparent, bounded, hypothetical model.

The model is a small system of logistic (sigmoid) equations. Nothing is fitted to
data, and no output is a clinical measurement. The equations are printed in
:func:`equations_text` and in the README so a reader can check them by hand.

Design rules enforced here
--------------------------
* Every outcome is ``sigmoid(latent)``, so it is strictly inside (0, 1).
* Subjective arousal, pleasure, comfort and satisfaction are *separate* outputs.
  Arousal is one moderate input to pleasure - the model does not equate them.
* Discomfort risk is a separate mechanism, not ``1 - comfort``.
* Random, unexplained variation is added on the latent scale (so bounds hold).
* Any orgasm-related estimate is optional, off by default, and labelled as an
  unvalidated hypothetical estimate.
* Only two interaction terms exist, and both are declared in ``model_config``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .model_config import ModelConfig
from .protocol import Protocol

OUTCOME_COLUMNS: list[str] = [
    "subjective_arousal",
    "pleasure",
    "comfort",
    "satisfaction",
    "discomfort_risk",
]
ORGASM_COLUMN = "orgasm_estimate_hypothetical"
INTERMEDIATE_COLUMNS: list[str] = [
    "adherence_effective",
    "consistency_effective",
    "feedback_loop",
    "communication_effective",
    "preference_match",
    "boundary_effective",
    "comfort_checks_effective",
    "safety_practices",
]
NOISE_CHANNELS: list[str] = [
    "arousal",
    "pleasure",
    "comfort",
    "satisfaction",
    "discomfort",
    "orgasm",
]


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically safe logistic function: sigma(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def make_noise(
    n: int, rng: np.random.Generator, include_orgasm: bool = False
) -> dict[str, np.ndarray]:
    """Draw standard normal noise, one independent channel per outcome.

    The experiment reuses the *same* noise draws in both conditions (common
    random numbers), so paired differences are not contaminated by noise
    differences between conditions.
    """
    channels = NOISE_CHANNELS if include_orgasm else NOISE_CHANNELS[:-1]
    return {channel: rng.standard_normal(n) for channel in channels}


def intermediate_terms(
    profiles: pd.DataFrame, protocol: Protocol, config: ModelConfig
) -> pd.DataFrame:
    """The protocol-derived terms that feed the outcome equations.

    Exposed separately so the dashboard can show *how* each slider enters the
    model, per couple, rather than hiding it inside one big formula.
    """
    p = protocol
    male_var = profiles["male_adherence_variability"].to_numpy(dtype=float)
    male_feedback = profiles["male_feedback_responsiveness"].to_numpy(dtype=float)
    female_specificity = profiles["female_preference_specificity"].to_numpy(dtype=float)
    female_comm_pref = profiles["female_communication_preference"].to_numpy(dtype=float)

    # Declared adherence is eroded by how variable the profile actually is.
    adherence_effective = _clip01(
        p.adherence - config.adherence_erosion_weight * male_var * (1.0 - p.adherence)
    )

    # Consistency only counts in proportion to how well the protocol is followed.
    consistency_effective = _clip01(
        p.consistency
        * (config.consistency_adherence_floor + (1.0 - config.consistency_adherence_floor) * adherence_effective)
    )

    # Couple-level feedback loop + declared interaction with communication preference.
    feedback_loop = _clip01(
        config.feedback_protocol_weight * (p.feedback_responsiveness + male_feedback)
        + config.interaction_male_feedback_female_communication * (female_comm_pref - 0.5)
    )

    communication_effective = _clip01(
        config.communication_protocol_weight * (p.communication + feedback_loop)
    )

    # Adaptation produces a match that scales with how specific the preferences are,
    # plus the declared interaction bonus.
    preference_match = _clip01(
        p.adaptation * (config.preference_match_floor + (1.0 - config.preference_match_floor) * female_specificity)
        + config.interaction_adaptation_pref_specificity * p.adaptation * (female_specificity - 0.5)
    )

    boundary_effective = _clip01(
        p.boundary_respect - config.boundary_erosion_weight * male_var * (1.0 - p.boundary_respect)
    )

    comfort_checks_effective = _clip01(
        p.comfort_checks * (config.comfort_check_boundary_floor + (1.0 - config.comfort_check_boundary_floor) * boundary_effective)
    )

    safety_practices = _clip01(0.5 * (boundary_effective + comfort_checks_effective))

    return pd.DataFrame(
        {
            "adherence_effective": adherence_effective,
            "consistency_effective": consistency_effective,
            "feedback_loop": feedback_loop,
            "communication_effective": communication_effective,
            "preference_match": preference_match,
            "boundary_effective": boundary_effective,
            "comfort_checks_effective": comfort_checks_effective,
            "safety_practices": safety_practices,
        },
        index=profiles.index,
    )


def simulate_outcomes(
    profiles: pd.DataFrame,
    protocol: Protocol,
    config: ModelConfig | None = None,
    noise: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Compute synthetic outcomes for one condition.

    Parameters
    ----------
    profiles:
        Output of :func:`vpe.population.generate_population`.
    protocol:
        The structured protocol to apply.
    config:
        Model coefficients. Defaults to :class:`ModelConfig`.
    noise:
        Optional pre-drawn noise (see :func:`make_noise`). If omitted, zero noise
        is used, which gives the *deterministic* part of the model.

    Returns
    -------
    pandas.DataFrame
        ``couple_id``, the five outcome columns, the optional orgasm-related
        column (only when enabled) and the intermediate terms.
    """
    config = config or ModelConfig()
    n = len(profiles)
    if noise is None:
        noise = {channel: np.zeros(n) for channel in NOISE_CHANNELS}
    for channel, draws in noise.items():
        if len(draws) != n:
            raise ValueError(f"noise channel '{channel}' has length {len(draws)}, expected {n}")

    terms = intermediate_terms(profiles, protocol, config)

    female_bio = profiles["female_biological_response_variability"].to_numpy(dtype=float)
    female_psych = profiles["female_psychological_context"].to_numpy(dtype=float)
    female_specificity = profiles["female_preference_specificity"].to_numpy(dtype=float)
    female_comfort_sens = profiles["female_comfort_sensitivity"].to_numpy(dtype=float)

    # ---------------- subjective arousal ----------------
    z_arousal = (
        config.arousal_intercept
        + config.w_arousal_consistency * terms["consistency_effective"].to_numpy()
        + config.w_arousal_adaptation * terms["preference_match"].to_numpy()
        + config.w_arousal_communication * terms["communication_effective"].to_numpy()
        + config.w_arousal_bio_variability * (female_bio - 0.5)
        + config.w_arousal_psych_context * (female_psych - 0.5)
        + config.w_arousal_comfort * terms["safety_practices"].to_numpy()
        + config.arousal_noise_sd * noise["arousal"]
    )
    arousal = sigmoid(z_arousal)

    # ---------------- comfort ----------------
    z_comfort = (
        config.comfort_intercept
        + config.w_comfort_boundary_respect * terms["boundary_effective"].to_numpy()
        + config.w_comfort_comfort_checks * terms["comfort_checks_effective"].to_numpy()
        + config.w_comfort_communication * terms["communication_effective"].to_numpy()
        + config.w_comfort_psych_context * (female_psych - 0.5)
        + config.w_comfort_comfort_sensitivity * (female_comfort_sens - 0.5)
        + config.w_comfort_adherence * terms["adherence_effective"].to_numpy()
        + config.comfort_noise_sd * noise["comfort"]
    )
    comfort = sigmoid(z_comfort)

    # ---------------- pleasure (arousal is one input, not the definition) ------------
    z_pleasure = (
        config.pleasure_intercept
        + config.w_pleasure_arousal * arousal
        + config.w_pleasure_comfort * comfort
        + config.w_pleasure_preference_match * terms["preference_match"].to_numpy()
        + config.w_pleasure_communication * terms["communication_effective"].to_numpy()
        + config.w_pleasure_consistency * terms["consistency_effective"].to_numpy()
        + config.w_pleasure_psych_context * (female_psych - 0.5)
        + config.w_pleasure_biological * (female_bio - 0.5)
        + config.pleasure_noise_sd * noise["pleasure"]
    )
    pleasure = sigmoid(z_pleasure)

    # ---------------- satisfaction ----------------
    z_satisfaction = (
        config.satisfaction_intercept
        + config.w_satisfaction_pleasure * pleasure
        + config.w_satisfaction_comfort * comfort
        + config.w_satisfaction_communication * terms["communication_effective"].to_numpy()
        + config.w_satisfaction_expectation_gap
        * (female_specificity * (1.0 - terms["preference_match"].to_numpy()))
        + config.satisfaction_noise_sd * noise["satisfaction"]
    )
    satisfaction = sigmoid(z_satisfaction)

    # ---------------- discomfort risk (separate mechanism) ----------------
    z_discomfort = (
        config.discomfort_intercept
        + config.w_discomfort_boundary_lapse * (1.0 - terms["boundary_effective"].to_numpy())
        + config.w_discomfort_comfort_check_gap * (1.0 - terms["comfort_checks_effective"].to_numpy())
        + config.w_discomfort_comfort_sensitivity * (female_comfort_sens - 0.5)
        + config.w_discomfort_communication_gap * (1.0 - terms["communication_effective"].to_numpy())
        + config.w_discomfort_adherence_instability * (1.0 - terms["consistency_effective"].to_numpy())
        + config.discomfort_noise_sd * noise["discomfort"]
    )
    discomfort_risk = sigmoid(z_discomfort)

    out = pd.DataFrame(
        {
            "couple_id": profiles["couple_id"].to_numpy(),
            "subjective_arousal": _clip01(arousal),
            "pleasure": _clip01(pleasure),
            "comfort": _clip01(comfort),
            "satisfaction": _clip01(satisfaction),
            "discomfort_risk": _clip01(discomfort_risk),
        }
    )

    if config.include_orgasm_estimate:
        z_orgasm = (
            config.orgasm_intercept
            + config.w_orgasm_arousal * arousal
            + config.w_orgasm_pleasure * pleasure
            + config.w_orgasm_comfort * comfort
            + config.orgasm_noise_sd * noise.get("orgasm", np.zeros(n))
        )
        out[ORGASM_COLUMN] = _clip01(sigmoid(z_orgasm))

    for column in INTERMEDIATE_COLUMNS:
        out[column] = terms[column].to_numpy()

    return out


def outcome_columns(config: ModelConfig | None = None) -> list[str]:
    """Outcome columns produced by :func:`simulate_outcomes` for this config."""
    config = config or ModelConfig()
    columns = list(OUTCOME_COLUMNS)
    if config.include_orgasm_estimate:
        columns.append(ORGASM_COLUMN)
    return columns


def equations_text(config: ModelConfig | None = None) -> str:
    """The model equations, with the current coefficient values substituted."""
    cfg = config or ModelConfig()
    orgasm_block = ""
    if cfg.include_orgasm_estimate:
        orgasm_block = (
            "\nOPTIONAL ORGASM-RELATED ESTIMATE (unvalidated, hypothetical, not a prediction)\n"
            f"  z_O = {cfg.orgasm_intercept} + {cfg.w_orgasm_arousal}*A + {cfg.w_orgasm_pleasure}*P"
            f" + {cfg.w_orgasm_comfort}*C + {cfg.orgasm_noise_sd}*eps_O\n"
            "  O   = sigma(z_O)\n"
        )
    return (
        "DEFINITIONS (all terms are per synthetic couple)\n"
        "  sigma(x) = 1 / (1 + exp(-x))           (bounded in (0,1))\n"
        "  adh_eff  = clip01(adh - w_e*var_m*(1 - adh))\n"
        "  con_eff  = clip01(con*(f_c + (1 - f_c)*adh_eff))\n"
        "  fb_loop  = clip01(w_f*(fb + fb_m) + i_1*(comm_pref_f - 0.5))\n"
        "  comm_eff = clip01(w_c*(comm + fb_loop))\n"
        "  match    = clip01(adapt*(f_p + (1 - f_p)*spec_f) + i_2*adapt*(spec_f - 0.5))\n"
        "  bound_eff= clip01(bound - w_b*var_m*(1 - bound))\n"
        "  cc_eff   = clip01(checks*(f_cc + (1 - f_cc)*bound_eff))\n"
        "  safety   = clip01(0.5*(bound_eff + cc_eff))\n"
        "\nSTRUCTURAL CONSTANTS (values substituted into the definitions above)\n"
        f"  w_e (adherence_erosion_weight)   = {cfg.adherence_erosion_weight}\n"
        f"  f_c (consistency_adherence_floor) = {cfg.consistency_adherence_floor}\n"
        f"  w_f (feedback_protocol_weight)   = {cfg.feedback_protocol_weight}\n"
        f"  w_c (communication_protocol_weight) = {cfg.communication_protocol_weight}\n"
        f"  f_p (preference_match_floor)     = {cfg.preference_match_floor}\n"
        f"  w_b (boundary_erosion_weight)    = {cfg.boundary_erosion_weight}\n"
        f"  f_cc (comfort_check_boundary_floor) = {cfg.comfort_check_boundary_floor}\n"
        f"  i_1 (interaction_male_feedback_female_communication) = "
        f"{cfg.interaction_male_feedback_female_communication}\n"
        f"  i_2 (interaction_adaptation_pref_specificity) = {cfg.interaction_adaptation_pref_specificity}\n"
        "\nOUTCOMES\n"
        f"  z_A = {cfg.arousal_intercept} + {cfg.w_arousal_consistency}*con_eff"
        f" + {cfg.w_arousal_adaptation}*match + {cfg.w_arousal_communication}*comm_eff"
        f" + {cfg.w_arousal_bio_variability}*(bio_f - 0.5) + {cfg.w_arousal_psych_context}*(psy_f - 0.5)"
        f" + {cfg.w_arousal_comfort}*safety + {cfg.arousal_noise_sd}*eps_A\n"
        "  A   = sigma(z_A)                       subjective arousal estimate\n"
        f"  z_C = {cfg.comfort_intercept} + {cfg.w_comfort_boundary_respect}*bound_eff"
        f" + {cfg.w_comfort_comfort_checks}*cc_eff + {cfg.w_comfort_communication}*comm_eff"
        f" + {cfg.w_comfort_psych_context}*(psy_f - 0.5) + {cfg.w_comfort_comfort_sensitivity}*(csens_f - 0.5)"
        f" + {cfg.w_comfort_adherence}*adh_eff + {cfg.comfort_noise_sd}*eps_C\n"
        "  C   = sigma(z_C)                       comfort estimate\n"
        f"  z_P = {cfg.pleasure_intercept} + {cfg.w_pleasure_arousal}*A + {cfg.w_pleasure_comfort}*C"
        f" + {cfg.w_pleasure_preference_match}*match + {cfg.w_pleasure_communication}*comm_eff"
        f" + {cfg.w_pleasure_consistency}*con_eff + {cfg.w_pleasure_psych_context}*(psy_f - 0.5)"
        f" + {cfg.w_pleasure_biological}*(bio_f - 0.5) + {cfg.pleasure_noise_sd}*eps_P\n"
        "  P   = sigma(z_P)                       pleasure estimate (NOT a clinical measurement)\n"
        f"  z_S = {cfg.satisfaction_intercept} + {cfg.w_satisfaction_pleasure}*P"
        f" + {cfg.w_satisfaction_comfort}*C + {cfg.w_satisfaction_communication}*comm_eff"
        f" + {cfg.w_satisfaction_expectation_gap}*(spec_f*(1 - match)) + {cfg.satisfaction_noise_sd}*eps_S\n"
        "  S   = sigma(z_S)                       satisfaction estimate\n"
        f"  z_D = {cfg.discomfort_intercept} + {cfg.w_discomfort_boundary_lapse}*(1 - bound_eff)"
        f" + {cfg.w_discomfort_comfort_check_gap}*(1 - cc_eff) + {cfg.w_discomfort_comfort_sensitivity}*(csens_f - 0.5)"
        f" + {cfg.w_discomfort_communication_gap}*(1 - comm_eff)"
        f" + {cfg.w_discomfort_adherence_instability}*(1 - con_eff) + {cfg.discomfort_noise_sd}*eps_D\n"
        "  D   = sigma(z_D)                       discomfort risk indicator (separate mechanism)\n"
        f"{orgasm_block}"
        "\nINTERPRETATION\n"
        "  All outcomes are unitless indices in (0,1) produced by hypothetical equations.\n"
        "  They are not probabilities of any real event, not percentages of real people\n"
        "  and not measurements of anybody's body or experience.\n"
        "\nPARAMETERS USED IN THIS RUN (name = value; none of these is fitted to data)\n"
        + "".join(f"  {name} = {value}\n" for name, value in cfg.numeric_items().items())
        + f"  include_orgasm_estimate = {cfg.include_orgasm_estimate}\n"
    )


def assumptions_summary(config: ModelConfig | None = None) -> list[dict[str, Any]]:
    """Machine-readable list of the structural assumptions behind the equations."""
    cfg = config or ModelConfig()
    return [
        {
            "assumption": "Arousal does not define pleasure",
            "implementation": f"pleasure weight on arousal = {cfg.w_pleasure_arousal} (moderate, not dominant)",
            "evidence": "research_supported_concept (concept only; the number is hypothetical)",
        },
        {
            "assumption": "Adherence variability erodes declared protocol adherence",
            "implementation": f"adherence_erosion_weight = {cfg.adherence_erosion_weight}",
            "evidence": "unvalidated_relationship",
        },
        {
            "assumption": "Adaptation matters more for specific preferences",
            "implementation": f"interaction_adaptation_pref_specificity = {cfg.interaction_adaptation_pref_specificity}",
            "evidence": "unvalidated_relationship",
        },
        {
            "assumption": "Male feedback pays off more with a partner who communicates",
            "implementation": (
                f"interaction_male_feedback_female_communication = "
                f"{cfg.interaction_male_feedback_female_communication}"
            ),
            "evidence": "unvalidated_relationship",
        },
        {
            "assumption": "Discomfort risk has its own mechanism",
            "implementation": "separate logistic equation driven by boundary lapses, absent checks, gaps and instability",
            "evidence": "research_supported_concept (concept only)",
        },
        {
            "assumption": "Unexplained variation is normal on the latent scale",
            "implementation": (
                f"noise sd: arousal {cfg.arousal_noise_sd}, pleasure {cfg.pleasure_noise_sd}, "
                f"comfort {cfg.comfort_noise_sd}, satisfaction {cfg.satisfaction_noise_sd}, "
                f"discomfort {cfg.discomfort_noise_sd}"
            ),
            "evidence": "hypothetical_assumption",
        },
    ]
