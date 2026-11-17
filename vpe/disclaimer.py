"""Shared integrity text, used by the app, the exports and the README."""

from __future__ import annotations

SYNTHETIC_DISCLAIMER = (
    "SYNTHETIC SIMULATION - NOT A MEASUREMENT OF REAL PEOPLE. Every row describes a "
    "computer-generated adult woman. The population's satisfaction level is calibrated to "
    "published aggregate survey percentages, but all technique, sequence and comfort "
    "mechanisms are hypothetical assumptions that no study has estimated. These outputs "
    "explore the consequences of stated assumptions; they do not establish that any "
    "technique, order or guide works in real people."
)

SHORT_DISCLAIMER = "Synthetic women, hypothetical mechanisms, calibrated null scenario."

KEY_LIMITATIONS: list[str] = [
    "No published study estimates the causal effect of a specific cunnilingus technique, "
    "pressure, rhythm or sequence. Every technique parameter is an explicit assumption.",
    "The population's satisfaction level is calibrated to published aggregate marginals "
    "(within about 2 percentage points per band at 1,000 women; more with smaller "
    "populations). Calibration fixes the level of the null scenario; it does not validate "
    "the mechanisms.",
    "Preference archetypes are a modelling device. Their weights are not prevalence "
    "estimates for any real population.",
    "The warm-up, overstimulation and rhythm mechanisms are anatomically plausible and "
    "consistent with the cited literature, but their functional forms are invented.",
    "Bootstrap intervals describe simulation noise across 1,000 synthetic women, "
    "conditional on the model. They are not confidence intervals for anything real.",
    "One-at-a-time sensitivity analysis shows which assumptions matter; it cannot show "
    "what happens when several assumptions are wrong together.",
    "Ordering effects are the least supported part of this model. Treat any ranking of "
    "sequences as a hypothesis to test, not a finding.",
    "The model cannot represent technique quality within a category: 'gentle suction' is "
    "one number, not a skill.",
]


def evidence_statement() -> str:
    """One paragraph stating exactly where this model's numbers come from."""
    return (
        "Evidence position. Six peer-reviewed sources inform this project. They support: the "
        "aggregate satisfaction marginals used to calibrate the null scenario (Ogallar-Blanco "
        "et al. 2022, N=1,080); that preference heterogeneity is large (Herbenick et al. 2018, "
        "N=1,055); that arousal, comfort, satisfaction and orgasm are separable (Rosen et al. "
        "2000); and that including more practices, especially cunnilingus, is associated with "
        "orgasm (Richters et al. 2006, N=19,307; Frederick et al. 2018, N=52,588). None of "
        "them estimates a technique-level or sequence-level effect. Every such parameter in "
        "this model is labelled an unvalidated assumption, and the app prints that label "
        "beside every result."
    )


CANNOT_ESTABLISH: list[str] = [
    "That any technique, order or guide improves real pleasure, comfort or satisfaction.",
    "The size, direction or existence of any real-world effect.",
    "Any real prevalence, success rate or percentage of women.",
    "Clinical validity of anything. These intervals are not clinical confidence intervals.",
    "That a calibrated model is a validated model.",
    "Anything about an individual person.",
]

CAN_ESTABLISH: list[str] = [
    "What a stated set of assumptions implies about a sequence of techniques.",
    "Which step of a guide is doing the work, under those assumptions (ablation).",
    "Whether ordering matters in this model, and by how much (sequence variants).",
    "Which preference profiles a technique is unsuitable for (subgroup analysis).",
    "How much of the answer rests on each coefficient (sensitivity).",
    "Whether the model reproduces published aggregate marginals (calibration check).",
]
