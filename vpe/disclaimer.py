"""Shared integrity text used by the app, exports and README."""

from __future__ import annotations

SYNTHETIC_DISCLAIMER = (
    "SYNTHETIC DATA - NOT REAL PARTICIPANTS. Each row represents a computer-generated "
    "adult couple. The empirical reference-category field is sampled from rounded "
    "aggregate percentages reported in a published Spanish women's survey (N=1,080); "
    "it is not an original participant record or a representative global sample. "
    "All continuous response-model coefficients, individual profile distributions, "
    "and the assumed guide-effect parameter are hypothetical and have not been fitted "
    "to participant-level data. These scenario outputs do NOT establish that a guide "
    "works in real humans."
)

SHORT_DISCLAIMER = (
    "Synthetic scenario: one reference distribution uses published aggregate survey "
    "percentages; guide effects and response equations remain hypothetical."
)

EVIDENCE_STATUS_STATEMENT = (
    "VPE 0.2.0 includes one empirical aggregate reference distribution for reported "
    "satisfaction with receiving oral sex (cunnilingus). The source is a voluntary, "
    "non-probability sample of 1,080 Spanish women aged 18-50. No participant-level "
    "records are used. The source does not test this guide. All continuous response "
    "equation coefficients and the guide-effect parameter remain hypothetical; they "
    "are not research-estimated."
)

KEY_ASSUMPTIONS: list[str] = [
    "The empirical reference category is resampled from rounded published aggregate percentages in Ogallar-Blanco et al. (2022), Table 4. The synthetic cohort is not the original sample, and the source was a voluntary non-probability sample of Spanish women aged 18-50.",
    "All other individual profile variables use convenience distributions chosen for this demonstration. They are not prevalence estimates for any real population.",
    "The guide's effect on the empirical ordinal satisfaction category is an explicit hypothetical logit-shift parameter. The source survey did not test this guide or estimate the parameter.",
    "The guide's structured protocol score is the equal-weight mean of seven sliders. Equal weighting is an engineering convention, not a validated scale.",
    "All continuous response-model outcomes are produced by logistic latent-variable equations. Their values are unitless model indices, not clinical measurements or probabilities of real events.",
    "Subjective arousal, pleasure, comfort and satisfaction are modelled as separate outputs. Arousal is not treated as equivalent to pleasure.",
    "Discomfort risk is a separate hypothetical mechanism. It is not defined as one minus comfort.",
    "The same synthetic population is used in both continuous model conditions and the same random noise draws are reused (common random numbers). This makes the model comparison paired, but does not create a real control group.",
    "The baseline condition is a fixed modelling convention, not a measured real-world control condition. Its values influence modelled differences.",
    "The empirical satisfaction scenario uses a latent logistic utility sampled within the published ordinal categories. This latent variable is a mathematical device, not a measurement reported in the study.",
    "Bootstrap intervals describe statistical variation conditional on the synthetic model and its assumptions. They do not quantify uncertainty in the assumptions or provide clinical confidence intervals.",
    "The Herbenick et al. (2018) study is used only to support preference heterogeneity; its clitoral-stimulation percentages concern intercourse, not cunnilingus technique.",
]

CANNOT_ESTABLISH: list[str] = [
    "That this guide improves pleasure, comfort, satisfaction or orgasm for real women.",
    "The size or direction of any real-world effect of the guide.",
    "A global or population-wide prevalence or success rate. The included reference survey is a self-selected Spanish sample.",
    "That factors reported as influencing satisfaction are causal effects.",
    "Clinical validity, individual predictions or clinical confidence intervals.",
    "Causal evidence. Sensitivity analysis describes the equations, not the real world.",
]

CAN_ESTABLISH: list[str] = [
    "The published aggregate satisfaction categories used to seed the synthetic reference cohort.",
    "What a clearly stated hypothetical guide-effect scenario implies under the chosen mathematical assumptions.",
    "How outputs change when hypothetical effect assumptions or protocol inputs change.",
    "Which research claims are directly relevant, indirectly relevant or not suitable for numerical calibration.",
    "Whether the code is reproducible and whether its internal calculations are mathematically consistent.",
]
