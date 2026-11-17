"""Published human-research anchors used by VPE 0.2.0.

Only rounded aggregate percentages reported in the cited papers are embedded.
No participant-level records are reproduced. The empirical cunnilingus-satisfaction
marginals are used to seed a synthetic reference cohort; they do not estimate the
effect of any particular guide or technique.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MODEL_REFERENCE_ID = "ssci_2022_receiving_oral_sex"

# Ogallar-Blanco et al. (2022), Table 4, N=1,080 Spanish women aged 18-50.
# Response options for satisfaction associated with receiving oral sex
# (cunnilingus), expressed as percentages of the study sample. Rounded to 0.1%.
SSCI_RECEIVING_ORAL_SEX = [
    {"category": "Not practiced", "key": "not_practiced", "percent": 5.3, "score": None},
    {"category": "Not satisfactory at all", "key": "not_satisfactory", "percent": 3.1, "score": 1},
    {"category": "Somewhat satisfactory", "key": "somewhat_satisfactory", "percent": 10.5, "score": 2},
    {"category": "Quite satisfactory", "key": "quite_satisfactory", "percent": 21.9, "score": 3},
    {"category": "Very satisfactory", "key": "very_satisfactory", "percent": 59.2, "score": 4},
]

# Table 3 of the same study: percentage selecting "It affects me a lot".
# These are self-reported perceived influences on sexual satisfaction, not
# causal effects and not cunnilingus-specific estimates.
SSCI_SATISFACTION_FACTORS = [
    {"factor": "Partner attraction", "percent_affects_a_lot": 72.5},
    {"factor": "Not feeling obliged to do unwanted activity", "percent_affects_a_lot": 72.1},
    {"factor": "Confidence with sexual partner", "percent_affects_a_lot": 66.8},
    {"factor": "Feeling comfortable with own body", "percent_affects_a_lot": 63.3},
    {"factor": "Feeling relaxed and comfortable", "percent_affects_a_lot": 51.5},
    {"factor": "Partner's actions / techniques", "percent_affects_a_lot": 53.7},
    {"factor": "Communication about likes, dislikes and boundaries", "percent_affects_a_lot": 39.3},
]

RESEARCH_STUDIES: list[dict[str, Any]] = [
    {
        "id": "ssci_2022",
        "title": "Going beyond ‘With a Partner’ and ‘Intercourse’: Does Anything Else Influence Sexual Satisfaction among Women? The Sexual Satisfaction Comprehensive Index",
        "authors": "Ogallar-Blanco, Lara-Moreno & Godoy-Izquierdo",
        "year": 2022,
        "journal": "International Journal of Environmental Research and Public Health, 19(16), 10232",
        "doi": "10.3390/ijerph191610232",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9408228/",
        "sample": "N=1,080 women aged 18-50 in Spain; voluntary online, non-probability sample",
        "relevance": "Direct self-reported satisfaction ratings for receiving oral sex (cunnilingus), plus broader factors respondents said influence sexual satisfaction.",
        "limitations": "Self-selected Spanish sample; not a representative global sample. Cross-sectional self-report. The factor ratings are perceived influence, not causal effects. No evaluation of this user's guide.",
        "use_in_vpe": "Rounded aggregate cunnilingus-satisfaction percentages seed the empirical reference distribution. Table 3 factor percentages are displayed as descriptive evidence only.",
    },
    {
        "id": "herbenick_2018",
        "title": "Women's Experiences With Genital Touching, Sexual Pleasure, and Orgasm: Results From a U.S. Probability Sample of Women Ages 18 to 94",
        "authors": "Herbenick, Fu, Arter, Sanders & Dodge",
        "year": 2018,
        "journal": "Journal of Sex & Marital Therapy, 44(2), 201-212",
        "doi": "10.1080/0092623X.2017.1346530",
        "url": "https://pubmed.ncbi.nlm.nih.gov/28678639/",
        "sample": "N=1,055 women aged 18-94 from a U.S. probability sample panel",
        "relevance": "Reports variation in genital-touch preferences. For orgasm during intercourse, 36.6% said clitoral stimulation was necessary and a further 36.0% said it made orgasms feel better.",
        "limitations": "The clitoral-stimulation figures concern intercourse, not cunnilingus or a specific oral-sex technique. The study reports diversity rather than one universally preferred pattern.",
        "use_in_vpe": "Supports modelling preference heterogeneity and avoiding one-size-fits-all claims. It does not supply coefficients for this guide's effect.",
    },
    {
        "id": "backstrom_2012",
        "title": "Women's Negotiation of Cunnilingus in College Hookups and Relationships",
        "authors": "Backstrom, Armstrong & Puentes",
        "year": 2012,
        "journal": "Journal of Sex Research, 49(1), 1-12",
        "doi": "10.1080/00224499.2011.585523",
        "url": "https://pubmed.ncbi.nlm.nih.gov/22010825/",
        "sample": "In-depth interviews with 43 college women; mean age approximately 21",
        "relevance": "Shows that attitudes and negotiation around cunnilingus can differ by relationship context and individual preference.",
        "limitations": "Small qualitative sample; not a prevalence estimate and not suitable for numerical population calibration.",
        "use_in_vpe": "Qualitative support for context and individual variation; no numerical effect size is imported.",
    },
    {
        "id": "rosen_2000_fsfi",
        "title": "The Female Sexual Function Index (FSFI): A Multidimensional Self-Report Instrument for the Assessment of Female Sexual Function",
        "authors": "Rosen et al.",
        "year": 2000,
        "journal": "Journal of Sex & Marital Therapy, 26(2), 191-208",
        "doi": "10.1080/009262300278597",
        "url": "https://pubmed.ncbi.nlm.nih.gov/10782451/",
        "sample": "131 controls and 128 age-matched participants with female sexual arousal disorder",
        "relevance": "Validated six-domain instrument covering desire, subjective arousal, lubrication, orgasm, satisfaction and pain.",
        "limitations": "A measurement instrument, not a study of cunnilingus-guide effectiveness. Its domain scores are not model coefficients for VPE.",
        "use_in_vpe": "Supports keeping sexual-function dimensions separate rather than collapsing them into one pleasure score.",
    },
]


def reference_distribution_table() -> pd.DataFrame:
    """Published aggregate response percentages used for the synthetic anchor."""
    return pd.DataFrame(SSCI_RECEIVING_ORAL_SEX)[["category", "percent", "score"]].copy()


def satisfaction_factors_table() -> pd.DataFrame:
    """Study-reported factors rated as affecting satisfaction 'a lot'."""
    return pd.DataFrame(SSCI_SATISFACTION_FACTORS).sort_values(
        "percent_affects_a_lot", ascending=False
    ).reset_index(drop=True)


def sample_reference_profiles(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample synthetic reference categories using published aggregate marginals.

    This is a synthetic re-sampling of rounded table percentages, not access to
    the original participant records. A latent logistic utility is sampled within
    each observed ordinal category so a hypothetical shift scenario can be shown.
    The latent utility itself is a mathematical device, not a measured variable.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    rows = SSCI_RECEIVING_ORAL_SEX
    keys = [row["key"] for row in rows]
    probs = np.asarray([row["percent"] for row in rows], dtype=float) / 100.0
    if not np.isclose(probs.sum(), 1.0, atol=1e-9):
        raise RuntimeError("empirical reference percentages must sum to 100%")

    category = rng.choice(keys, size=n, p=probs)
    latent = np.full(n, np.nan, dtype=float)

    # Conditional ordinal distribution among those who report practising it.
    practiced_rows = rows[1:]
    practiced_probs = np.asarray([r["percent"] for r in practiced_rows], dtype=float)
    practiced_probs = practiced_probs / practiced_probs.sum()
    cum = np.cumsum(practiced_probs)
    lower = np.concatenate(([0.0], cum[:-1]))

    eps = np.finfo(float).eps * 10
    for i, row in enumerate(practiced_rows):
        mask = category == row["key"]
        count = int(mask.sum())
        if count:
            u = rng.uniform(lower[i] + eps, cum[i] - eps, size=count)
            latent[mask] = np.log(u / (1.0 - u))

    return pd.DataFrame(
        {
            "empirical_reference_category": category,
            "empirical_reference_latent_utility": latent,
        }
    )


def research_metadata() -> dict[str, Any]:
    """Machine-readable provenance for experiment settings and JSON exports."""
    return {
        "reference_distribution_source_id": MODEL_REFERENCE_ID,
        "reference_distribution_source": RESEARCH_STUDIES[0],
        "reference_distribution": SSCI_RECEIVING_ORAL_SEX,
        "reported_satisfaction_factors": SSCI_SATISFACTION_FACTORS,
        "other_studies": RESEARCH_STUDIES[1:],
        "data_handling": (
            "Only rounded published aggregate percentages are used. No raw or participant-level "
            "records are included. The synthetic reference cohort is not the original sample."
        ),
    }
