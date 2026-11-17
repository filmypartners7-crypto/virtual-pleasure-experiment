"""Research and evidence database for VPE 0.3.0.

Every model parameter in this project points at an entry here, and every entry
states exactly what the source does and does not support. Two rules:

1.  **No invented citations.** A study appears here only if its bibliographic
    details were checked against the published record. Each carries
    ``citation_verified`` and the date it was checked.
2.  **No invented coefficients.** Where no study estimates a quantity, the
    parameter is labelled ``hypothetical_assumption`` or
    ``unvalidated_relationship`` and says so in the app.

The honest summary for this domain: **no published study estimates the causal
effect of specific cunnilingus techniques on pleasure.** What exists is (a)
aggregate survey marginals for satisfaction and orgasm by activity, (b) evidence
that preference heterogeneity is large, and (c) anatomy/physiology. This project
therefore calibrates its *population and null scenario* to real marginals and
labels every technique effect as an explicit, inspectable assumption. It never
presents a technique effect as a research finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

MODEL_VERSION = "0.3.0"
CITATION_CHECK_DATE = "2026-11-17"


# --------------------------------------------------------------------------- #
# Evidence levels
# --------------------------------------------------------------------------- #

EVIDENCE_LEVELS: dict[str, dict[str, str]] = {
    "published_aggregate": {
        "code": "A",
        "label": "Published aggregate data",
        "meaning": (
            "Rounded aggregate statistics reported in a peer-reviewed source, used to "
            "calibrate a population or a null scenario. Not participant-level data."
        ),
    },
    "research_supported_concept": {
        "code": "B",
        "label": "Research-supported concept",
        "meaning": (
            "A qualitative claim that the literature supports (for example: preference "
            "heterogeneity is large). The numeric value in this model is still a choice."
        ),
    },
    "research_informed_range": {
        "code": "C",
        "label": "Research-informed range",
        "meaning": (
            "The plausible range is bounded by what sources report, but the point value "
            "was not estimated from data."
        ),
    },
    "hypothetical_assumption": {
        "code": "D",
        "label": "Hypothetical assumption",
        "meaning": "Chosen so the model behaves sensibly. No empirical source.",
    },
    "unvalidated_relationship": {
        "code": "E",
        "label": "Unvalidated model relationship",
        "meaning": (
            "A structural link between variables invented for this model. Never tested. "
            "This is the status of every technique-effect parameter here."
        ),
    },
}


# --------------------------------------------------------------------------- #
# Studies
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Study:
    id: str
    authors: str
    year: int
    title: str
    journal: str
    doi: str
    url: str
    sample: str
    findings: tuple[str, ...]
    supports: tuple[str, ...]      # what VPE legitimately uses it for
    does_not_support: tuple[str, ...]
    limitations: str
    citation_verified: bool = True
    verified_on: str = CITATION_CHECK_DATE

    def as_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "authors": self.authors,
            "year": self.year,
            "title": self.title,
            "journal": self.journal,
            "doi": self.doi,
            "sample": self.sample,
            "findings": " | ".join(self.findings),
            "supports": " | ".join(self.supports),
            "does_not_support": " | ".join(self.does_not_support),
            "limitations": self.limitations,
            "citation_verified": self.citation_verified,
            "verified_on": self.verified_on,
        }


STUDIES: tuple[Study, ...] = (
    Study(
        id="ogallar_blanco_2022",
        authors="Ogallar-Blanco, Lara-Moreno & Godoy-Izquierdo",
        year=2022,
        title=(
            "Going beyond 'With a Partner' and 'Intercourse': Does Anything Else Influence "
            "Sexual Satisfaction among Women? The Sexual Satisfaction Comprehensive Index"
        ),
        journal="International Journal of Environmental Research and Public Health, 19(16), 10232",
        doi="10.3390/ijerph191610232",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9408228/",
        sample="N=1,080 women aged 18-50 in Spain; voluntary online non-probability sample",
        findings=(
            "Satisfaction with receiving oral sex (cunnilingus): 5.3% not practised, "
            "3.1% not satisfactory at all, 10.5% somewhat, 21.9% quite, 59.2% very satisfactory "
            "(Table 4); mean 3.27 +/- 1.11 on a 0-4 scale",
            "Factors rated as affecting sexual satisfaction 'a lot' include partner attraction "
            "72.5%, not feeling obliged to do unwanted activity 72.1%, confidence with partner "
            "66.8%, comfort with own body 63.3%, partner's techniques 53.7%, feeling relaxed "
            "51.5%, communication about likes and boundaries 39.3% (Table 3)",
        ),
        supports=(
            "Population calibration target: the model's null scenario reproduces the reported "
            "satisfaction marginals for receiving oral sex",
            "Support for modelling comfort, obligation-freedom and communication as separate "
            "contributors (perceived influence, not causal effects)",
        ),
        does_not_support=(
            "Any effect of a specific technique or sequence",
            "Causal inference of any kind - the factor ratings are perceived influence",
            "Prevalence in any population outside the sample",
        ),
        limitations=(
            "Self-selected Spanish sample; cross-sectional self-report; not globally "
            "representative; did not test any guide."
        ),
    ),
    Study(
        id="richters_2006",
        authors="Richters, de Visser, Rissel & Smith",
        year=2006,
        title="Sexual practices at last heterosexual encounter and occurrence of orgasm in a national survey",
        journal="The Journal of Sex Research, 43(3), 217-226",
        doi="10.1080/00224490609552320",
        url="https://pubmed.ncbi.nlm.nih.gov/17599244/",
        sample="N=19,307 Australians aged 16-59; national telephone probability survey (ASHR 2001-02)",
        findings=(
            "Women reported orgasm in 69% of most recent heterosexual encounters; men in 95%",
            "Women were more likely to reach orgasm in encounters including more practices, "
            "especially cunnilingus",
        ),
        supports=(
            "That the *set* of practices performed, not only demographics, is associated with "
            "orgasm - the premise for modelling protocols rather than fixed individual traits",
            "Direction of a practice-inclusion effect on orgasm (association, not causation)",
        ),
        does_not_support=(
            "Any estimate of a specific technique's effect on pleasure",
            "Any claim that adding a practice causes orgasm",
            "Sequence or ordering effects - the survey measured sets, not order",
        ),
        limitations=(
            "Cross-sectional self-report of the last encounter; orgasm reported, not observed; "
            "heterosexual encounters only; 2001-02 Australian sample."
        ),
    ),
    Study(
        id="herbenick_2018",
        authors="Herbenick, Fu, Arter, Sanders & Dodge",
        year=2018,
        title=(
            "Women's Experiences With Genital Touching, Sexual Pleasure, and Orgasm: Results "
            "From a U.S. Probability Sample of Women Ages 18 to 94"
        ),
        journal="Journal of Sex & Marital Therapy, 44(2), 201-212",
        doi="10.1080/0092623X.2017.1346530",
        url="https://pubmed.ncbi.nlm.nih.gov/28678639/",
        sample="N=1,055 women aged 18-94; U.S. probability-based panel (KnowledgePanel)",
        findings=(
            "During intercourse, 36.6% reported needing clitoral stimulation to orgasm; 36.0% "
            "said it was not necessary but enhanced orgasm; 18.4% reported penetration alone "
            "was sufficient; 7.5% did not orgasm during intercourse",
            "Genital-touch preferences varied substantially across respondents",
        ),
        supports=(
            "Preference heterogeneity as a modelled feature - the single most important "
            "structural assumption in this project",
            "That direct clitoral stimulation is necessary or enhancing for a large majority "
            "(an aggregate, not a technique effect)",
        ),
        does_not_support=(
            "Any specific cunnilingus technique, pressure, rhythm or sequence",
            "The proportion of women who require a particular approach",
        ),
        limitations=(
            "U.S. probability panel, self-report, cross-sectional; figures concern orgasm during "
            "intercourse, not cunnilingus technique."
        ),
    ),
    Study(
        id="frederick_2018",
        authors="Frederick, St. John, Garcia & Lloyd",
        year=2018,
        title=(
            "Differences in Orgasm Frequency Among Gay, Lesbian, Bisexual, and Heterosexual Men "
            "and Women in a U.S. National Sample"
        ),
        journal="Archives of Sexual Behavior, 47(1), 273-288",
        doi="10.1007/s10508-017-0939-z",
        url="https://doi.org/10.1007/s10508-017-0939-z",
        sample="N=52,588 U.S. adults; national probability sample",
        findings=(
            "Large orgasm-frequency differences by partner gender and by which practices were "
            "included; practices that provide direct clitoral stimulation are associated with "
            "higher orgasm frequency for women",
        ),
        supports=(
            "Preference and practice heterogeneity at scale",
            "That practice inclusion, not individual deficit, explains much of the variation",
        ),
        does_not_support=(
            "Technique-level or sequence-level effects",
            "Causal claims - the design is cross-sectional",
        ),
        limitations="Cross-sectional self-report; orgasm frequency, not pleasure or comfort.",
    ),
    Study(
        id="backstrom_2012",
        authors="Backstrom, Armstrong & Puentes",
        year=2012,
        title="Women's Negotiation of Cunnilingus in College Hookups and Relationships",
        journal="Journal of Sex Research, 49(1), 1-12",
        doi="10.1080/00224499.2011.585523",
        url="https://pubmed.ncbi.nlm.nih.gov/22010825/",
        sample="In-depth interviews with 43 college women",
        findings=(
            "Attitudes to and negotiation of cunnilingus differed by relationship context and "
            "individual preference",
        ),
        supports=(
            "Modelling consent/negotiation and relationship context as distinct from technique",
            "Support for the 'not feeling obliged' mechanism",
        ),
        does_not_support=(
            "Any prevalence estimate or numeric effect size",
            "Any technique or protocol effect",
        ),
        limitations="Small qualitative sample; not generalisable numerically.",
    ),
    Study(
        id="rosen_2000_fsfi",
        authors="Rosen, Brown, Heiman, Leiblum, Meston, Shabsigh, Ferguson & D'Agostino",
        year=2000,
        title=(
            "The Female Sexual Function Index (FSFI): A Multidimensional Self-Report Instrument "
            "for the Assessment of Female Sexual Function"
        ),
        journal="Journal of Sex & Marital Therapy, 26(2), 191-208",
        doi="10.1080/009262300278597",
        url="https://pubmed.ncbi.nlm.nih.gov/10782451/",
        sample="Validation study: 131 controls and 128 age-matched participants with female sexual arousal disorder",
        findings=(
            "Six separable domains: desire, subjective arousal, lubrication, orgasm, "
            "satisfaction and pain",
        ),
        supports=(
            "Keeping arousal, comfort/pain, satisfaction and orgasm as separate outcomes "
            "rather than one composite 'pleasure' score",
        ),
        does_not_support=(
            "Any coefficient in this model",
            "Any technique or protocol effect",
        ),
        limitations="A measurement instrument, not a study of technique effectiveness.",
    ),
)

STUDY_INDEX: dict[str, Study] = {study.id: study for study in STUDIES}


# --------------------------------------------------------------------------- #
# Parameters -> evidence
# --------------------------------------------------------------------------- #

# Global model constants and the evidence status of each. Technique-level
# parameters are declared in techniques.py and are all level E by construction.
PARAMETER_EVIDENCE: dict[str, dict[str, str]] = {
    "population_class_weights": {
        "evidence": "hypothetical_assumption",
        "study": "",
        "note": "Latent preference archetypes are a modelling device. Their weights are not prevalence.",
    },
    "preference_heterogeneity": {
        "evidence": "research_supported_concept",
        "study": "herbenick_2018",
        "note": "That heterogeneity is large is supported; the specific spread here is a choice.",
    },
    "separate_outcomes": {
        "evidence": "research_supported_concept",
        "study": "rosen_2000_fsfi",
        "note": "Arousal, comfort, satisfaction and orgasm are modelled separately.",
    },
    "satisfaction_calibration_target": {
        "evidence": "published_aggregate",
        "study": "ogallar_blanco_2022",
        "note": "The null scenario is fitted to reproduce these marginals.",
    },
    "practice_inclusion_effect": {
        "evidence": "research_supported_concept",
        "study": "richters_2006",
        "note": "Association only; the model's magnitude is not estimated from it.",
    },
    "obligation_and_consent": {
        "evidence": "research_supported_concept",
        "study": "backstrom_2012",
        "note": "Consent/negotiation modelled separately from technique.",
    },
    "technique_effects": {
        "evidence": "unvalidated_relationship",
        "study": "",
        "note": (
            "No published study estimates the effect of specific cunnilingus techniques. "
            "Every technique parameter in techniques.py is an explicit assumption."
        ),
    },
    "sequence_and_ordering": {
        "evidence": "unvalidated_relationship",
        "study": "",
        "note": (
            "Ordering effects (warm-up before direct clitoral contact) are anatomically "
            "plausible but untested as a model relationship."
        ),
    },
    "warmup_requirement": {
        "evidence": "hypothetical_assumption",
        "study": "",
        "note": "Minutes of indirect stimulation before direct glans contact is comfortable.",
    },
    "overstimulation_threshold": {
        "evidence": "hypothetical_assumption",
        "study": "",
        "note": "Point at which added intensity becomes aversive rather than pleasurable.",
    },
    "rhythm_dependence": {
        "evidence": "hypothetical_assumption",
        "study": "",
        "note": "Benefit from sustained, unchanging stimulation.",
    },
    "latent_trajectory_form": {
        "evidence": "unvalidated_relationship",
        "study": "",
        "note": "Arousal accumulating minute-by-minute is a modelling convenience.",
    },
}

# Calibration target: SSCI Table 4, receiving oral sex, satisfaction categories.
SSCI_SATISFACTION_CATEGORIES: tuple[tuple[str, str, float], ...] = (
    ("not_practiced", "Not practiced", 5.3),
    ("not_satisfactory", "Not satisfactory at all", 3.1),
    ("somewhat_satisfactory", "Somewhat satisfactory", 10.5),
    ("quite_satisfactory", "Quite satisfactory", 21.9),
    ("very_satisfactory", "Very satisfactory", 59.2),
)

# The four satisfaction levels conditional on practising (percentages renormalised
# from the table above). This is the calibration target for the null scenario.
SSCI_PRACTISED_BANDS: tuple[float, ...] = tuple(
    round(percent / 94.7 * 100.0, 2) for _, _, percent in SSCI_SATISFACTION_CATEGORIES[1:]
)

CALIBRATION_RESULT: dict[str, Any] = {
    "target_source": "ogallar_blanco_2022",
    "target_table": "Table 4, receiving oral sex (cunnilingus), N=1,080",
    "target_bands_percent": SSCI_PRACTISED_BANDS,
    "band_labels": ["not satisfactory", "somewhat", "quite", "very"],
    # Filled in by calibration.py at build time; see CALIBRATION_REPORT there.
    "achieved_bands_percent": None,
    "max_deviation_points": None,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def studies_frame() -> pd.DataFrame:
    """All studies as a display/export table."""
    return pd.DataFrame([study.as_row() for study in STUDIES])


def parameter_evidence_frame() -> pd.DataFrame:
    """Every global parameter with its evidence level spelled out."""
    rows = []
    for name, spec in PARAMETER_EVIDENCE.items():
        level = EVIDENCE_LEVELS[spec["evidence"]]
        study = STUDY_INDEX.get(spec["study"]) if spec["study"] else None
        rows.append(
            {
                "parameter": name,
                "evidence_code": level["code"],
                "evidence_status": level["label"],
                "source": f"{study.authors} ({study.year})" if study else "none",
                "note": spec["note"],
            }
        )
    return pd.DataFrame(rows)


def evidence_coverage() -> pd.DataFrame:
    """How many parameters sit at each evidence level."""
    counts: dict[str, int] = {key: 0 for key in EVIDENCE_LEVELS}
    for spec in PARAMETER_EVIDENCE.values():
        counts[spec["evidence"]] += 1
    return pd.DataFrame(
        [
            {
                "code": EVIDENCE_LEVELS[key]["code"],
                "evidence_status": EVIDENCE_LEVELS[key]["label"],
                "parameters": counts[key],
            }
            for key in EVIDENCE_LEVELS
        ]
    )


def studies_for(parameter: str) -> tuple[Study, ...]:
    """The studies cited for one parameter (empty tuple if none)."""
    spec = PARAMETER_EVIDENCE.get(parameter)
    if not spec or not spec["study"]:
        return ()
    return (STUDY_INDEX[spec["study"]],)


def unverified_citations() -> Iterable[Study]:
    """Any study whose citation has not been checked against the published record."""
    return (study for study in STUDIES if not study.citation_verified)


def evidence_summary_text() -> str:
    return (
        "Evidence position of VPE 0.3.0. Six peer-reviewed sources are used. They support: "
        "the aggregate satisfaction marginals used to calibrate the null scenario; the claim "
        "that preference heterogeneity is large; that arousal, comfort, satisfaction and "
        "orgasm are separable; and that including more practices, especially cunnilingus, is "
        "associated with orgasm. NO published study estimates the causal effect of specific "
        "cunnilingus techniques, pressures, rhythms or sequences. Every technique parameter in "
        "this model is therefore an explicit, labelled assumption, and the app states this on "
        "every results page. The simulation explores the consequences of stated assumptions; "
        "it does not discover facts about people."
    )
