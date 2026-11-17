"""Technique catalogue.

Every technique carries the parameters the sequence model needs, and every one of
those parameters is an **explicit assumption** (evidence level E): no published
study estimates the effect of a specific cunnilingus technique. The catalogue is
deliberately inspectable - the app shows this table, and you can argue with it.

Parameter meanings (all on 0-1 scales unless noted):

``directness``
    How directly the technique targets the clitoral glans. 0.0 = no clitoral
    contact at all, 1.0 = direct glans contact.
``intensity``
    Default intensity. A step can override it (a guide that says "firm" is more
    intense than one that says "light").
``warmup_need``
    How much arousal this technique requires before it is pleasant. Applied as a
    penalty when it is used before the woman's warm-up requirement is met.
``overstim_load``
    How much this technique loads the overstimulation channel - the mechanism by
    which "more" stops being better.
``rhythm_sensitivity``
    How much this technique rewards being held steady rather than varied.
``verbal``
    True for communication steps (check-ins, pauses). They consume time, raise
    comfort and lower discomfort, and do not add arousal directly.
``comfort_boost``
    How much a verbal step raises the comfort estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Technique:
    key: str
    label: str
    directness: float
    intensity: float
    warmup_need: float
    overstim_load: float
    rhythm_sensitivity: float
    verbal: bool
    comfort_boost: float
    aliases: tuple[str, ...]
    description: str
    evidence: str = "unvalidated_relationship"

    def as_row(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "technique": self.label,
            "directness": self.directness,
            "intensity": self.intensity,
            "warmup_need": self.warmup_need,
            "overstim_load": self.overstim_load,
            "rhythm_sensitivity": self.rhythm_sensitivity,
            "verbal": self.verbal,
            "description": self.description,
        }


TECHNIQUES: tuple[Technique, ...] = (
    Technique(
        key="thighs_kissing",
        label="Kissing thighs and inner legs",
        directness=0.05,
        intensity=0.20,
        warmup_need=0.00,
        overstim_load=0.00,
        rhythm_sensitivity=0.10,
        verbal=False,
        comfort_boost=0.0,
        aliases=("inner thigh", "thighs", "kiss her leg", "legs", "kissing her thighs"),
        description="Non-genital priming. Builds anticipation with no overstimulation risk.",
    ),
    Technique(
        key="outer_labia",
        label="Outer labia, broad strokes",
        directness=0.15,
        intensity=0.30,
        warmup_need=0.15,
        overstim_load=0.05,
        rhythm_sensitivity=0.30,
        verbal=False,
        comfort_boost=0.0,
        aliases=("outer labia", "labia majora", "outer lips", "labia", "vulva", "flat tongue over"),
        description="Broad, low-intensity warm-up over the labia majora.",
    ),
    Technique(
        key="inner_labia",
        label="Inner labia",
        directness=0.35,
        intensity=0.45,
        warmup_need=0.30,
        overstim_load=0.15,
        rhythm_sensitivity=0.40,
        verbal=False,
        comfort_boost=0.0,
        aliases=("inner labia", "labia minora", "inner lips", "labia"),
        description="Tracing the labia minora; a step up in directness from the outer labia.",
    ),
    Technique(
        key="vaginal_opening",
        label="Vaginal opening",
        directness=0.30,
        intensity=0.50,
        warmup_need=0.30,
        overstim_load=0.20,
        rhythm_sensitivity=0.40,
        verbal=False,
        comfort_boost=0.0,
        aliases=("vaginal opening", "vagina", "entrance", "opening", "introitus"),
        description="Contact at the vaginal opening, often with the tip of the tongue.",
    ),
    Technique(
        key="perineum",
        label="Perineum",
        directness=0.25,
        intensity=0.45,
        warmup_need=0.20,
        overstim_load=0.10,
        rhythm_sensitivity=0.35,
        verbal=False,
        comfort_boost=0.0,
        aliases=("perineum", "between vagina and anus", "taint"),
        description="The area between vagina and anus; highly responsive for some, unwanted for others.",
    ),
    Technique(
        key="clitoral_hood",
        label="Clitoral hood (indirect)",
        directness=0.60,
        intensity=0.45,
        warmup_need=0.50,
        overstim_load=0.25,
        rhythm_sensitivity=0.60,
        verbal=False,
        comfort_boost=0.0,
        aliases=("clitoral hood", "hood", "prepuce", "indirect", "around the clitoris", "over the clitoris"),
        description="Indirect clitoral work through the hood. The standard bridge before direct contact.",
    ),
    Technique(
        key="clitoral_glans",
        label="Clitoral glans (direct)",
        directness=0.95,
        intensity=0.65,
        warmup_need=0.85,
        overstim_load=0.60,
        rhythm_sensitivity=0.80,
        verbal=False,
        comfort_boost=0.0,
        aliases=("glans", "direct clitoral", "directly on the clitoris", "the clitoris", "clit", "lick the clitoris"),
        description="Direct glans contact. Highest payoff and highest overstimulation risk.",
    ),
    Technique(
        key="flat_tongue",
        label="Flat tongue pressure",
        directness=0.80,
        intensity=0.55,
        warmup_need=0.60,
        overstim_load=0.30,
        rhythm_sensitivity=0.70,
        verbal=False,
        comfort_boost=0.0,
        aliases=("flat tongue", "broad tongue", "wide tongue", "flat of the tongue"),
        description="Broad, diffuse pressure over the glans and hood - usually gentler than a point.",
    ),
    Technique(
        key="pointed_tongue",
        label="Pointed tongue strokes",
        directness=0.85,
        intensity=0.60,
        warmup_need=0.70,
        overstim_load=0.45,
        rhythm_sensitivity=0.65,
        verbal=False,
        comfort_boost=0.0,
        aliases=("tip of the tongue", "pointed tongue", "tongue tip", "flick", "circles", "side to side"),
        description="Precise, concentrated stimulation. Effective for some, too sharp for others.",
    ),
    Technique(
        key="suction",
        label="Gentle suction",
        directness=0.90,
        intensity=0.80,
        warmup_need=0.80,
        overstim_load=0.75,
        rhythm_sensitivity=0.75,
        verbal=False,
        comfort_boost=0.0,
        aliases=("suction", "suck", "sucking", "pull", "draw"),
        description="Gentle suction on the glans or hood. High intensity; an escalation, not an opener.",
    ),
    Technique(
        key="fingers_external",
        label="Fingers, external pressure",
        directness=0.55,
        intensity=0.60,
        warmup_need=0.40,
        overstim_load=0.30,
        rhythm_sensitivity=0.55,
        verbal=False,
        comfort_boost=0.0,
        aliases=("external pressure", "above the pubic bone", "press down", "fingers on the clitoris", "two fingers"),
        description="Manual pressure alongside the mouth; also a way to spare the jaw.",
    ),
    Technique(
        key="fingers_internal",
        label="Fingers, internal",
        directness=0.40,
        intensity=0.65,
        warmup_need=0.40,
        overstim_load=0.35,
        rhythm_sensitivity=0.50,
        verbal=False,
        comfort_boost=0.0,
        aliases=("fingers inside", "internal", "penetration", "one finger", "two fingers inside", "g-spot"),
        description="Internal stimulation, only where welcomed. Requires lubrication and consent.",
    ),
    Technique(
        key="check_in",
        label="Check in verbally",
        directness=0.00,
        intensity=0.00,
        warmup_need=0.00,
        overstim_load=0.00,
        rhythm_sensitivity=0.00,
        verbal=True,
        comfort_boost=1.0,
        aliases=("check in", "ask", "more, less", "how does that feel", "communicate", "feedback", "tell me"),
        description="A short verbal check. Costs time, raises comfort, lowers discomfort risk.",
    ),
    Technique(
        key="pause",
        label="Pause / hold still",
        directness=0.00,
        intensity=0.00,
        warmup_need=0.00,
        overstim_load=0.00,
        rhythm_sensitivity=0.00,
        verbal=True,
        comfort_boost=0.5,
        aliases=("pause", "stop", "hold still", "wait", "boundary", "consent"),
        description="An explicit pause or stopping rule. Raises comfort, never adds arousal.",
    ),
)

TECHNIQUE_INDEX: dict[str, Technique] = {technique.key: technique for technique in TECHNIQUES}

VERBAL_KEYS: tuple[str, ...] = tuple(t.key for t in TECHNIQUES if t.verbal)
STIMULATION_KEYS: tuple[str, ...] = tuple(t.key for t in TECHNIQUES if not t.verbal)


def get(key: str) -> Technique:
    if key not in TECHNIQUE_INDEX:
        raise ValueError(f"unknown technique: {key}")
    return TECHNIQUE_INDEX[key]


def techniques_frame() -> pd.DataFrame:
    return pd.DataFrame([technique.as_row() for technique in TECHNIQUES])


def aliases_frame() -> pd.DataFrame:
    rows = []
    for technique in TECHNIQUES:
        for alias in technique.aliases:
            rows.append({"technique": technique.label, "key": technique.key, "phrase": alias})
    return pd.DataFrame(rows)
