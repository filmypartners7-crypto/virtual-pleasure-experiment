"""Rule-based guide parser.

Turns pasted guide text into a sequence of structured steps. This is a
deterministic keyword parser, **not** a language model: it finds technique
phrases, estimates durations and intensity modifiers, and reports what it could
not classify. That is why the app makes you review and correct every step before
running an experiment - the parser is a first pass, and you are the authority on
what your guide means.

Nothing here is hidden: :func:`parse_guide` returns, for every step, the exact
source text it matched and the alternatives it rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .techniques import TECHNIQUES, TECHNIQUE_INDEX, Technique

DEFAULT_STEP_MINUTES = 2.0
MAX_STEP_MINUTES = 15.0
MAX_TOTAL_MINUTES = 45.0

# Phrases that appear under more than one technique; resolved by context words.
AMBIGUOUS_ALIASES: dict[str, tuple[str, ...]] = {
    "labia": ("outer_labia", "inner_labia"),
}

CONTEXT_HINTS: dict[str, tuple[str, ...]] = {
    "outer_labia": ("outer", "majora", "outside", "lips on the outside"),
    "inner_labia": ("inner", "minora", "inside", "between"),
}

INTENSITY_WORDS: tuple[tuple[tuple[str, ...], float], ...] = (
    (("very gentle", "feather", "barely", "whisper"), -0.20),
    (("gentle", "light", "soft", "slowly", "slow", "delicate", "lightly"), -0.12),
    (("firm", "firmer", "harder", "pressure", "press", "deeper"), 0.12),
    (("fast", "faster", "rapid", "quick", "vigorous", "intense"), 0.18),
)

NUMBER_WORDS: dict[str, float] = {
    "a": 1.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "ten": 10.0,
    "half": 0.5,
    "few": 3.0,
    "several": 4.0,
    "couple": 2.0,
}

# A block starts a new step when it begins with a list marker or a heading-like line.
BLOCK_SPLIT = re.compile(r"\n\s*\n")
LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d{1,2}[.)]|step\s*\d{1,2}\s*[:.\-]|phase\s*\d{1,2}\s*[:.\-])\s*", re.IGNORECASE)
DURATION_PATTERN = re.compile(
    r"(?<![a-z])(\d+(?:\.\d+)?|a|an|one|two|three|four|five|ten|half|few|several|couple)"
    r"\s+(second|sec|minute|min)s?\b"
    r"|(?<![a-z])(\d+)\s*(s|m)\b",
    re.IGNORECASE,
)

# A mention is discarded when it is negated: "not the glans", "never suction",
# "without direct contact". The window is deliberately short so that "no
# clitoral contact" does not also suppress a later, positive mention.
NEGATION_WORDS: tuple[str, ...] = (
    "not ", "never", "no ", "without", "avoid", "instead of", "rather than",
    "don't", "dont", "skip", "stop before", "before ", "not yet",
)
NEGATION_WINDOW = 26
MIN_STEP_MINUTES = 0.1


@dataclass
class ParsedStep:
    index: int
    technique_key: str
    label: str
    duration_min: float
    intensity: float
    source_text: str
    matched_phrase: str
    confidence: str                 # "high" | "medium" | "low"
    alternatives: tuple[str, ...] = ()
    duration_inferred: bool = True

    def as_row(self) -> dict[str, Any]:
        return {
            "step": self.index,
            "technique": self.label,
            "technique_key": self.technique_key,
            "minutes": round(self.duration_min, 1),
            "intensity": round(self.intensity, 2),
            "confidence": self.confidence,
            "matched_phrase": self.matched_phrase,
            "source_text": self.source_text,
        }


@dataclass
class ParseResult:
    steps: list[ParsedStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unclassified_blocks: list[str] = field(default_factory=list)
    total_minutes: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.steps)

    def frame(self) -> pd.DataFrame:
        if not self.steps:
            return pd.DataFrame(
                columns=[
                    "step", "technique", "technique_key", "minutes",
                    "intensity", "confidence", "matched_phrase", "source_text",
                ]
            )
        return pd.DataFrame([step.as_row() for step in self.steps])

    def summary_text(self) -> str:
        recognised = sum(1 for step in self.steps if step.confidence != "low")
        return (
            f"{len(self.steps)} steps detected, {recognised} with high or medium confidence, "
            f"{len(self.unclassified_blocks)} block(s) not recognised, "
            f"{self.total_minutes:.1f} minutes total."
        )


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

def _alias_index() -> list[tuple[str, str]]:
    """(alias, technique_key) pairs, longest alias first so specific phrases win."""
    pairs: list[tuple[str, str]] = []
    for technique in TECHNIQUES:
        for alias in technique.aliases:
            pairs.append((alias.lower(), technique.key))
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return pairs


_ALIASES = _alias_index()


def split_blocks(text: str) -> list[str]:
    """Split guide text into step-sized blocks.

    Blank lines separate blocks; a numbered or bulleted line always starts a new
    block. Long unbroken prose is also split on sentence boundaries so a
    single-paragraph guide still yields steps.
    """
    if not text or not text.strip():
        return []

    blocks: list[str] = []
    for chunk in BLOCK_SPLIT.split(text.replace("\r\n", "\n")):
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = [line for line in chunk.split("\n") if line.strip()]
        if len(lines) > 1 and any(LIST_MARKER.match(line) for line in lines):
            for line in lines:
                cleaned = LIST_MARKER.sub("", line).strip()
                if cleaned:
                    blocks.append(cleaned)
        elif len(chunk) > 400:
            for sentence_group in re.split(r"(?<=[.!?])\s+(?=[A-Z])", chunk):
                if sentence_group.strip():
                    blocks.append(sentence_group.strip())
        else:
            blocks.append(LIST_MARKER.sub("", chunk).strip())
    return [block for block in blocks if block]


def _resolve_ambiguous(alias: str, context: str) -> tuple[str, str]:
    """Resolve a shared alias using nearby context words. Returns (key, confidence)."""
    candidates = AMBIGUOUS_ALIASES[alias]
    for candidate in candidates:
        hints = CONTEXT_HINTS.get(candidate, ())
        if any(hint in context for hint in hints):
            return candidate, "high"
    return candidates[0], "low"


def find_techniques(text: str) -> list[tuple[int, str, str, str]]:
    """Locate technique mentions in order.

    Returns ``(position, technique_key, matched_alias, confidence)`` tuples, with
    overlapping shorter aliases discarded so "clitoral hood" is not also read as
    "clitoris".
    """
    lowered = text.lower()
    raw: list[tuple[int, int, str, str]] = []
    for alias, key in _ALIASES:
        for match in re.finditer(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lowered):
            raw.append((match.start(), match.end(), alias, key))

    raw.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    kept: list[tuple[int, int, str, str]] = []
    for start, end, alias, key in raw:
        if any(not (end <= s or start >= e) for s, e, _, _ in kept):
            continue
        kept.append((start, end, alias, key))
    kept.sort(key=lambda item: item[0])

    resolved: list[tuple[int, str, str, str]] = []
    for start, _end, alias, key in kept:
        window = lowered[max(0, start - NEGATION_WINDOW):start]
        if any(word in window for word in NEGATION_WORDS):
            continue                     # e.g. "not the glans", "never suction"
        confidence = "high"
        if alias in AMBIGUOUS_ALIASES:
            key, confidence = _resolve_ambiguous(alias, lowered)
        resolved.append((start, key, alias, confidence))

    # keep one step per technique per block, in order of first appearance
    seen: set[str] = set()
    deduped: list[tuple[int, str, str, str]] = []
    for item in resolved:
        if item[1] in seen:
            continue
        seen.add(item[1])
        deduped.append(item)
    return deduped


def extract_duration_minutes(text: str) -> tuple[float | None, bool]:
    """Read an explicit duration. Returns (minutes or None, was_explicit)."""
    match = DURATION_PATTERN.search(text.lower())
    if not match:
        return None, False
    number_token = (match.group(1) or match.group(3) or "").lower()
    unit = (match.group(2) or match.group(4) or "").lower()
    try:
        amount = float(number_token)
    except ValueError:
        amount = NUMBER_WORDS.get(number_token, 1.0)
    if unit in {"second", "sec", "s"}:
        minutes = amount / 60.0
    else:
        minutes = amount
    return max(minutes, 1.0 / 60.0), True


def extract_intensity_modifier(text: str) -> float:
    """Nudge the technique's default intensity by the words the guide uses."""
    lowered = text.lower()
    delta = 0.0
    for words, adjustment in INTENSITY_WORDS:
        if any(word in lowered for word in words):
            delta += adjustment
    return max(-0.3, min(0.3, delta))


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def parse_guide(text: str, default_minutes: float = DEFAULT_STEP_MINUTES) -> ParseResult:
    """Parse pasted guide text into reviewable steps.

    The result is a *proposal*. Every step keeps its source text so the user can
    see exactly what was matched, and the app requires confirmation before any
    experiment runs.
    """
    result = ParseResult()
    blocks = split_blocks(text)
    if not blocks:
        result.warnings.append("No text supplied, or the text contained no recognisable steps.")
        return result

    index = 0
    for block in blocks:
        mentions = find_techniques(block)
        if not mentions:
            result.unclassified_blocks.append(block[:180])
            result.warnings.append(
                "Could not classify this block - map it to a technique manually: "
                f"\"{block[:80]}...\""
            )
            continue

        explicit_duration, was_explicit = extract_duration_minutes(block)
        intensity_delta = extract_intensity_modifier(block)

        # A block naming several techniques becomes several steps, in order.
        share = max(
            MIN_STEP_MINUTES,
            (explicit_duration if explicit_duration else default_minutes) / len(mentions),
        )
        for position, (_, key, alias, confidence) in enumerate(mentions):
            technique: Technique = TECHNIQUE_INDEX[key]
            index += 1
            intensity = 0.0 if technique.verbal else max(
                0.0, min(1.0, technique.intensity + intensity_delta)
            )
            alternatives = tuple(
                candidate
                for candidate in AMBIGUOUS_ALIASES.get(alias, ())
                if candidate != key
            )
            if len(mentions) > 1 and not was_explicit:
                confidence = "medium" if confidence == "high" else confidence
            result.steps.append(
                ParsedStep(
                    index=index,
                    technique_key=key,
                    label=technique.label,
                    duration_min=min(share, MAX_STEP_MINUTES),
                    intensity=intensity,
                    source_text=block[:300],
                    matched_phrase=alias,
                    confidence=confidence,
                    alternatives=alternatives,
                    duration_inferred=not was_explicit,
                )
            )

    total = sum(step.duration_min for step in result.steps)
    if total > MAX_TOTAL_MINUTES:
        scale = MAX_TOTAL_MINUTES / total
        for step in result.steps:
            step.duration_min = max(1.0 / 60.0, step.duration_min * scale)
        result.warnings.append(
            f"Total duration exceeded {MAX_TOTAL_MINUTES:.0f} minutes and was scaled down to fit."
        )
    result.total_minutes = sum(step.duration_min for step in result.steps)

    if any(step.duration_inferred for step in result.steps):
        result.warnings.append(
            "Some durations were not stated in your guide; a default of "
            f"{default_minutes:.0f} minute(s) per step was assumed. Edit them below."
        )
    if any(step.alternatives for step in result.steps):
        result.warnings.append(
            "Some phrases were ambiguous (for example a bare 'labia'). Check the technique "
            "assignment for those steps."
        )
    return result


def sequence_signature(steps: list[ParsedStep]) -> str:
    """Short readable signature of the parsed sequence, e.g. 'outer_labia>hood>glans'."""
    return ">".join(step.technique_key for step in steps)
