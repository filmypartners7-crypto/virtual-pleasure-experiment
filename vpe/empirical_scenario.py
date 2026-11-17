"""Ordinal satisfaction scenario anchored to a published aggregate distribution.

The starting distribution is resampled from reported receiving-cunnilingus
satisfaction percentages in a published Spanish women's survey. The guide's
impact is represented by a user-selected hypothetical logit shift. No published
study estimates that shift for a guide like the one entered here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .protocol import Protocol, PROTOCOL_VARIABLES
from .research import SSCI_RECEIVING_ORAL_SEX

CATEGORY_LABELS = {
    "not_practiced": "Not practiced",
    "not_satisfactory": "Not satisfactory at all",
    "somewhat_satisfactory": "Somewhat satisfactory",
    "quite_satisfactory": "Quite satisfactory",
    "very_satisfactory": "Very satisfactory",
}
CATEGORY_SCORES = {
    "not_practiced": np.nan,
    "not_satisfactory": 1.0,
    "somewhat_satisfactory": 2.0,
    "quite_satisfactory": 3.0,
    "very_satisfactory": 4.0,
}
PRACTICED_KEYS = [
    "not_satisfactory",
    "somewhat_satisfactory",
    "quite_satisfactory",
    "very_satisfactory",
]


def protocol_quality_score(protocol: Protocol) -> float:
    """Equal-weight mean of seven sliders; a transparent design convention."""
    return float(np.mean([getattr(protocol, name) for name in PROTOCOL_VARIABLES]))


def _ordinal_category_from_utility(utility: np.ndarray) -> np.ndarray:
    """Map logistic latent utility to categories using source-derived cut-points."""
    percentages = np.asarray([row["percent"] for row in SSCI_RECEIVING_ORAL_SEX[1:]], dtype=float)
    probs = percentages / percentages.sum()
    thresholds = np.log(np.cumsum(probs[:-1]) / (1.0 - np.cumsum(probs[:-1])))
    indices = np.searchsorted(thresholds, utility, side="right")
    return np.asarray(PRACTICED_KEYS, dtype=object)[indices]


def run_empirical_satisfaction_scenario(
    profiles: pd.DataFrame,
    guide_protocol: Protocol,
    baseline_protocol: Protocol,
    assumed_effect_logit: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Apply a hypothetical shift to an empirically anchored synthetic cohort.

    `assumed_effect_logit` is an explicit scenario assumption: change in latent
    log-odds per full 0-to-1 increase in the equal-weight protocol score. Zero is
    the null-effect scenario. It is not estimated from the source survey.
    """
    effect = float(assumed_effect_logit)
    if not np.isfinite(effect) or effect < -3.0 or effect > 3.0:
        raise ValueError("assumed_effect_logit must be finite and lie in [-3, 3]")

    required = {"couple_id", "empirical_reference_category", "empirical_reference_latent_utility"}
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"profiles missing empirical reference fields: {sorted(missing)}")

    base_score = protocol_quality_score(baseline_protocol)
    guide_score = protocol_quality_score(guide_protocol)
    score_difference = guide_score - base_score
    latent_shift = effect * score_difference

    categories = profiles["empirical_reference_category"].astype(str).to_numpy()
    latent = profiles["empirical_reference_latent_utility"].to_numpy(dtype=float)
    practiced = categories != "not_practiced"

    guide_categories = categories.copy()
    guide_categories[practiced] = _ordinal_category_from_utility(latent[practiced] + latent_shift)

    baseline_scores = np.asarray([CATEGORY_SCORES[x] for x in categories], dtype=float)
    guide_scores = np.asarray([CATEGORY_SCORES[x] for x in guide_categories], dtype=float)

    frame = pd.DataFrame(
        {
            "couple_id": profiles["couple_id"].to_numpy(),
            "baseline_category": categories,
            "baseline_category_label": [CATEGORY_LABELS[x] for x in categories],
            "guide_scenario_category": guide_categories,
            "guide_scenario_category_label": [CATEGORY_LABELS[x] for x in guide_categories],
            "baseline_ordinal_score": baseline_scores,
            "guide_scenario_ordinal_score": guide_scores,
            "ordinal_score_change": guide_scores - baseline_scores,
            "practiced_in_reference": practiced,
        }
    )

    changed = frame.loc[practiced, "ordinal_score_change"]
    metadata: dict[str, float | str] = {
        "baseline_protocol_quality_score": base_score,
        "guide_protocol_quality_score": guide_score,
        "protocol_score_difference": score_difference,
        "assumed_effect_logit_per_full_score_unit": effect,
        "applied_latent_logit_shift": latent_shift,
        "effect_formula": "latent_shift = assumed_effect_logit * (guide_mean_slider_score - baseline_mean_slider_score)",
        "protocol_score_formula": "equal-weight arithmetic mean of the seven 0-1 protocol sliders",
        "assumption_warning": (
            "The effect coefficient and equal-weight protocol score are hypothetical. "
            "The published survey supplies only the starting category distribution; it does not estimate guide effects."
        ),
        "practiced_count": int(practiced.sum()),
        "not_practiced_count": int((~practiced).sum()),
        "moved_up_count": int((changed > 0).sum()),
        "moved_down_count": int((changed < 0).sum()),
        "unchanged_count": int((changed == 0).sum()),
    }
    return frame, metadata


def empirical_scenario_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare baseline reference and guide scenario category shares."""
    order = [row["key"] for row in SSCI_RECEIVING_ORAL_SEX]
    labels = {row["key"]: row["category"] for row in SSCI_RECEIVING_ORAL_SEX}
    n = max(len(frame), 1)
    rows = []
    for key in order:
        baseline_pct = 100.0 * float((frame["baseline_category"] == key).sum()) / n
        guide_pct = 100.0 * float((frame["guide_scenario_category"] == key).sum()) / n
        rows.append(
            {
                "category_key": key,
                "category": labels[key],
                "baseline_reference_percent": baseline_pct,
                "guide_scenario_percent": guide_pct,
            }
        )
    return pd.DataFrame(rows)
