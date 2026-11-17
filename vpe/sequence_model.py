"""The sequence model: a minute-by-minute simulation of one protocol.

Why this shape. Ordering, ablation and substitution only mean something if the
model tracks *time*. So each woman's arousal is integrated minute by minute
against whatever technique is being applied at that moment, and four mechanisms
do the work:

1.  **Fit.** A technique is pleasant in proportion to how close its intensity and
    directness are to that woman's preference, with a tolerance band. Same
    technique, different woman, different result.
2.  **Warm-up.** Direct clitoral contact applied before a woman's warm-up
    requirement is met is penalised and adds discomfort. This is the mechanism
    behind "do not open with suction on the glans".
3.  **Overstimulation.** Every technique loads an overstimulation channel; past
    her threshold, added intensity subtracts from arousal and adds discomfort.
    This is why "more" stops working.
4.  **Rhythm.** Sustained, unchanging stimulation accumulates a plateau bonus
    proportional to her rhythm dependence; switching technique resets it.

Everything is bounded: outcomes are logistic functions of accumulated state, so
they lie strictly in (0, 1) and are unitless indices.

Evidence status: mechanisms 1-4 are anatomically plausible and consistent with
the sources in ``evidence.py``, but **no published study estimates any of these
parameters for cunnilingus**. All coefficients here are explicit assumptions
(level E), and the sensitivity analysis exists so you can see how much the
conclusions depend on them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

from .population import generate_population
from .protocol import Protocol
from .techniques import TECHNIQUE_INDEX

OUTCOME_COLUMNS: tuple[str, ...] = ("pleasure", "comfort", "discomfort", "satisfaction")

SATISFACTION_BAND_CUTS: tuple[float, float, float] = (0.25, 0.45, 0.65)
SATISFACTION_BAND_LABELS: tuple[str, ...] = ("not satisfactory", "somewhat", "quite", "very")

MAX_MINUTES = 60


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


@dataclass(frozen=True)
class ModelParameters:
    """Every global coefficient of the sequence model, in one inspectable object."""

    # integration
    dt_min: float = 1.0
    arousal_gain: float = 0.62
    arousal_decay: float = 0.11
    warmup_penalty: float = 1.25
    overstim_penalty: float = 1.05
    rhythm_bonus_scale: float = 0.55
    plateau_minutes: float = 8.0

    # discomfort accumulation
    discomfort_warmup_weight: float = 1.10
    discomfort_overstim_weight: float = 0.95
    discomfort_baseline: float = 0.02
    discomfort_saturation: float = 0.120

    # comfort outcome
    comfort_intercept: float = 1.15
    comfort_verbal_weight: float = 1.60
    comfort_discomfort_weight: float = 2.30
    comfort_sensitivity_weight: float = 1.10
    comfort_communication_weight: float = 0.55
    comfort_obligation_weight: float = 0.75

    # pleasure outcome
    pleasure_intercept: float = -1.35
    pleasure_peak_weight: float = 2.70
    pleasure_sustained_weight: float = 0.95
    pleasure_comfort_weight: float = 0.85
    pleasure_discomfort_weight: float = 1.70

    # discomfort outcome
    discomfort_intercept: float = -2.10
    discomfort_acc_weight: float = 3.30

    # satisfaction outcome (intercept and scale are the calibrated parameters)
    satisfaction_intercept: float = -1.10
    satisfaction_scale: float = 1.20
    satisfaction_pleasure_weight: float = 1.70
    satisfaction_comfort_weight: float = 0.95
    satisfaction_discomfort_weight: float = 1.25
    satisfaction_unmet_weight: float = 0.85

    def __post_init__(self) -> None:
        for name, (low, high) in PARAMETER_BOUNDS.items():
            value = float(getattr(self, name))
            if not (low <= value <= high):
                raise ValueError(
                    f"{name}={value} is outside the documented range [{low}, {high}]"
                )

    def numeric_items(self) -> dict[str, float]:
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}

    def with_overrides(self, **overrides: float) -> "ModelParameters":
        known = {field.name for field in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown model parameters: {sorted(unknown)}")
        return ModelParameters(**{**asdict(self), **overrides})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Documented bounds. A coefficient outside these would make the simulation
# uninterpretable, so the model refuses rather than silently producing nonsense.
PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "dt_min": (0.1, 10.0),
    "arousal_gain": (0.0, 3.0),
    "arousal_decay": (0.0, 1.0),
    "warmup_penalty": (0.0, 5.0),
    "overstim_penalty": (0.0, 5.0),
    "rhythm_bonus_scale": (0.0, 3.0),
    "plateau_minutes": (1.0, 60.0),
    "discomfort_warmup_weight": (0.0, 5.0),
    "discomfort_overstim_weight": (0.0, 5.0),
    "discomfort_baseline": (0.0, 1.0),
    "discomfort_saturation": (0.01, 5.0),
    "comfort_intercept": (-6.0, 6.0),
    "comfort_verbal_weight": (0.0, 6.0),
    "comfort_discomfort_weight": (0.0, 8.0),
    "comfort_sensitivity_weight": (0.0, 5.0),
    "comfort_communication_weight": (0.0, 5.0),
    "comfort_obligation_weight": (0.0, 5.0),
    "pleasure_intercept": (-6.0, 6.0),
    "pleasure_peak_weight": (0.0, 8.0),
    "pleasure_sustained_weight": (0.0, 5.0),
    "pleasure_comfort_weight": (0.0, 5.0),
    "pleasure_discomfort_weight": (0.0, 8.0),
    "discomfort_intercept": (-8.0, 4.0),
    "discomfort_acc_weight": (0.0, 10.0),
    "satisfaction_intercept": (-6.0, 6.0),
    "satisfaction_scale": (0.1, 8.0),
    "satisfaction_pleasure_weight": (0.0, 6.0),
    "satisfaction_comfort_weight": (0.0, 6.0),
    "satisfaction_discomfort_weight": (0.0, 6.0),
    "satisfaction_unmet_weight": (0.0, 6.0),
}


DEFAULT_PARAMETERS = ModelParameters()


# --------------------------------------------------------------------------- #
# Schedule: turn a protocol into per-minute arrays
# --------------------------------------------------------------------------- #

def build_schedule(protocol: Protocol, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand a protocol into per-tick ``(technique_index, intensity, verbal)`` arrays."""
    keys = list(TECHNIQUE_INDEX)
    key_to_index = {key: i for i, key in enumerate(keys)}

    tech_indices: list[int] = []
    intensities: list[float] = []
    verbal_flags: list[bool] = []

    for step in protocol.steps:
        ticks = max(1, int(round(step.duration_min / dt)))
        technique = step.technique
        tech_indices.extend([key_to_index[step.technique_key]] * ticks)
        intensities.extend([step.resolved_intensity()] * ticks)
        verbal_flags.extend([technique.verbal] * ticks)

    if len(tech_indices) > MAX_MINUTES * int(round(1.0 / dt)):
        raise ValueError(
            f"protocol is longer than the {MAX_MINUTES}-minute simulation window"
        )
    return (
        np.asarray(tech_indices, dtype=np.int64),
        np.asarray(intensities, dtype=float),
        np.asarray(verbal_flags, dtype=bool),
    )


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

def simulate_protocol(
    profiles: pd.DataFrame,
    protocol: Protocol,
    parameters: ModelParameters | None = None,
    keep_trajectory: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Simulate one protocol for every woman in ``profiles``.

    Returns ``(outcomes, trajectory)``. ``trajectory`` is ``None`` unless
    ``keep_trajectory`` is True, in which case it holds the mean arousal curve and
    is used by the app to show *how* a sequence plays out over time.
    """
    params = parameters or DEFAULT_PARAMETERS
    dt = params.dt_min
    n = len(profiles)
    if n == 0:
        raise ValueError("cannot simulate an empty population")

    tech_indices, intensities, verbal_flags = build_schedule(protocol, dt=dt)
    technique_keys = list(TECHNIQUE_INDEX)
    directness = np.array([TECHNIQUE_INDEX[k].directness for k in technique_keys])
    warmup_need = np.array([TECHNIQUE_INDEX[k].warmup_need for k in technique_keys])
    overstim_load = np.array([TECHNIQUE_INDEX[k].overstim_load for k in technique_keys])
    comfort_boost = np.array([TECHNIQUE_INDEX[k].comfort_boost for k in technique_keys])
    rhythm_sensitivity = np.array([TECHNIQUE_INDEX[k].rhythm_sensitivity for k in technique_keys])

    pref_k = profiles["preferred_intensity"].to_numpy(dtype=float)
    pref_d = profiles["preferred_directness"].to_numpy(dtype=float)
    tolerance = profiles["intensity_tolerance"].to_numpy(dtype=float)
    woman_warmup = profiles["warmup_need_min"].to_numpy(dtype=float)
    over_threshold = profiles["overstim_threshold"].to_numpy(dtype=float)
    rhythm_dep = profiles["rhythm_dependence"].to_numpy(dtype=float)
    comfort_sens = profiles["comfort_sensitivity"].to_numpy(dtype=float)
    comm_open = profiles["communication_openness"].to_numpy(dtype=float)
    obligation_sens = profiles["obligation_sensitivity"].to_numpy(dtype=float)

    arousal = np.zeros(n)
    discomfort_acc = np.zeros(n)
    overstim_acc = np.zeros(n)
    peak = np.zeros(n)
    sustained = np.zeros(n)
    elapsed = np.zeros(n)
    verbal_minutes = np.zeros(n)
    plateau = np.zeros(n)
    previous_tech = np.full(n, -1, dtype=np.int64)
    achieved_directness = np.zeros(n)
    trajectory: list[float] = []

    for tick in range(len(tech_indices)):
        tech_idx = int(tech_indices[tick])
        key = technique_keys[tech_idx]
        is_verbal = bool(verbal_flags[tick])

        if is_verbal:
            # Communication costs time: arousal drifts down, comfort accrues.
            verbal_minutes += dt
            arousal = np.clip(arousal - 0.02 * dt * arousal, 0.0, 1.0)
            plateau = np.zeros(n)                       # a pause breaks the rhythm
            previous_tech = np.full(n, -1, dtype=np.int64)
            trajectory.append(float(arousal.mean()))
            continue

        k = np.full(n, float(intensities[tick]))
        d = np.full(n, float(directness[tech_idx]))

        # 1. fit to her preferences
        fit_intensity = np.exp(-(((k - pref_k) / tolerance) ** 2))
        fit_directness = 1.0 - np.abs(d - pref_d)
        base_fit = np.clip(fit_intensity * fit_directness, 0.0, 1.0)

        # 2. warm-up deficit: direct contact before she is ready
        deficit = np.where(
            woman_warmup > 0,
            np.clip(1.0 - elapsed / np.maximum(woman_warmup, 1e-6), 0.0, 1.0),
            0.0,
        )
        premature = deficit * warmup_need[tech_idx] * d

        # 3. overstimulation: past her threshold, intensity subtracts
        exposure = overstim_load[tech_idx] * (0.40 + 0.60 * k)
        overstim = np.clip(exposure - over_threshold, 0.0, None)

        # 4. rhythm: sustained, unchanging stimulation accumulates a bonus
        same = previous_tech == tech_idx
        plateau = np.where(same, plateau + dt, 0.0)
        rhythm_bonus = rhythm_dep * np.minimum(plateau / params.plateau_minutes, 1.0)

        d_arousal = dt * (
            params.arousal_gain * base_fit * (1.0 + params.rhythm_bonus_scale * rhythm_bonus) * (1.0 - arousal)
            - params.arousal_decay * arousal
            - params.warmup_penalty * premature
            - params.overstim_penalty * overstim
        )
        arousal = np.clip(arousal + d_arousal, 0.0, 1.0)

        peak = np.maximum(peak, arousal)
        sustained += dt * (arousal > 0.70)
        discomfort_acc += dt * (
            params.discomfort_warmup_weight * premature
            + params.discomfort_overstim_weight * overstim
            + params.discomfort_baseline * comfort_sens
        )
        overstim_acc += dt * overstim
        achieved_directness = np.maximum(achieved_directness, np.where(k > 0.05, d, 0.0))

        elapsed += dt
        previous_tech = np.full(n, tech_idx, dtype=np.int64)
        trajectory.append(float(arousal.mean()))

    stimulation_minutes = np.maximum(elapsed, 1e-6)
    total_minutes = np.maximum(elapsed + verbal_minutes, 1e-6)
    # Checks per ten minutes of stimulation, saturating: three check-ins in a
    # 25-minute protocol counts as real communication; one does not.
    check_density = np.minimum(verbal_minutes / stimulation_minutes * 10.0 / 2.0, 1.0)
    # Discomfort is measured as a RATE per minute of encounter, then saturated.
    # This is what stops a long, well-paced protocol being punished simply for
    # lasting longer than a short, badly sequenced one.
    discomfort_rate = discomfort_acc / total_minutes
    dis_index = 1.0 - np.exp(-discomfort_rate / params.discomfort_saturation)

    # ---- outcomes --------------------------------------------------------
    comfort = sigmoid(
        params.comfort_intercept
        + params.comfort_verbal_weight * check_density
        - params.comfort_discomfort_weight * dis_index
        - params.comfort_sensitivity_weight * (comfort_sens - 0.5)
        + params.comfort_communication_weight * (comm_open - 0.5)
        - params.comfort_obligation_weight * obligation_sens * (1.0 - check_density)
    )

    pleasure = sigmoid(
        params.pleasure_intercept
        + params.pleasure_peak_weight * peak
        + params.pleasure_sustained_weight * np.minimum(sustained / 6.0, 1.0)
        + params.pleasure_comfort_weight * comfort
        - params.pleasure_discomfort_weight * dis_index
    )

    discomfort = sigmoid(
        params.discomfort_intercept + params.discomfort_acc_weight * dis_index
    )

    unmet = np.clip(pref_d - achieved_directness, 0.0, 1.0)
    satisfaction = sigmoid(
        params.satisfaction_scale
        * (
            params.satisfaction_intercept
            + params.satisfaction_pleasure_weight * pleasure
            + params.satisfaction_comfort_weight * comfort
            - params.satisfaction_discomfort_weight * dis_index
            - params.satisfaction_unmet_weight * unmet
        )
    )

    outcomes = pd.DataFrame(
        {
            "woman_id": profiles["woman_id"].to_numpy(),
            "pleasure": np.clip(pleasure, 0.0, 1.0),
            "comfort": np.clip(comfort, 0.0, 1.0),
            "discomfort": np.clip(discomfort, 0.0, 1.0),
            "satisfaction": np.clip(satisfaction, 0.0, 1.0),
            "peak_arousal": np.clip(peak, 0.0, 1.0),
            "sustained_minutes": sustained,
            "overstim_exposure": overstim_acc,
            "unmet_directness": unmet,
            "protocol": protocol.name,
        }
    )

    trajectory_frame = None
    if keep_trajectory:
        trajectory_frame = pd.DataFrame(
            {
                "minute": np.arange(1, len(trajectory) + 1) * dt,
                "mean_arousal": trajectory,
            }
        )
    return outcomes, trajectory_frame


def simulate_many(
    profiles: pd.DataFrame,
    protocols: dict[str, Protocol],
    parameters: ModelParameters | None = None,
) -> pd.DataFrame:
    """Simulate several protocols on the same women and stack the results."""
    frames = []
    for label, protocol in protocols.items():
        outcomes, _ = simulate_protocol(profiles, protocol, parameters)
        frame = outcomes.copy()
        frame["condition"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def parameters_frame(parameters: ModelParameters | None = None) -> pd.DataFrame:
    """The model constants with their evidence status, for display and export."""
    from .evidence import EVIDENCE_LEVELS, PARAMETER_EVIDENCE

    params = parameters or DEFAULT_PARAMETERS
    rows = []
    for name, value in params.numeric_items().items():
        spec = PARAMETER_EVIDENCE.get(name)
        if spec is None:
            spec = {
                "evidence": "unvalidated_relationship",
                "note": "Structural constant of the sequence model.",
            }
        level = EVIDENCE_LEVELS[spec["evidence"]]
        rows.append(
            {
                "parameter": name,
                "value": value,
                "evidence_code": level["code"],
                "evidence_status": level["label"],
                "note": spec.get("note", ""),
            }
        )
    return pd.DataFrame(rows)


def quick_check(seed: int = 42, n: int = 1000) -> pd.DataFrame:
    """Small smoke helper: baseline versus the structured default protocol."""
    from .protocol import BASELINE_PROTOCOL, default_structured_protocol

    profiles = generate_population(n, seed=seed)
    frames = simulate_many(
        profiles,
        {"baseline": BASELINE_PROTOCOL, "structured": default_structured_protocol()},
    )
    return (
        frames.groupby("condition")[list(OUTCOME_COLUMNS)]
        .mean()
        .round(3)
        .reset_index()
    )
