"""Export helpers: CSV of synthetic results and JSON of experiment settings.

Both exports carry the synthetic-data disclaimer so that a file cannot circulate
without it.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

from .disclaimer import SYNTHETIC_DISCLAIMER
from .experiment import ExperimentResult

CSV_DISCLAIMER_PREFIX = "#"


def results_frame(result: ExperimentResult) -> pd.DataFrame:
    """Tidy per-couple table: profile, both conditions and the paired difference."""
    frame = result.profiles.copy()
    frame = frame.merge(result.paired, on="couple_id", how="left", validate="one_to_one")
    frame = frame.merge(result.empirical_scenario, on="couple_id", how="left", validate="one_to_one")
    return frame


def results_to_csv_bytes(result: ExperimentResult) -> bytes:
    """CSV of synthetic results, with the disclaimer as leading comment lines."""
    frame = results_frame(result)
    header_lines = [
        f"{CSV_DISCLAIMER_PREFIX} {SYNTHETIC_DISCLAIMER}",
        f"{CSV_DISCLAIMER_PREFIX} model_version={result.settings['model_version']} "
        f"seed={result.settings['random_seed']} "
        f"population_size={result.settings['population_size']} "
        f"run_fingerprint={result.settings['run_fingerprint']}",
        f"{CSV_DISCLAIMER_PREFIX} interval_note={result.settings['interval_labelling']}",
    ]
    buffer = io.StringIO()
    buffer.write("\n".join(header_lines))
    buffer.write("\n")
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def settings_to_json_bytes(result: ExperimentResult) -> bytes:
    """JSON record of settings, model version, seed, equations and disclaimers."""
    payload: dict[str, Any] = dict(result.settings)
    payload["exported_rows"] = int(len(result.paired))
    payload["column_notes"] = {
        "diff_*": "guide value minus baseline value for the same synthetic couple",
        "profile columns": "computer-generated profile variables; not real people",
        "outcome columns": "hypothetical response-model output in (0,1); unitless indices, not measurements",
        "empirical_reference_category": "Synthetic category resampled from rounded published aggregate percentages; not a source participant record",
        "empirical_reference_latent_utility": "Mathematical latent value sampled inside the source-derived ordinal category; not a measured variable",
        "baseline_category / guide_scenario_category": "Ordinal reference category and hypothetical guide scenario category; guide-effect parameter is not estimated from the source study",
    }
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def read_exported_csv(payload: bytes) -> pd.DataFrame:
    """Read back a CSV produced by :func:`results_to_csv_bytes` (used in tests)."""
    return pd.read_csv(io.BytesIO(payload), comment=CSV_DISCLAIMER_PREFIX)


def read_exported_json(payload: bytes) -> dict[str, Any]:
    """Read back a JSON export (used in tests)."""
    return json.loads(payload.decode("utf-8"))
