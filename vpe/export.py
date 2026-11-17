"""Exports: CSV of per-woman results and JSON of the full experiment record."""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

from .disclaimer import SYNTHETIC_DISCLAIMER
from .experiment import ExperimentResult

COMMENT = "#"


def results_frame(result: ExperimentResult) -> pd.DataFrame:
    """One row per synthetic woman: profile, both conditions, and the difference."""
    frame = result.profiles.copy()
    frame = frame.merge(result.paired, on="woman_id", how="left", validate="one_to_one")
    return frame


def results_to_csv_bytes(result: ExperimentResult) -> bytes:
    frame = results_frame(result)
    header = [
        f"{COMMENT} {SYNTHETIC_DISCLAIMER}",
        f"{COMMENT} model_version={result.settings['model_version']} seed={result.seed} "
        f"population_size={result.population_size} fingerprint={result.fingerprint()}",
        f"{COMMENT} protocol_a={result.label_a} | protocol_b={result.label_b}",
        f"{COMMENT} technique parameters are hypothetical; see the JSON export for the full evidence record",
    ]
    buffer = io.StringIO()
    buffer.write("\n".join(header) + "\n")
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def settings_to_json_bytes(result: ExperimentResult) -> bytes:
    payload: dict[str, Any] = dict(result.settings)
    payload["results_summary"] = result.summary.to_dict(orient="records")
    payload["paired_intervals"] = result.intervals.to_dict(orient="records")
    payload["ablations"] = result.ablations.to_dict(orient="records") if not result.ablations.empty else []
    payload["sequence_variants"] = (
        result.sequence_variants.to_dict(orient="records") if not result.sequence_variants.empty else []
    )
    payload["substitutions"] = (
        result.substitutions.to_dict(orient="records") if not result.substitutions.empty else []
    )
    payload["calibration_check"] = result.calibration.to_dict(orient="records")
    payload["column_notes"] = {
        "diff_*": "protocol B value minus protocol A value for the same synthetic woman",
        "profile columns": "computer-generated; not real people",
        "outcome columns": "unitless model indices in (0,1), not measurements",
    }
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def read_exported_csv(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload), comment=COMMENT)


def read_exported_json(payload: bytes) -> dict[str, Any]:
    return json.loads(payload.decode("utf-8"))
