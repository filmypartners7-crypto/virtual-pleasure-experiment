"""Streamlit dashboard for VPE 0.3.0.

Flow: paste your guide -> the parser proposes steps -> you correct them -> choose
what to compare -> run -> read distributions, ablations, ordering and subgroup
results -> export.

The parser is a keyword matcher, not a language model. The review step is not a
formality: correct the technique, duration and intensity of every step before you
trust any result.
"""

from __future__ import annotations

import inspect

import pandas as pd
import streamlit as st

from vpe import visualization as viz
from vpe.calibration import CALIBRATION_REPORT, calibration_check, reference_distribution_frame
from vpe.disclaimer import (
    CANNOT_ESTABLISH,
    CAN_ESTABLISH,
    KEY_LIMITATIONS,
    SYNTHETIC_DISCLAIMER,
    evidence_statement,
)
from vpe.evidence import (
    MODEL_VERSION,
    evidence_coverage,
    evidence_summary_text,
    parameter_evidence_frame,
    studies_frame,
)
from vpe.experiment import run_experiment
from vpe.export import results_to_csv_bytes, settings_to_json_bytes
from vpe.guide_parser import parse_guide
from vpe.population import (
    ARCHETYPES,
    NUMERIC_PROFILE_COLUMNS,
    archetypes_frame,
    generate_population,
    population_variability,
    population_variables_frame,
)
from vpe.protocol import (
    BASELINE_DEFINITION,
    BASELINE_PROTOCOL,
    Protocol,
    Step,
    default_structured_protocol,
)
from vpe.sequence_model import DEFAULT_PARAMETERS, OUTCOME_COLUMNS, parameters_frame
from vpe.techniques import TECHNIQUES, TECHNIQUE_INDEX, techniques_frame

RESULT_KEY = "vpe3_result"
STEPS_KEY = "vpe3_steps"
TEXT_KEY = "vpe3_guide_text"

_PLOT_KW = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.plotly_chart).parameters
    else {"use_container_width": True}
)
_DATA_KW = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.dataframe).parameters
    else {"use_container_width": True}
)

OUTCOME_LABELS = {
    "pleasure": "Pleasure",
    "comfort": "Comfort",
    "discomfort": "Discomfort",
    "satisfaction": "Satisfaction",
}


def chart(figure) -> None:
    st.plotly_chart(figure, **_PLOT_KW)


def table(frame: pd.DataFrame, **kwargs) -> None:
    st.dataframe(frame, hide_index=True, **_DATA_KW, **kwargs)


def steps_frame_from_state() -> pd.DataFrame:
    steps = st.session_state.get(STEPS_KEY, [])
    if not steps:
        return pd.DataFrame(columns=["step", "technique_key", "minutes", "intensity"])
    return pd.DataFrame(steps)


def state_from_frame(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for position, row in enumerate(frame.to_dict("records"), start=1):
        technique_key = row.get("technique_key")
        if technique_key not in TECHNIQUE_INDEX:
            continue
        minutes = float(row.get("minutes", 2.0))
        intensity = row.get("intensity")
        if intensity is None or pd.isna(intensity):
            intensity = None
        rows.append(
            {
                "step": position,
                "technique_key": technique_key,
                "minutes": max(0.1, min(15.0, minutes)),
                "intensity": None if intensity is None else max(0.0, min(1.0, float(intensity))),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Virtual Pleasure Experiment 0.3.0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Virtual Pleasure Experiment")
st.caption(
    f"Model version {MODEL_VERSION} - a sequence simulation over 1,000 synthetic adult women, "
    "with a null scenario calibrated to published survey marginals."
)

st.warning(
    "**Synthetic women, hypothetical mechanisms.** The population's satisfaction level is "
    "calibrated to published aggregate data. Every technique, sequence and comfort mechanism "
    "is an assumption that no study has estimated. Results explore what your assumptions imply; "
    "they are not evidence that a technique works."
)

tabs = st.tabs(
    [
        "A - Guide input",
        "B - Compare",
        "C - Results",
        "D - Steps & order",
        "E - Individual differences",
        "F - Evidence",
        "G - Limitations",
    ]
)

# --------------------------------------------------------------------------- #
# A. Guide input
# --------------------------------------------------------------------------- #

with tabs[0]:
    st.subheader("A. Paste your guide")
    st.caption(
        "The parser is a keyword matcher, not a language model. It will misread some lines. "
        "You correct it in the table below - that correction is the experiment's input, so it "
        "is worth doing carefully."
    )

    guide_text = st.text_area(
        "Your guide",
        value=st.session_state.get(TEXT_KEY, ""),
        height=260,
        placeholder="Paste your full guide here, one step per line or numbered.",
        key="guide_text_area",
    )

    parse_col, load_col, clear_col = st.columns([1, 1, 1])
    if parse_col.button("Parse guide into steps", type="primary"):
        st.session_state[TEXT_KEY] = guide_text
        result = parse_guide(guide_text)
        if not result.steps:
            st.error("No steps recognised. Add technique words such as 'labia', 'hood', 'glans'.")
        else:
            st.session_state[STEPS_KEY] = state_from_frame(result.frame())
            st.success(result.summary_text())
            for warning in result.warnings:
                st.info(warning)
            if result.unclassified_blocks:
                st.warning(
                    "These blocks were not recognised and are not in the sequence. Add them "
                    "manually with the 'Add a step' control:\n\n- "
                    + "\n- ".join(result.unclassified_blocks)
                )

    if load_col.button("Load a paced example"):
        st.session_state[STEPS_KEY] = state_from_frame(
            pd.DataFrame(
                [
                    {
                        "technique_key": step.technique_key,
                        "minutes": round(float(step.duration_min), 2),
                        "intensity": step.resolved_intensity(),
                    }
                    for step in default_structured_protocol().steps
                ]
            )
        )
        st.info("Loaded the paced example protocol. Edit it, or paste your own guide above.")

    if clear_col.button("Clear steps"):
        st.session_state[STEPS_KEY] = []
        st.session_state.pop(RESULT_KEY, None)

    st.divider()
    st.markdown("**Review and correct the sequence**")
    st.caption(
        "Change the technique, minutes or intensity of any row. Duration matters as much as "
        "order: a step that lasts 20 seconds is not the same intervention as one that lasts "
        "three minutes."
    )

    current = steps_frame_from_state()
    if current.empty:
        st.info("No steps yet. Parse a guide, load the example, or add a step below.")
        seed_frame = pd.DataFrame([{"technique_key": "outer_labia", "minutes": 3.0, "intensity": 0.30}])
    else:
        seed_frame = current[["technique_key", "minutes", "intensity"]]

    edited = st.data_editor(
        seed_frame,
        num_rows="dynamic",
        key="step_editor",
        column_config={
            "technique_key": st.column_config.SelectboxColumn(
                "technique",
                help="Technique key. See the catalogue in section F.",
                options=[technique.key for technique in TECHNIQUES],
                required=True,
            ),
            "minutes": st.column_config.NumberColumn(
                "minutes", min_value=0.1, max_value=15.0, step=0.1, format="%.1f"
            ),
            "intensity": st.column_config.NumberColumn(
                "intensity (blank = default)", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"
            ),
        },
        **_DATA_KW,
    )
    st.session_state[STEPS_KEY] = state_from_frame(edited)

    total_minutes = sum(row["minutes"] for row in st.session_state[STEPS_KEY])
    st.caption(f"{len(st.session_state[STEPS_KEY])} steps, {total_minutes:.1f} minutes total.")

    with st.expander("Technique catalogue and what each parameter means"):
        table(techniques_frame())

# --------------------------------------------------------------------------- #
# B. Compare
# --------------------------------------------------------------------------- #

with tabs[1]:
    st.subheader("B. What should be compared?")
    st.caption(
        "Both conditions run on the same 1,000 synthetic women, so a difference is never caused "
        "by a different random draw."
    )

    user_steps = st.session_state.get(STEPS_KEY, [])
    user_protocol: Protocol | None = None
    if user_steps:
        try:
            user_protocol = Protocol.from_rows(user_steps, name="your guide")
        except ValueError as error:
            st.error(f"Your step table is not yet a valid protocol: {error}")

    options: dict[str, Protocol] = {"Baseline (unstructured convention)": BASELINE_PROTOCOL}
    if user_protocol is not None:
        options["Your guide"] = user_protocol
    options["Paced example"] = default_structured_protocol()

    left, right = st.columns(2)
    label_a = left.selectbox("Condition A", list(options), index=0)
    default_b_index = 1 if len(options) > 1 else 0
    label_b = right.selectbox("Condition B", list(options), index=default_b_index)

    size_col, seed_col, run_col = st.columns([1, 1, 1])
    population_size = size_col.number_input("Synthetic women", 100, 20000, 1000, 100)
    seed = seed_col.number_input("Random seed", 0, 2**32 - 1, 42, 1)
    run_clicked = run_col.button("Run experiment", type="primary")

    with st.expander("The baseline convention, in full", expanded=False):
        st.write(BASELINE_DEFINITION)
        table(BASELINE_PROTOCOL.as_frame())

    if run_clicked:
        if label_a == label_b:
            st.error("Choose two different conditions to compare.")
        elif user_protocol is None and "Your guide" in (label_a, label_b):
            st.error("Your guide has no steps yet. Parse or build one in section A.")
        else:
            with st.spinner("Simulating both protocols, ablations, ordering variants and sensitivity..."):
                result = run_experiment(
                    population_size=int(population_size),
                    seed=int(seed),
                    protocol_a=options[label_a],
                    protocol_b=options[label_b],
                    label_a=label_a,
                    label_b=label_b,
                )
            st.session_state[RESULT_KEY] = result
            st.success(
                f"Ran {result.population_size:,} synthetic women. Fingerprint "
                f"`{result.fingerprint()}`."
            )

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.info("No experiment yet. Pick two conditions and press **Run experiment**.")
    else:
        st.caption(
            f"Loaded run: seed {result.seed}, {result.population_size:,} women, "
            f"{result.label_a} vs {result.label_b}, fingerprint `{result.fingerprint()}`."
        )
        table(pd.DataFrame([result.settings["protocol_a"]]).drop(columns=["steps"]).rename(
            columns=lambda c: f"A_{c}"))

# --------------------------------------------------------------------------- #
# C. Results
# --------------------------------------------------------------------------- #

with tabs[2]:
    st.subheader("C. Results")
    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.info("Run an experiment in section B first. Nothing is pre-computed.")
    else:
        st.markdown(
            f"Mean differences are **{result.label_b} minus {result.label_a}**, paired within "
            "woman. Intervals are bootstrap percentile intervals conditional on the model - not "
            "confidence intervals for anything real."
        )
        merged = result.summary.merge(
            result.intervals[["outcome", "interval_low", "interval_high", "interval_type"]],
            on="outcome",
            how="left",
        )
        table(merged)

        left, right = st.columns(2)
        with left:
            chart(viz.trajectory_chart(result.trajectory_a, result.trajectory_b, result.label_a, result.label_b))
        with right:
            chart(viz.paired_scatter(result.paired, "pleasure", result.label_a, result.label_b))

        outcome = st.selectbox(
            "Distribution to inspect",
            options=list(OUTCOME_COLUMNS),
            format_func=lambda key: OUTCOME_LABELS.get(key, key),
            key="dist_outcome",
        )
        chart(viz.distribution_comparison(result.condition_a, result.condition_b, outcome, result.label_a, result.label_b))

        st.divider()
        st.markdown("**Export**")
        st.caption(SYNTHETIC_DISCLAIMER)
        download_left, download_right = st.columns(2)
        download_left.download_button(
            "Download per-woman CSV",
            data=results_to_csv_bytes(result),
            file_name=f"vpe3_results_seed{result.seed}.csv",
            mime="text/csv",
        )
        download_right.download_button(
            "Download full JSON record",
            data=settings_to_json_bytes(result),
            file_name=f"vpe3_record_seed{result.seed}.json",
            mime="application/json",
        )

# --------------------------------------------------------------------------- #
# D. Steps and order
# --------------------------------------------------------------------------- #

with tabs[3]:
    st.subheader("D. What each step does, and whether order matters")
    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.info("Run an experiment in section B first.")
    else:
        outcome = st.selectbox(
            "Outcome",
            options=list(OUTCOME_COLUMNS),
            format_func=lambda key: OUTCOME_LABELS.get(key, key),
            key="step_outcome",
        )

        st.markdown(f"**Ablation: remove one step of {result.label_b} at a time**")
        st.caption(
            "A negative bar means removing that step made things worse, so the step was doing "
            "work under these assumptions."
        )
        if result.ablations.empty:
            st.info("No ablation results.")
        else:
            chart(viz.ablation_chart(result.ablations, outcome))
            table(
                result.ablations[
                    ["removed_step", "removed_technique", f"delta_{outcome}"]
                    + [c for c in result.ablations.columns if c.startswith("delta_") and c != f"delta_{outcome}"]
                ].round(3)
            )

        st.divider()
        st.markdown("**Ordering: the same steps in a different order**")
        if result.sequence_variants.empty:
            st.info("No ordering variants available for this protocol.")
        else:
            chart(viz.variants_chart(result.sequence_variants, outcome))
            table(result.sequence_variants.round(3))

        st.divider()
        st.markdown("**Substitution: swap one technique at the same position**")
        if result.substitutions.empty:
            st.info("No substitutions were applicable to this protocol.")
        else:
            table(result.substitutions.round(3))

# --------------------------------------------------------------------------- #
# E. Individual differences
# --------------------------------------------------------------------------- #

with tabs[4]:
    st.subheader("E. Individual differences")
    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.info("Run an experiment in section B first.")
    else:
        outcome = st.selectbox(
            "Outcome",
            options=list(OUTCOME_COLUMNS),
            format_func=lambda key: OUTCOME_LABELS.get(key, key),
            key="indiv_outcome",
        )
        st.markdown("**Which preference profiles gained, and which lost**")
        chart(viz.subgroup_chart(result.subgroups, outcome))

        group_by = st.selectbox(
            "Group by",
            options=["archetype", *NUMERIC_PROFILE_COLUMNS],
            key="group_by",
        )
        if group_by != "archetype":
            from vpe.statistics import subgroup_summary

            binned = result.profiles.copy()
            binned[group_by] = pd.qcut(
                binned[group_by], q=3, labels=["low", "middle", "high"], duplicates="drop"
            ).astype(str)
            table(subgroup_summary(result.paired, binned, by=group_by, outcome=outcome).round(3))
        else:
            table(result.subgroups.round(3))

        st.markdown("**Example women**")
        st.caption(
            "The largest gains and the largest losses under the change. Synthetic individuals, "
            "shown with the profile variables that drove their response."
        )
        merged = result.profiles.merge(
            result.paired[["woman_id", f"a_{outcome}", f"b_{outcome}", f"diff_{outcome}"]],
            on="woman_id",
            how="left",
        )
        columns = [
            "woman_id", "archetype", f"a_{outcome}", f"b_{outcome}", f"diff_{outcome}",
            "preferred_intensity", "preferred_directness", "warmup_need_min",
            "overstim_threshold", "comfort_sensitivity",
        ]
        st.markdown("_Largest gains_")
        table(merged.nlargest(5, f"diff_{outcome}")[columns].round(3))
        st.markdown("_Largest losses_")
        table(merged.nsmallest(5, f"diff_{outcome}")[columns].round(3))

        with st.expander("Population description"):
            table(archetypes_frame())
            table(population_variability(result.profiles))
            table(population_variables_frame())

# --------------------------------------------------------------------------- #
# F. Evidence
# --------------------------------------------------------------------------- #

with tabs[5]:
    st.subheader("F. Evidence")
    st.write(evidence_summary_text())

    st.markdown("**Calibration check**")
    st.write(CALIBRATION_REPORT["statement"])
    check = calibration_check()
    table(check["table"])
    chart(viz.calibration_chart(check["table"]))
    st.caption(
        f"Worst deviation: {check['max_deviation_points']} percentage points "
        f"({'within' if check['within_tolerance'] else 'outside'} the "
        f"{check['tolerance_points']:.0f}-point tolerance; deviations grow with smaller "
        f"populations)."
    )
    with st.expander("Published category table as reported"):
        table(reference_distribution_frame())

    st.divider()
    st.markdown("**Studies used, and what each does and does not support**")
    studies = studies_frame()
    for row in studies.to_dict("records"):
        with st.expander(f"{row['authors']} ({row['year']}) - {row['journal']}"):
            st.markdown(f"**{row['title']}**")
            st.markdown(f"DOI: [{row['doi']}](https://doi.org/{row['doi']})")
            st.markdown(f"*Sample:* {row['sample']}")
            st.markdown(f"*Findings:* {row['findings']}")
            st.markdown(f"*Supports:* {row['supports']}")
            st.markdown(f"*Does not support:* {row['does_not_support']}")
            st.markdown(f"*Limitations:* {row['limitations']}")
            st.caption(
                f"Citation verified against the published record: "
                f"{'yes' if row['citation_verified'] else 'NO'} ({row['verified_on']})"
            )

    st.divider()
    st.markdown("**Evidence status of every parameter**")
    table(parameter_evidence_frame())
    chart(
        viz._style(
            __import__("plotly.express", fromlist=["px"]).bar(
                evidence_coverage(), x="evidence_status", y="parameters",
                color_discrete_sequence=[viz.ACCENT], text="parameters",
            ).update_traces(marker_line_width=0, textposition="outside").update_layout(
                xaxis_title="", yaxis_title="Parameters", showlegend=False
            ),
            "Where the numbers come from",
            "Level E is the honest home of every technique effect in this model.",
        )
    )
    table(parameters_frame(DEFAULT_PARAMETERS))

# --------------------------------------------------------------------------- #
# G. Limitations
# --------------------------------------------------------------------------- #

with tabs[6]:
    st.subheader("G. What this can and cannot establish")
    st.error(SYNTHETIC_DISCLAIMER)
    st.write(evidence_statement())

    st.markdown("**Documented limitations**")
    for item in KEY_LIMITATIONS:
        st.markdown(f"- {item}")

    st.markdown("**Cannot establish**")
    for item in CANNOT_ESTABLISH:
        st.markdown(f"- {item}")

    st.markdown("**Can establish**")
    for item in CAN_ESTABLISH:
        st.markdown(f"- {item}")

st.divider()
st.caption(
    "VPE 0.3.0 is an exploration tool for stated assumptions. It must not be described as a "
    "study, a trial, a validation, or evidence that any technique or guide works."
)
