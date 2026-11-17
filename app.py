"""Streamlit dashboard for the Virtual Pleasure Experiment.

Run with::

    streamlit run app.py

The dashboard has seven sections (A-G), including research-source provenance. It shows a clear
empty state until an experiment has been run, and it never presents simulated
output as evidence about real people.
"""

from __future__ import annotations

import inspect

import pandas as pd
import streamlit as st

from vpe import visualization as viz
from vpe.disclaimer import (
    CANNOT_ESTABLISH,
    CAN_ESTABLISH,
    EVIDENCE_STATUS_STATEMENT,
    KEY_ASSUMPTIONS,
    SYNTHETIC_DISCLAIMER,
)
from vpe.experiment import ExperimentResult, run_experiment
from vpe.empirical_scenario import empirical_scenario_distribution
from vpe.research import RESEARCH_STUDIES, reference_distribution_table, satisfaction_factors_table
from vpe.export import results_to_csv_bytes, settings_to_json_bytes
from vpe.model_config import MODEL_VERSION, ModelConfig, default_config
from vpe.population import (
    PROFILE_COLUMNS,
    distributions_table,
    population_variability,
)
from vpe.protocol import (
    BASELINE_DEFINITION,
    BASELINE_PROTOCOL,
    PROTOCOL_VARIABLE_INFO,
    PROTOCOL_VARIABLES,
    Protocol,
    default_guide_protocol,
)
from vpe.response_model import OUTCOME_COLUMNS, equations_text, outcome_columns
from vpe.statistics import paired_interval_table, sensitivity_analysis, subgroup_summary

RESULT_KEY = "vpe_result"
SENSITIVITY_KEY = "vpe_sensitivity"

# Streamlit renamed `use_container_width` to `width`; support both.
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

OUTCOME_CHOICES = {
    "subjective_arousal": "Subjective arousal",
    "pleasure": "Pleasure",
    "comfort": "Comfort",
    "satisfaction": "Satisfaction",
    "discomfort_risk": "Discomfort risk",
}

SUBGROUP_CHOICES = {
    "female_comfort_sensitivity": "Comfort sensitivity (synthetic profile)",
    "female_preference_specificity": "Preference specificity (synthetic profile)",
    "female_psychological_context": "Psychological context (synthetic profile)",
    "female_communication_preference": "Communication preference (synthetic profile)",
    "male_feedback_responsiveness": "Male feedback responsiveness (synthetic profile)",
    "male_adherence_variability": "Male adherence variability (synthetic profile)",
    "female_age_category": "Female age category",
    "male_age_category": "Male age category",
    "relationship_duration_years": "Relationship duration (years)",
}


def chart(figure) -> None:
    st.plotly_chart(figure, **_PLOT_KW)


def table(frame: pd.DataFrame, **kwargs) -> None:
    st.dataframe(frame, hide_index=True, **_DATA_KW, **kwargs)


def current_result() -> ExperimentResult | None:
    return st.session_state.get(RESULT_KEY)


def empty_state(section: str) -> None:
    st.info(
        f"No experiment has been run yet. {section} fills in after you press "
        "**Run experiment** in section A. Nothing is pre-computed or mocked."
    )


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Virtual Pleasure Experiment (VPE)",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Virtual Pleasure Experiment (VPE)")
st.caption(
    f"Model version {MODEL_VERSION} - a research-anchored synthetic scenario model "
    "with 1,000 configurable adult couples."
)

st.error(
    "**Research anchor, synthetic participants.** One satisfaction reference distribution is "
    "resampled from published aggregate survey percentages for receiving oral sex. The guide's "
    "effect is not measured by that study: all protocol-effect settings and continuous response "
    "equations are hypothetical. Results are not real-world success rates."
)

tabs = st.tabs(
    [
        "A - Experiment configuration",
        "B - Population overview",
        "C - Experiment results",
        "D - Individual variation",
        "E - Sensitivity analysis",
        "F - Scientific limitations",
        "G - Research evidence",
    ]
)

# --------------------------------------------------------------------------- #
# A. Experiment configuration
# --------------------------------------------------------------------------- #

with tabs[0]:
    st.subheader("A. Experiment configuration")

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        population_size = st.number_input(
            "Synthetic couples",
            min_value=10,
            max_value=20000,
            value=1000,
            step=100,
            help="Default is 1,000 couples, as specified. Each couple has one synthetic male and one synthetic female adult profile.",
        )
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2**32 - 1,
            value=42,
            step=1,
            help="Same seed + same protocol + same coefficients reproduce exactly the same results.",
        )
        include_orgasm = st.checkbox(
            "Also compute the optional orgasm-related estimate",
            value=False,
            help=(
                "Off by default. When enabled it is a separate, unvalidated hypothetical model "
                "estimate. The model never treats orgasm as a consequence of any protocol."
            ),
        )

    with col_right:
        st.markdown("**Protocol variables** (each on a 0-1 scale)")
        defaults = default_guide_protocol()
        protocol_values: dict[str, float] = {}
        slider_columns = st.columns(2)
        for index, variable in enumerate(PROTOCOL_VARIABLES):
            info = PROTOCOL_VARIABLE_INFO[variable]
            target = slider_columns[index % 2]
            protocol_values[variable] = target.slider(
                info["label"],
                min_value=0.0,
                max_value=1.0,
                value=float(getattr(defaults, variable)),
                step=0.05,
                help=info["help"],
                key=f"protocol_{variable}",
            )

    assumed_effect_logit = st.slider(
        "Hypothetical guide-effect assumption (logit units per full protocol-score point)",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
        help=(
            "This is NOT estimated from research. Zero means no assumed change from the published "
            "reference distribution. Positive/negative values explore favourable/unfavourable "
            "what-if scenarios. The source survey did not test your guide."
        ),
    )

    guide_text = st.text_area(
        "Guide reference text (optional, for your own reference)",
        height=140,
        placeholder="Paste your guide here. It is stored with the experiment settings but is NOT parsed.",
        help=(
            "The application does not read, summarise or interpret this text. You decide which "
            "slider value corresponds to which part of your guide."
        ),
    )

    run_col, rerun_col = st.columns([1, 3])
    run_clicked = run_col.button("Run experiment", type="primary")

    if run_clicked:
        try:
            protocol = Protocol(**protocol_values, label="user_configured", guide_reference_text=guide_text)
            config = default_config(include_orgasm_estimate=bool(include_orgasm))
            with st.spinner("Generating synthetic population and running both conditions..."):
                result = run_experiment(
                    population_size=int(population_size),
                    seed=int(seed),
                    protocol=protocol,
                    config=config,
                    assumed_effect_logit=float(assumed_effect_logit),
                )
            st.session_state[RESULT_KEY] = result
            st.session_state.pop(SENSITIVITY_KEY, None)
            st.success(
                f"Ran {result.population_size:,} paired synthetic couples. Seed {result.seed}, "
                f"run fingerprint `{result.fingerprint()}`."
            )
        except ValueError as error:
            st.error(f"Could not run the experiment: {error}")

    result = current_result()
    if result is not None:
        with rerun_col:
            new_seed = st.number_input(
                "Re-run with a different seed",
                min_value=0,
                max_value=2**32 - 1,
                value=(result.seed + 1) % (2**32),
                step=1,
                key="rerun_seed",
            )
            if st.button("Re-run with this seed"):
                protocol_dict = {
                    key: value
                    for key, value in result.settings["protocol"].items()
                    if key in Protocol.__dataclass_fields__
                }
                config_dict = {
                    key: value
                    for key, value in result.settings["model_config"].items()
                    if key in ModelConfig.__dataclass_fields__
                }
                with st.spinner("Re-running with a new synthetic population..."):
                    rerun = run_experiment(
                        population_size=result.population_size,
                        seed=int(new_seed),
                        protocol=Protocol(**protocol_dict),
                        config=ModelConfig(**config_dict),
                        assumed_effect_logit=float(result.settings.get("assumed_effect_logit", 0.0)),
                    )
                st.session_state[RESULT_KEY] = rerun
                st.session_state.pop(SENSITIVITY_KEY, None)
                st.success(f"Re-ran with seed {rerun.seed}. Fingerprint `{rerun.fingerprint()}`.")

    with st.expander("How each structured variable enters the model", expanded=False):
        st.markdown(
            "Each slider maps to a named term in the hypothetical response equations. The guide "
            "text is not automatically interpreted: you manually map your guide to these sliders. "
            "The empirical satisfaction scenario separately uses the equal-weight average of the "
            "seven slider values and the explicitly hypothetical effect coefficient above."
        )
        table(
            pd.DataFrame(
                [
                    {
                        "variable": variable,
                        "label": PROTOCOL_VARIABLE_INFO[variable]["label"],
                        "how it enters the model": PROTOCOL_VARIABLE_INFO[variable]["help"],
                    }
                    for variable in PROTOCOL_VARIABLES
                ]
            )
        )
        st.markdown("**Baseline condition (visible and fixed)**")
        st.write(BASELINE_DEFINITION)
        baseline_frame = pd.DataFrame(
            [
                {
                    "variable": variable,
                    "label": PROTOCOL_VARIABLE_INFO[variable]["label"],
                    "baseline value": round(float(getattr(BASELINE_PROTOCOL, variable)), 2),
                }
                for variable in PROTOCOL_VARIABLES
            ]
        )
        table(baseline_frame)

    with st.expander("The model equations and coefficient evidence", expanded=False):
        config_for_text = default_config(include_orgasm_estimate=bool(include_orgasm))
        st.code(equations_text(config_for_text), language="text")
        st.markdown(EVIDENCE_STATUS_STATEMENT)
        table(config_for_text.evidence_table())
        chart(viz.evidence_counts_chart(config_for_text.evidence_summary()))

# --------------------------------------------------------------------------- #
# B. Population overview
# --------------------------------------------------------------------------- #

with tabs[1]:
    st.subheader("B. Population overview")
    result = current_result()
    if result is None:
        empty_state("The population overview")
    else:
        profiles = result.profiles
        metric_columns = st.columns(4)
        metric_columns[0].metric("Synthetic couples", f"{len(profiles):,}")
        metric_columns[1].metric("Synthetic profiles", f"{len(profiles) * 2:,}")
        metric_columns[2].metric("Random seed", f"{result.seed}")
        metric_columns[3].metric("Run fingerprint", result.fingerprint())

        st.caption(
            "All profiles are computer-generated adults (18+). Age bands and profile variables are "
            "sampled from documented distributions; they are not prevalence estimates for any real "
            "population."
        )

        st.info(
            "The empirical reference category is resampled from rounded published aggregate "
            "percentages. Other individual profile traits are generated from hypothetical "
            "distributions and are not calibrated to that survey."
        )
        variable = st.selectbox(
            "Profile variable to display",
            options=PROFILE_COLUMNS + ["relationship_duration_years", "empirical_reference_category"],
            format_func=lambda name: name.replace("_", " "),
        )
        chart(viz.population_histogram(profiles, variable))

        st.markdown("**Population variability**")
        table(population_variability(profiles))

        with st.expander("Documented profile distributions"):
            table(distributions_table())

        with st.expander("Empirical reference-category composition"):
            reference_counts = (
                profiles["empirical_reference_category"]
                .value_counts(normalize=True)
                .mul(100)
                .rename("synthetic cohort percent")
                .reset_index()
                .rename(columns={"index": "category"})
            )
            table(reference_counts)
            st.caption("These percentages vary around the published aggregate marginals because 1,000 synthetic rows are sampled.")

        with st.expander("Age composition of the synthetic population"):
            age_table = (
                pd.concat(
                    [
                        profiles["male_age_category"].value_counts().rename("male profiles"),
                        profiles["female_age_category"].value_counts().rename("female profiles"),
                    ],
                    axis=1,
                )
                .fillna(0)
                .astype(int)
                .sort_index()
            )
            table(age_table.reset_index().rename(columns={"index": "age category"}))

# --------------------------------------------------------------------------- #
# C. Experiment results
# --------------------------------------------------------------------------- #

with tabs[2]:
    st.subheader("C. Experiment results")
    result = current_result()
    if result is None:
        empty_state("The results section")
    else:
        st.markdown(
            "Mean paired differences are reported with bootstrap percentile intervals "
            "**conditional on the model**. These are not clinical confidence intervals and they "
            "exclude the (larger) uncertainty in the model's assumptions."
        )
        intervals = paired_interval_table(result.paired, result.outcomes)
        table(
            result.summary.merge(intervals, on=["outcome", "mean_paired_diff"], how="left")
        )

        chart(viz.summary_dot_plot(result.summary))

        first, second = st.columns(2)
        with first:
            chart(viz.condition_overlay(result.baseline, result.guide, "pleasure"))
            chart(viz.condition_overlay(result.baseline, result.guide, "comfort"))
        with second:
            chart(viz.condition_overlay(result.baseline, result.guide, "subjective_arousal"))
            chart(viz.condition_overlay(result.baseline, result.guide, "satisfaction"))

        st.markdown("**Discomfort risk indicator (separate mechanism)**")
        st.caption(
            "Discomfort risk is computed by its own equation - boundary lapses, absent comfort "
            "checks, communication gaps and instability. It is not defined as 1 minus comfort."
        )
        chart(viz.condition_overlay(result.baseline, result.guide, "discomfort_risk"))

        st.divider()
        st.markdown("**Research-anchored satisfaction scenario**")
        st.caption(
            "This is separate from the continuous response equations above. The starting categories "
            "are resampled from a published survey of receiving oral sex. The guide shift is a "
            "hypothetical what-if parameter, not an effect measured in that study."
        )
        emp_distribution = empirical_scenario_distribution(result.empirical_scenario)
        chart(viz.empirical_satisfaction_comparison(emp_distribution))
        emp_meta = result.empirical_scenario_metadata
        practiced = max(int(emp_meta["practiced_count"]), 1)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Moved to a higher category", f"{100 * emp_meta['moved_up_count'] / practiced:.1f}%")
        m2.metric("Unchanged category", f"{100 * emp_meta['unchanged_count'] / practiced:.1f}%")
        m3.metric("Moved to a lower category", f"{100 * emp_meta['moved_down_count'] / practiced:.1f}%")
        m4.metric("Assumed logit shift", f"{emp_meta['applied_latent_logit_shift']:.2f}")
        st.caption(
            f"Movement percentages use {emp_meta['practiced_count']:,} synthetic participants classified as having practised receiving oral sex in the reference survey; the {emp_meta['not_practiced_count']:,} 'not practiced' profiles are held unchanged. These are scenario movements, not observed improvements."
        )
        with st.expander("How the research-anchored scenario is calculated"):
            st.write(emp_meta["assumption_warning"])
            st.code(
                "guide_shift = assumed_effect_logit × (guide_mean_slider_score - baseline_mean_slider_score)\n"
                "guide_category = ordinal_cutpoints(published_reference_utility + guide_shift)",
                language="text",
            )
            st.json(emp_meta)
            table(emp_distribution)

        if "orgasm_estimate_hypothetical" in result.outcomes:
            st.warning(
                "The optional orgasm-related estimate is displayed as an unvalidated hypothetical "
                "model output. It is not a prediction about any person and must not be presented "
                "as a success rate."
            )
            chart(viz.condition_overlay(result.baseline, result.guide, "orgasm_estimate_hypothetical"))

        with st.expander("Export results"):
            st.caption(SYNTHETIC_DISCLAIMER)
            download_columns = st.columns(2)
            download_columns[0].download_button(
                "Download results CSV",
                data=results_to_csv_bytes(result),
                file_name=f"vpe_results_seed{result.seed}.csv",
                mime="text/csv",
                help="Per-couple synthetic profiles, both conditions and paired differences. The disclaimer is the header of the file.",
            )
            download_columns[1].download_button(
                "Download settings JSON",
                data=settings_to_json_bytes(result),
                file_name=f"vpe_settings_seed{result.seed}.json",
                mime="application/json",
                help="Protocol, baseline definition, coefficients, evidence labels, equations, seed and model version.",
            )
            st.json(
                {
                    "model_version": result.settings["model_version"],
                    "random_seed": result.settings["random_seed"],
                    "population_size": result.settings["population_size"],
                    "run_fingerprint": result.settings["run_fingerprint"],
                    "protocol": result.settings["protocol"],
                    "interval_labelling": result.settings["interval_labelling"],
                }
            )

# --------------------------------------------------------------------------- #
# D. Individual variation
# --------------------------------------------------------------------------- #

with tabs[3]:
    st.subheader("D. Individual variation")
    result = current_result()
    if result is None:
        empty_state("The individual-variation section")
    else:
        outcome = st.selectbox(
            "Outcome",
            options=[column for column in OUTCOME_COLUMNS if column in result.outcomes],
            format_func=lambda name: OUTCOME_CHOICES.get(name, name),
            key="variation_outcome",
        )
        left, right = st.columns(2)
        with left:
            chart(viz.paired_scatter(result.paired, outcome))
        with right:
            chart(viz.paired_difference_hist(result.paired, outcome))

        st.markdown("**Subgroup comparison**")
        subgroup_variable = st.selectbox(
            "Group couples by",
            options=list(SUBGROUP_CHOICES),
            format_func=lambda name: SUBGROUP_CHOICES[name],
            key="subgroup_variable",
        )
        bins = 3 if pd.api.types.is_numeric_dtype(result.profiles[subgroup_variable]) else 0
        if bins:
            subgroup_table = subgroup_summary(
                result.paired, result.profiles, by=subgroup_variable, outcome=outcome, bins=bins
            )
        else:
            subgroup_table = subgroup_summary(
                result.paired, result.profiles, by=subgroup_variable, outcome=outcome
            )
        chart(viz.subgroup_bars(subgroup_table, outcome))
        table(subgroup_table)

        st.markdown("**Example synthetic response profiles**")
        st.caption(
            "The couples with the largest positive and largest negative paired differences, shown "
            "with their generated profile variables. These are synthetic individuals, not case studies."
        )
        difference_column = f"diff_{outcome}"
        merged = result.profiles.merge(
            result.paired[["couple_id", difference_column]], on="couple_id", how="left"
        )
        columns_to_show = [
            "couple_id",
            "male_age_category",
            "female_age_category",
            difference_column,
            *PROFILE_COLUMNS,
        ]
        most_increase = merged.nlargest(5, difference_column)[columns_to_show]
        most_decrease = merged.nsmallest(5, difference_column)[columns_to_show]
        st.markdown("_Largest increases (guide minus baseline)_")
        table(most_increase)
        st.markdown("_Largest decreases (guide minus baseline)_")
        table(most_decrease)

# --------------------------------------------------------------------------- #
# E. Sensitivity analysis
# --------------------------------------------------------------------------- #

with tabs[4]:
    st.subheader("E. Sensitivity analysis")
    result = current_result()
    if result is None:
        empty_state("The sensitivity section")
    else:
        st.markdown(
            "Each coefficient is moved up and down by a fixed percentage (clipped to its documented "
            "range) and the model is re-run with identical noise draws. The result shows how much "
            "the conclusions depend on the assumptions - it is **not** evidence about real people."
        )
        controls = st.columns(3)
        sensitivity_outcome = controls[0].selectbox(
            "Outcome",
            options=[column for column in OUTCOME_COLUMNS if column in result.outcomes],
            format_func=lambda name: OUTCOME_CHOICES.get(name, name),
            key="sensitivity_outcome",
        )
        relative_delta = controls[1].slider(
            "Coefficient change (%)", min_value=5, max_value=50, value=20, step=5
        )
        compute = controls[2].button("Compute sensitivity")

        if compute:
            protocol_dict = {
                key: value
                for key, value in result.settings["protocol"].items()
                if key in Protocol.__dataclass_fields__
            }
            config_dict = {
                key: value
                for key, value in result.settings["model_config"].items()
                if key in ModelConfig.__dataclass_fields__
            }
            with st.spinner("Re-running the model for each coefficient..."):
                frame = sensitivity_analysis(
                    profiles=result.profiles,
                    protocol=Protocol(**protocol_dict),
                    config=ModelConfig(**config_dict),
                    seed=result.seed,
                    outcomes=[sensitivity_outcome],
                    relative_delta=relative_delta / 100.0,
                )
            st.session_state[SENSITIVITY_KEY] = {
                "fingerprint": result.fingerprint(),
                "outcome": sensitivity_outcome,
                "delta": relative_delta,
                "frame": frame,
            }

        stored = st.session_state.get(SENSITIVITY_KEY)
        if stored is None:
            st.info("Press **Compute sensitivity** to run the one-at-a-time analysis.")
        elif stored["fingerprint"] != result.fingerprint():
            st.info("The experiment changed since the last sensitivity run. Press **Compute sensitivity** again.")
        else:
            frame = stored["frame"]
            chart(viz.sensitivity_bars(frame, stored["outcome"]))
            display = frame.reindex(
                frame[f"delta_{stored['outcome']}"].abs().sort_values(ascending=False).index
            )
            table(display)
            st.caption(
                "Largest absolute effects first. A large effect means the model's answer depends "
                "strongly on a coefficient nobody has validated."
            )

# --------------------------------------------------------------------------- #
# F. Scientific limitations
# --------------------------------------------------------------------------- #

with tabs[5]:
    st.subheader("F. Scientific limitations")
    st.error(SYNTHETIC_DISCLAIMER)

    st.markdown("**Four different kinds of uncertainty, kept apart**")
    table(
        pd.DataFrame(
            [
                {
                    "kind": "Model output",
                    "what it is": "A number produced by the equations in vpe/response_model.py for one synthetic couple.",
                    "status": "Deterministic given inputs, coefficients and noise.",
                },
                {
                    "kind": "Statistical uncertainty (conditional on the model)",
                    "what it is": "Bootstrap percentile intervals for mean paired differences, plus sampling variation from 1,000 synthetic couples.",
                    "status": "Describes the simulation only. Not a confidence interval for anything real.",
                },
                {
                    "kind": "Uncertainty from assumptions",
                    "what it is": "Continuous response coefficients, protocol-effect mapping and the guide-effect parameter are hypothetical.",
                    "status": "Not quantified. Section E explores some response-model assumptions; the empirical scenario exposes its effect coefficient directly.",
                },
                {
                    "kind": "Real-world empirical evidence",
                    "what it is": "Measurements from real people under a controlled design.",
                    "status": "Published aggregate self-report data are included as a reference anchor. There are no human outcome data for this particular guide or its effect.",
                },
            ]
        )
    )

    st.markdown("**What this simulation cannot establish**")
    for item in CANNOT_ESTABLISH:
        st.markdown(f"- {item}")

    st.markdown("**What it can establish**")
    for item in CAN_ESTABLISH:
        st.markdown(f"- {item}")

    st.markdown("**Documented assumptions**")
    for item in KEY_ASSUMPTIONS:
        st.markdown(f"- {item}")

    st.markdown("**Evidence status of the coefficients**")
    st.write(EVIDENCE_STATUS_STATEMENT)
    config_for_limitations = default_config()
    table(config_for_limitations.evidence_table())

    st.markdown("**Equations used**")
    st.code(equations_text(config_for_limitations), language="text")



# --------------------------------------------------------------------------- #
# G. Research evidence
# --------------------------------------------------------------------------- #

with tabs[6]:
    st.subheader("G. Human research anchors")
    st.success(
        "This version includes actual published aggregate human-survey percentages. "
        "The data anchor the starting distribution only; they do not estimate your guide's effect."
    )
    st.markdown("**Direct reference: satisfaction with receiving oral sex (cunnilingus)**")
    primary = RESEARCH_STUDIES[0]
    st.write(
        f"{primary['authors']} ({primary['year']}) studied {primary['sample']}. "
        "The table below reproduces the paper's rounded aggregate response percentages. "
        "VPE samples synthetic category labels from these percentages; it does not contain the original responses."
    )
    table(reference_distribution_table())
    st.caption(primary["limitations"])
    st.markdown(f"[Read the published study]({primary['url']}) · DOI: {primary['doi']}")

    st.markdown("**What respondents said affects sexual satisfaction**")
    st.write(
        "The same study asked how much different factors affected respondents' sexual satisfaction. "
        "These are perceived influences, not causal effects and not estimates specific to cunnilingus."
    )
    factors = satisfaction_factors_table()
    chart(viz.satisfaction_factors_chart(factors))
    table(factors)

    st.markdown("**Other research used to shape interpretation**")
    for study in RESEARCH_STUDIES[1:]:
        with st.expander(f"{study['authors']} ({study['year']}) - {study['title']}"):
            st.write(f"**Sample:** {study['sample']}")
            st.write(f"**Relevance:** {study['relevance']}")
            st.write(f"**Limitations:** {study['limitations']}")
            st.write(f"**Use in VPE:** {study['use_in_vpe']}")
            st.markdown(f"[Open source]({study['url']}) · DOI: {study['doi']}")

    st.warning(
        "Research anchoring does not turn a hypothetical guide effect into an empirical finding. "
        "The slider in section A is a what-if assumption. Use zero for the no-effect scenario; "
        "positive and negative values illustrate sensitivity to that assumption."
    )

st.divider()
st.caption(
    "VPE is a research-anchored computational scenario tool. It is not a clinical trial, "
    "validation or test of real-world effectiveness for the user's guide."
)
