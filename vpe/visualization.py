"""Interactive Plotly figures for the dashboard.

The visual style is a clean research/laboratory aesthetic: white background, muted
greys for the baseline, a single deep blue for the guide condition and one warm
accent for risk indicators. All charts state that the data are simulated.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BASELINE_COLOR = "#8d99a6"
GUIDE_COLOR = "#1f4e79"
ACCENT_COLOR = "#b2182b"
PALETTE = [GUIDE_COLOR, BASELINE_COLOR, "#2a9d8f", "#e9c46a", "#b2182b"]

TEMPLATE = "plotly_white"
CHART_HEIGHT = 340

OUTCOME_LABELS = {
    "subjective_arousal": "Subjective arousal (model index)",
    "pleasure": "Pleasure (model index)",
    "comfort": "Comfort (model index)",
    "satisfaction": "Satisfaction (model index)",
    "discomfort_risk": "Discomfort risk indicator (model index)",
    "orgasm_estimate_hypothetical": "Orgasm-related estimate (hypothetical)",
}


def _style(figure: go.Figure, title: str, note: str = "Simulated model output, not real data.") -> go.Figure:
    figure.update_layout(
        title={"text": f"{title}<br><sup>{note}</sup>", "x": 0.0, "xanchor": "left"},
        template=TEMPLATE,
        height=CHART_HEIGHT,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        font={"family": "Inter, Helvetica, Arial, sans-serif", "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
        hoverlabel={"bgcolor": "white"},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#eef1f4", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#eef1f4", zeroline=False)
    return figure


def population_histogram(profiles: pd.DataFrame, column: str, title: str | None = None) -> go.Figure:
    """Distribution of one synthetic profile variable."""
    label = title or column.replace("_", " ")
    figure = px.histogram(
        profiles,
        x=column,
        nbins=40,
        color_discrete_sequence=[GUIDE_COLOR],
        opacity=0.85,
    )
    figure.update_traces(marker_line_width=0)
    return _style(figure, f"Population distribution: {label}", "Synthetic profiles. Not a prevalence estimate.")


def condition_overlay(
    baseline: pd.DataFrame,
    guide: pd.DataFrame,
    column: str,
    title: str | None = None,
) -> go.Figure:
    """Overlaid distributions of one outcome in both conditions."""
    label = title or OUTCOME_LABELS.get(column, column)
    combined = pd.concat(
        [
            pd.DataFrame({"condition": "Baseline", "value": baseline[column]}),
            pd.DataFrame({"condition": "Guide", "value": guide[column]}),
        ],
        ignore_index=True,
    )
    figure = px.histogram(
        combined,
        x="value",
        color="condition",
        barmode="overlay",
        histnorm="probability density",
        nbins=45,
        opacity=0.6,
        color_discrete_map={"Baseline": BASELINE_COLOR, "Guide": GUIDE_COLOR},
        category_orders={"condition": ["Baseline", "Guide"]},
    )
    figure.update_traces(marker_line_width=0)
    return _style(figure, label)


def paired_scatter(paired: pd.DataFrame, column: str, title: str | None = None) -> go.Figure:
    """Per-couple baseline versus guide value, with the identity line."""
    label = OUTCOME_LABELS.get(column, column)
    x = paired[f"baseline_{column}"]
    y = paired[f"guide_{column}"]
    figure = px.scatter(
        x=x,
        y=y,
        opacity=0.35,
        color_discrete_sequence=[GUIDE_COLOR],
        labels={"x": "Baseline", "y": "Guide"},
    )
    figure.update_traces(marker={"size": 5, "line": {"width": 0}})
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": ACCENT_COLOR, "width": 1},
            name="No difference",
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        xaxis_title="Baseline (model index)",
        yaxis_title="Guide (model index)",
        showlegend=True,
    )
    note = "Points above the dashed line are couples where the guide scored higher."
    return _style(figure, title or f"Paired comparison: {label}", note)


def paired_difference_hist(paired: pd.DataFrame, column: str, title: str | None = None) -> go.Figure:
    """Distribution of within-couple differences."""
    label = OUTCOME_LABELS.get(column, column)
    figure = px.histogram(
        paired,
        x=f"diff_{column}",
        nbins=50,
        color_discrete_sequence=[GUIDE_COLOR],
        opacity=0.85,
    )
    figure.update_traces(marker_line_width=0)
    figure.add_vline(x=0, line_dash="dash", line_color=ACCENT_COLOR, line_width=1)
    figure.update_layout(xaxis_title="Guide minus baseline (model index units)")
    note = "Within-couple difference. The model's noise is shared across conditions."
    return _style(figure, title or f"Paired differences: {label}", note)


def subgroup_bars(subgroup_table: pd.DataFrame, outcome: str = "pleasure", title: str | None = None) -> go.Figure:
    """Mean paired difference per subgroup, with a simple +/- 1 SE whisker."""
    label = OUTCOME_LABELS.get(outcome, outcome)
    table = subgroup_table.copy()
    table["se"] = table["sd_paired_diff"] / table["n"].pow(0.5)
    table["subgroup"] = table["subgroup"].astype(str)
    figure = px.bar(
        table,
        x="subgroup",
        y="mean_paired_diff",
        error_y="se",
        color_discrete_sequence=[GUIDE_COLOR],
        hover_data={"n": True, "baseline_mean": ":.3f", "guide_mean": ":.3f", "se": ":.4f"},
    )
    figure.update_traces(marker_line_width=0)
    figure.add_hline(y=0, line_dash="dash", line_color=ACCENT_COLOR, line_width=1)
    figure.update_layout(
        xaxis_title=table["grouped_by"].iloc[0].replace("_", " ") if len(table) else "subgroup",
        yaxis_title="Mean paired difference",
    )
    note = "Whiskers are +/- 1 standard error across synthetic couples, not a confidence interval."
    return _style(figure, title or f"Subgroup comparison: {label}", note)


def sensitivity_bars(
    sensitivity_table: pd.DataFrame, outcome: str = "pleasure", title: str | None = None
) -> go.Figure:
    """Tornado-style view of how each coefficient moves the mean outcome."""
    label = OUTCOME_LABELS.get(outcome, outcome)
    column = f"delta_{outcome}"
    if column not in sensitivity_table.columns:
        raise ValueError(f"sensitivity table has no column '{column}'")

    table = sensitivity_table[["parameter", column]].copy()
    order = (
        table.groupby("parameter")[column]
        .apply(lambda s: float(s.abs().max()))
        .sort_values(ascending=True)
        .index.tolist()
    )
    figure = px.bar(
        table,
        x=column,
        y="parameter",
        color=table[column] > 0,
        color_discrete_map={True: GUIDE_COLOR, False: ACCENT_COLOR},
        orientation="h",
        category_orders={"parameter": order},
        labels={column: "Change in mean outcome"},
    )
    figure.update_traces(marker_line_width=0)
    figure.update_layout(showlegend=False, height=max(CHART_HEIGHT, 16 * len(order) + 120))
    note = f"One-at-a-time change in mean {label.lower()}. A property of the equations, not evidence about people."
    return _style(figure, title or f"Sensitivity of {label}", note)


def summary_dot_plot(summary_table: pd.DataFrame, title: str = "Mean outcome by condition") -> go.Figure:
    """Dot plot of baseline and guide means for every outcome."""
    frame = summary_table.melt(
        id_vars=["outcome"],
        value_vars=["baseline_mean", "guide_mean"],
        var_name="condition",
        value_name="mean",
    )
    frame["condition"] = frame["condition"].map(
        {"baseline_mean": "Baseline", "guide_mean": "Guide"}
    )
    frame["outcome"] = frame["outcome"].map(lambda c: OUTCOME_LABELS.get(c, c))
    figure = px.line(
        frame,
        x="mean",
        y="outcome",
        color="condition",
        markers=True,
        color_discrete_map={"Baseline": BASELINE_COLOR, "Guide": GUIDE_COLOR},
        category_orders={"condition": ["Baseline", "Guide"]},
    )
    figure.update_traces(marker={"size": 11, "line": {"width": 0}}, line={"width": 2})
    figure.update_layout(xaxis_title="Mean model index", yaxis_title="")
    return _style(figure, title, "Means of synthetic model output in (0,1).")


def evidence_counts_chart(evidence_table: pd.DataFrame) -> go.Figure:
    """How many parameters sit at each evidence level."""
    figure = px.bar(
        evidence_table,
        x="evidence_status",
        y="parameters",
        color_discrete_sequence=[ACCENT_COLOR],
        text="parameters",
    )
    figure.update_traces(marker_line_width=0, textposition="outside")
    figure.update_layout(xaxis_title="", yaxis_title="Number of parameters", showlegend=False)
    return _style(
        figure,
        "Evidence status of model parameters",
        "Continuous response coefficients and guide-effect assumptions remain hypothetical; aggregate reference categories are published survey percentages.",
    )


def empirical_satisfaction_comparison(summary: pd.DataFrame) -> go.Figure:
    """Published aggregate anchor versus assumption-driven guide scenario."""
    frame = summary.melt(
        id_vars=["category"],
        value_vars=["baseline_reference_percent", "guide_scenario_percent"],
        var_name="condition",
        value_name="percent",
    )
    frame["condition"] = frame["condition"].map(
        {
            "baseline_reference_percent": "Published reference resample",
            "guide_scenario_percent": "Hypothetical guide scenario",
        }
    )
    figure = px.bar(
        frame,
        x="category",
        y="percent",
        color="condition",
        barmode="group",
        color_discrete_map={
            "Published reference resample": BASELINE_COLOR,
            "Hypothetical guide scenario": GUIDE_COLOR,
        },
        category_orders={
            "category": [
                "Not practiced", "Not satisfactory at all", "Somewhat satisfactory",
                "Quite satisfactory", "Very satisfactory",
            ]
        },
        labels={"category": "Reported satisfaction category", "percent": "Synthetic cohort (%)"},
    )
    figure.update_layout(yaxis_range=[0, 100], xaxis_tickangle=-15)
    return _style(
        figure,
        "Empirically anchored satisfaction categories",
        "Starting margins resampled from a Spanish survey; guide shift is hypothetical, not measured evidence.",
    )


def satisfaction_factors_chart(factors: pd.DataFrame) -> go.Figure:
    """Descriptive ratings of factors perceived to affect satisfaction a lot."""
    figure = px.bar(
        factors.sort_values("percent_affects_a_lot", ascending=True),
        x="percent_affects_a_lot",
        y="factor",
        orientation="h",
        color_discrete_sequence=[GUIDE_COLOR],
        labels={"percent_affects_a_lot": "Respondents (%)", "factor": "Self-reported factor"},
    )
    figure.update_layout(xaxis_range=[0, 100], height=460)
    return _style(
        figure,
        "Factors respondents said affect sexual satisfaction 'a lot'",
        "Spanish self-selected survey; perceived influence, not causal effects or cunnilingus-specific effects.",
    )


def all_outcome_labels(columns: Sequence[str]) -> dict[str, str]:
    return {column: OUTCOME_LABELS.get(column, column) for column in columns}
