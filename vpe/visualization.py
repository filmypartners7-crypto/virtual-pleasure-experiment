"""Plotly figures. Research/laboratory aesthetic, muted palette, honest subtitles."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR_A = "#8d99a6"
COLOR_B = "#1f4e79"
ACCENT = "#b2182b"
TEMPLATE = "plotly_white"
HEIGHT = 340

OUTCOME_LABELS = {
    "pleasure": "Pleasure (model index)",
    "comfort": "Comfort (model index)",
    "discomfort": "Discomfort (model index)",
    "satisfaction": "Satisfaction (model index)",
    "peak_arousal": "Peak arousal (model index)",
}


def _style(figure: go.Figure, title: str, note: str = "Synthetic model output, not real data.") -> go.Figure:
    figure.update_layout(
        title={"text": f"{title}<br><sup>{note}</sup>", "x": 0.0, "xanchor": "left"},
        template=TEMPLATE,
        height=HEIGHT,
        margin={"l": 10, "r": 10, "t": 72, "b": 10},
        font={"family": "Inter, Helvetica, Arial, sans-serif", "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#eef1f4", zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor="#eef1f4", zeroline=False)
    return figure


def distribution_comparison(
    condition_a: pd.DataFrame, condition_b: pd.DataFrame, outcome: str, label_a: str, label_b: str
) -> go.Figure:
    combined = pd.concat(
        [
            pd.DataFrame({"condition": label_a, "value": condition_a[outcome]}),
            pd.DataFrame({"condition": label_b, "value": condition_b[outcome]}),
        ],
        ignore_index=True,
    )
    figure = px.histogram(
        combined, x="value", color="condition", barmode="overlay",
        histnorm="probability density", nbins=45, opacity=0.6,
        color_discrete_map={label_a: COLOR_A, label_b: COLOR_B},
        category_orders={"condition": [label_a, label_b]},
    )
    figure.update_traces(marker_line_width=0)
    return _style(figure, OUTCOME_LABELS.get(outcome, outcome))


def paired_scatter(paired: pd.DataFrame, outcome: str, label_a: str, label_b: str) -> go.Figure:
    figure = px.scatter(
        x=paired[f"a_{outcome}"], y=paired[f"b_{outcome}"], opacity=0.35,
        color_discrete_sequence=[COLOR_B], labels={"x": label_a, "y": label_b},
    )
    figure.update_traces(marker={"size": 5, "line": {"width": 0}})
    figure.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="No difference",
                   line={"dash": "dash", "color": ACCENT, "width": 1}, hoverinfo="skip")
    )
    figure.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1],
                         xaxis_title=label_a, yaxis_title=label_b)
    return _style(figure, f"Per-woman comparison: {OUTCOME_LABELS.get(outcome, outcome)}",
                  "Above the dashed line = this woman did better under B.")


def trajectory_chart(trajectory_a: pd.DataFrame, trajectory_b: pd.DataFrame, label_a: str, label_b: str) -> go.Figure:
    frame = pd.concat(
        [
            pd.DataFrame({"minute": trajectory_a["minute"], "mean_arousal": trajectory_a["mean_arousal"], "condition": label_a}),
            pd.DataFrame({"minute": trajectory_b["minute"], "mean_arousal": trajectory_b["mean_arousal"], "condition": label_b}),
        ],
        ignore_index=True,
    )
    figure = px.line(frame, x="minute", y="mean_arousal", color="condition", markers=False,
                     color_discrete_map={label_a: COLOR_A, label_b: COLOR_B},
                     category_orders={"condition": [label_a, label_b]})
    figure.update_traces(line={"width": 2})
    figure.update_layout(xaxis_title="Minute of encounter", yaxis_title="Mean arousal (model index)")
    return _style(figure, "Arousal over time",
                  "Where a sequence gains or loses its effect. Mean across 1,000 synthetic women.")


def ablation_chart(ablations: pd.DataFrame, outcome: str = "pleasure") -> go.Figure:
    column = f"delta_{outcome}"
    if ablations.empty or column not in ablations.columns:
        return _style(go.Figure(), "Ablation", "No ablation results.")
    table = ablations[["removed_technique", column]].copy()
    figure = px.bar(table, x=column, y="removed_technique", orientation="h",
                    color=table[column] > 0, color_discrete_map={True: COLOR_B, False: ACCENT},
                    labels={column: f"Change in mean {outcome}"})
    figure.update_traces(marker_line_width=0)
    figure.update_layout(showlegend=False, height=max(HEIGHT, 40 * len(table) + 140),
                         yaxis_title="")
    return _style(figure, f"What each step contributes: {outcome}",
                  "Removing that step and re-running. Negative = the step was helping.")


def variants_chart(variants: pd.DataFrame, outcome: str = "pleasure") -> go.Figure:
    column = f"delta_{outcome}"
    if variants.empty or column not in variants.columns:
        return _style(go.Figure(), "Sequence variants", "No variant results.")
    table = variants[["variant", column]].copy()
    figure = px.bar(table, x="variant", y=column, color_discrete_sequence=[COLOR_B],
                    labels={column: f"Change in mean {outcome} vs as written"})
    figure.update_traces(marker_line_width=0)
    figure.add_hline(y=0, line_dash="dash", line_color=ACCENT, line_width=1)
    figure.update_layout(xaxis_title="", height=380)
    return _style(figure, f"Does order matter? ({outcome})",
                  "Same steps, different order. A hypothesis under these assumptions, not a finding.")


def sensitivity_chart(sensitivity: pd.DataFrame, outcome: str = "pleasure") -> go.Figure:
    column = f"delta_{outcome}"
    if sensitivity.empty or column not in sensitivity.columns:
        return _style(go.Figure(), "Sensitivity", "No sensitivity results.")
    table = sensitivity[["parameter", column]].copy()
    order = (table.groupby("parameter")[column].apply(lambda s: float(s.abs().max()))
             .sort_values(ascending=True).index.tolist())
    figure = px.bar(table, x=column, y="parameter", orientation="h",
                    color=table[column] > 0, color_discrete_map={True: COLOR_B, False: ACCENT},
                    category_orders={"parameter": order},
                    labels={column: "Change in mean outcome"})
    figure.update_traces(marker_line_width=0)
    figure.update_layout(showlegend=False, height=max(HEIGHT, 18 * len(order) + 140), yaxis_title="")
    return _style(figure, f"Sensitivity of {outcome}",
                  "One-at-a-time change in each coefficient. A property of the equations.")


def calibration_chart(calibration: pd.DataFrame) -> go.Figure:
    frame = calibration.melt(id_vars=["band"], value_vars=["published_percent", "model_percent"],
                             var_name="series", value_name="percent")
    frame["series"] = frame["series"].map({"published_percent": "Published (N=1,080)",
                                          "model_percent": "This model"})
    figure = px.bar(frame, x="band", y="percent", color="series", barmode="group",
                    color_discrete_map={"Published (N=1,080)": COLOR_A, "This model": COLOR_B},
                    text_auto=".1f")
    figure.update_traces(marker_line_width=0, textposition="outside")
    figure.update_layout(xaxis_title="Satisfaction band", yaxis_title="Percent of women", yaxis_range=[0, 80])
    return _style(figure, "Calibration check",
                  "Null scenario versus published aggregate marginals. Level only - not technique validity.")


def subgroup_chart(subgroups: pd.DataFrame, outcome: str = "pleasure") -> go.Figure:
    if subgroups.empty:
        return _style(go.Figure(), "Subgroups", "No subgroup results.")
    table = subgroups.copy()
    table["se"] = table["sd_diff"] / table["n"].pow(0.5)
    figure = px.bar(table, x="subgroup", y="mean_diff", error_y="se",
                    color_discrete_sequence=[COLOR_B],
                    hover_data={"n": True, "condition_a_mean": ":.3f", "condition_b_mean": ":.3f"})
    figure.update_traces(marker_line_width=0)
    figure.add_hline(y=0, line_dash="dash", line_color=ACCENT, line_width=1)
    figure.update_layout(xaxis_title="Preference archetype (modelling device)",
                         yaxis_title=f"Mean change in {outcome}")
    return _style(figure, f"Who gains, who loses: {outcome}",
                  "Whiskers are +/- 1 SE across synthetic women, not a confidence interval.")
