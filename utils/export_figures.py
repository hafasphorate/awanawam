"""Helpers that make exported PNGs self-describing (title, subtitle, legend)."""
import plotly.graph_objects as go


def _legend_eligible(trace) -> bool:
    if getattr(trace, "showlegend", None) is False or not getattr(trace, "name", None):
        return False
    marker = getattr(trace, "marker", None)
    if marker is not None and getattr(marker, "showscale", None):
        return False
    return True


def prepare_export_figure(fig, title=None, subtitle=None, width=1400, height=None):
    """Return a copy of a Plotly figure with a title, subtitle and legend for PNG export."""
    export = go.Figure(fig)
    current_title = export.layout.title.text if export.layout.title else None
    main_title = title or current_title or "Figure"
    text = f"<b>{main_title}</b>"
    if subtitle:
        text += f"<br><sup>{subtitle}</sup>"

    export.update_layout(
        title=dict(text=text, x=0.01, xanchor="left", y=0.97, yanchor="top"),
        margin=dict(l=70, r=40, t=110, b=70),
        width=width,
        height=height or export.layout.height or 800,
    )
    if sum(_legend_eligible(trace) for trace in export.data) >= 1:
        export.update_layout(
            showlegend=True,
            legend=dict(
                title=dict(text="Legend"),
                bgcolor="rgba(0,0,0,0.35)",
                bordercolor="#888888",
                borderwidth=1,
            ),
        )
    return export
