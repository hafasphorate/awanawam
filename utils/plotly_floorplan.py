FLOORPLAN_FIGURE_WIDTH = 900
FLOORPLAN_FIGURE_HEIGHT = 600

_COLORBAR_LAYOUT = {
    "len": 0.82,
    "lenmode": "fraction",
    "thickness": 18,
    "thicknessmode": "pixels",
    "x": 1.02,
    "xanchor": "left",
    "y": 0.5,
    "yanchor": "middle",
    "outlinewidth": 0,
}


def configure_floorplan_figure(fig):
    """Apply consistent dimensions, coordinate scaling, and colorbar geometry."""
    fig.update_layout(
        width=FLOORPLAN_FIGURE_WIDTH,
        height=FLOORPLAN_FIGURE_HEIGHT,
        margin=dict(l=50, r=105, t=50, b=45),
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1, constrain="domain")

    for trace in fig.data:
        marker = getattr(trace, "marker", None)
        if marker is not None and getattr(marker, "showscale", False):
            marker.colorbar.update(_COLORBAR_LAYOUT)
        elif getattr(trace, "showscale", False):
            trace.colorbar.update(_COLORBAR_LAYOUT)

    return fig