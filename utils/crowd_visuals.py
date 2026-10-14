import math
import re

import numpy as np


def crowd_metric_label(metric_name):
    """Return a display label without changing a metric's stored/exported key."""
    label = re.sub(r"peak[\s_-]+density", "highest density", str(metric_name), flags=re.IGNORECASE)
    return label.replace("_", " ").title()


def direction_vector(
    x,
    y,
    relative_degrees,
    spine_bearing_from_north,
    clockwise=True,
    standard_deviation=0.0,
    base_length=1.0,
):
    """Convert a spine-relative direction into an uncertainty-scaled floorplan arrow."""
    relative_degrees = float(relative_degrees)
    spine_bearing_from_north = float(spine_bearing_from_north)
    standard_deviation = float(standard_deviation)
    if not np.isfinite(standard_deviation):
        standard_deviation = 0.0
    standard_deviation = max(0.0, standard_deviation)

    bearing = (
        spine_bearing_from_north + relative_degrees
        if clockwise
        else spine_bearing_from_north - relative_degrees
    ) % 360
    uncertainty = min(standard_deviation, 180.0)
    length_scale = max(0.2, 1.0 - uncertainty / 225.0)
    opacity = max(0.15, 1.0 - uncertainty / 180.0 * 0.85)
    radians = math.radians(bearing)
    length = float(base_length) * length_scale
    return {
        "x_start": float(x),
        "y_start": float(y),
        "x_end": float(x) + math.sin(radians) * length,
        "y_end": float(y) + math.cos(radians) * length,
        "opacity": opacity,
        "bearing": bearing,
    }


def add_direction_arrows(
    figure,
    x_values,
    y_values,
    directions,
    spine_bearing_from_north,
    clockwise=True,
    standard_deviations=None,
    base_length=1.0,
    color="#00FFFF",
):
    """Overlay arrows on a floorplan figure, optionally scaling by uncertainty."""
    if standard_deviations is None:
        standard_deviations = np.zeros(len(directions))

    for x, y, direction, standard_deviation in zip(
        x_values, y_values, directions, standard_deviations
    ):
        if not np.isfinite(direction) or not np.isfinite(x) or not np.isfinite(y):
            continue
        arrow = direction_vector(
            x,
            y,
            direction,
            spine_bearing_from_north,
            clockwise=clockwise,
            standard_deviation=standard_deviation,
            base_length=base_length,
        )
        figure.add_annotation(
            x=arrow["x_end"],
            y=arrow["y_end"],
            ax=arrow["x_start"],
            ay=arrow["y_start"],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=color,
            opacity=arrow["opacity"],
        )


def deviation_ray_segments(
    x,
    y,
    deviation_degrees,
    spine_bearing_from_north,
    length,
):
    """Return four rays for both directions along an undirected spine axis."""
    deviation = float(deviation_degrees)
    if not np.isfinite(deviation):
        return []
    deviation = min(90.0, max(0.0, deviation))
    segments = []
    for axis_direction in (0.0, 180.0):
        for offset in (deviation, -deviation):
            bearing = math.radians(
                (float(spine_bearing_from_north) + axis_direction + offset) % 360
            )
            segments.append(
                (
                    float(x),
                    float(y),
                    float(x) + math.sin(bearing) * float(length),
                    float(y) + math.cos(bearing) * float(length),
                )
            )
    return segments


def add_deviation_rays(
    figure,
    x_values,
    y_values,
    deviations,
    spine_bearing_from_north,
    length=1.0,
    color="#FFFFFF",
):
    """Overlay four directional rays per valid deviation value."""
    line_x = []
    line_y = []
    for x, y, deviation in zip(x_values, y_values, deviations):
        for x_start, y_start, x_end, y_end in deviation_ray_segments(
            x, y, deviation, spine_bearing_from_north, length
        ):
            line_x.extend([x_start, x_end, None])
            line_y.extend([y_start, y_end, None])
    if line_x:
        figure.add_scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color=color, width=1.2),
            name="Deviation directions",
            hoverinfo="skip",
            showlegend=False,
        )
