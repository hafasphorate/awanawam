import io
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Optional DXF parsing support
try:
    import ezdxf

    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False


# ==============================================================================
# PAGE CONFIG & STYLES
# ==============================================================================
st.set_page_config(
    page_title="CAD Point & Tracking Alignment Workspace",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stButton > button { font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "dxf_walls" not in st.session_state:
    st.session_state.dxf_walls = []
if "vga_grid_df" not in st.session_state:
    st.session_state.vga_grid_df = None
if "calibration_points" not in st.session_state:
    st.session_state.calibration_points = []
if "editing_point_idx" not in st.session_state:
    st.session_state.editing_point_idx = None
if "processed_click_sig" not in st.session_state:
    st.session_state.processed_click_sig = None


# ==============================================================================
# HELPER FUNCTIONS & DRAWING UTILS
# ==============================================================================
def parse_dxf_bytes(file_bytes):
    """Parse DXF file bytes into line segments [(x1, y1, x2, y2), ...]."""
    if not HAS_EZDXF:
        st.error(
            "ezdxf library is not installed. Please install ezdxf to import DXF files."
        )
        return []

    try:
        doc = ezdxf.read(io.BytesIO(file_bytes))
        msp = doc.modelspace()
        walls = []

        for entity in msp:
            if entity.dxftype() == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                walls.append((start.x, start.y, end.x, end.y))
            elif entity.dxftype() == "LWPOLYLINE":
                points = entity.get_points("xy")
                for i in range(len(points) - 1):
                    walls.append(
                        (points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
                    )
                if entity.closed and len(points) > 1:
                    walls.append(
                        (
                            points[-1][0],
                            points[-1][1],
                            points[0][0],
                            points[0][1],
                        )
                    )
            elif entity.dxftype() == "POLYLINE":
                points = [v.dxf.location[:2] for v in entity.vertices]
                for i in range(len(points) - 1):
                    walls.append(
                        (points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
                    )
                if entity.is_closed and len(points) > 1:
                    walls.append(
                        (
                            points[-1][0],
                            points[-1][1],
                            points[0][0],
                            points[0][1],
                        )
                    )

        return walls
    except Exception as e:
        st.error(f"Failed to parse DXF file: {str(e)}")
        return []


def create_base_cad_figure(walls, vga_df=None):
    """Generate basic Plotly figure with DXF walls and optional VGA grid data."""
    fig = go.Figure()

    # Draw DXF Walls
    if walls:
        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in walls:
            x_coords.extend([x1, x2, None])
            y_coords.extend([y1, y2, None])

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line=dict(color="#4A5568", width=1),
                name="CAD Walls",
                hoverinfo="none",
            )
        )

    # Draw VGA Grid points if available
    if vga_df is not None and not vga_df.empty:
        if "x" in vga_df.columns and "y" in vga_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=vga_df["x"],
                    y=vga_df["y"],
                    mode="markers",
                    marker=dict(size=4, color="#3182CE", opacity=0.5),
                    name="VGA Grid",
                    hoverinfo="x+y",
                )
            )

    return fig


# ==============================================================================
# APP LAYOUT & TABS
# ==============================================================================
st.title("🏗️ CAD Point & Tracking Alignment Workspace")

tab_import, tab_canvas, tab_tracking = st.tabs(
    ["📥 1. Data Import", "🎯 2. Point Calibration Canvas", "🎥 3. Tracking View"]
)

# ------------------------------------------------------------------------------
# TAB 1: DATA IMPORT
# ------------------------------------------------------------------------------
with tab_import:
    st.subheader("Import Floorplan & Analysis Data")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### DXF CAD Floorplan")
        dxf_file = st.file_uploader("Upload DXF File", type=["dxf"])
        if dxf_file is not None:
            walls = parse_dxf_bytes(dxf_file.getvalue())
            if walls:
                st.session_state.dxf_walls = walls
                st.success(f"Loaded {len(walls)} wall segments from DXF.")

    with col2:
        st.markdown("#### VGA / Analysis JSON Data")
        json_file = st.file_uploader("Upload VGA JSON Data", type=["json"])
        if json_file is not None:
            try:
                data = json.load(json_file)
                vga_data = data.get("vga_results", data.get("vga_grid", []))
                if vga_data:
                    st.session_state.vga_grid_df = pd.DataFrame(vga_data)
                    st.success(
                        f"Loaded {len(st.session_state.vga_grid_df)} grid nodes."
                    )
                else:
                    st.warning(
                        "JSON loaded, but 'vga_results' or 'vga_grid' key was empty."
                    )
            except Exception as e:
                st.error(f"Error parsing JSON: {str(e)}")


# ------------------------------------------------------------------------------
# TAB 2: POINT CALIBRATION CANVAS
# ------------------------------------------------------------------------------
with tab_canvas:
    st.subheader("Interactive Point Placement & Target Alignment")

    # Render Base Figure
    fig = create_base_cad_figure(
        st.session_state.dxf_walls, st.session_state.vga_grid_df
    )

    # Add existing calibration points
    for idx, pt in enumerate(st.session_state.calibration_points):
        is_editing = idx == st.session_state.editing_point_idx
        color = "#E53E3E" if is_editing else "#38A169"
        marker_symbol = "cross" if is_editing else "circle"

        fig.add_trace(
            go.Scatter(
                x=[pt["x"]],
                y=[pt["y"]],
                mode="markers+text",
                marker=dict(size=12, color=color, symbol=marker_symbol),
                text=[f" P{idx+1}"],
                textposition="top right",
                name=f"Point {idx+1}",
                showlegend=False,
            )
        )

    # --------------------------------------------------------------------------
    # FIX #1: Plotly Zoom/Pan persistence using uirevision
    # --------------------------------------------------------------------------
    fig.update_layout(
        template="plotly_dark",
        height=620,
        xaxis=dict(
            title="X Coordinate",
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
        ),
        yaxis=dict(
            title="Y Coordinate",
            showgrid=True,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        dragmode="pan",
        hovermode="closest",
        # uirevision keeps current zoom/pan active across st.rerun()
        uirevision="PERMANENT_CANVAS_LOCK",
    )

    # Render Plotly chart with click tracking
    chart_events = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="cad_interactive_canvas",
    )

    # Detect & Handle Point Click Selection
    if chart_events and "selection" in chart_events:
        points = chart_events["selection"].get("points", [])
        if points:
            click_pt = points[0]
            click_sig = f"{click_pt.get('x')}_{click_pt.get('y')}"

            if click_sig != st.session_state.processed_click_sig:
                st.session_state.processed_click_sig = click_sig
                new_x = click_pt.get("x")
                new_y = click_pt.get("y")

                if (
                    st.session_state.editing_point_idx is not None
                    and st.session_state.editing_point_idx
                    < len(st.session_state.calibration_points)
                ):
                    # Update target point
                    idx = st.session_state.editing_point_idx
                    st.session_state.calibration_points[idx]["x"] = new_x
                    st.session_state.calibration_points[idx]["y"] = new_y
                    st.session_state.editing_point_idx = None
                else:
                    # Append new calibration point
                    st.session_state.calibration_points.append(
                        {
                            "name": f"P{len(st.session_state.calibration_points)+1}",
                            "x": new_x,
                            "y": new_y,
                        }
                    )
                st.rerun()

    # --------------------------------------------------------------------------
    # POINT MANAGEMENT TABLE & FIX #2: Toggle Edit Mode
    # --------------------------------------------------------------------------
    st.markdown("### Calibration Points List")

    if not st.session_state.calibration_points:
        st.info("Click anywhere on the plot above to place calibration points.")
    else:
        cols = st.columns([1, 2, 2, 2, 1])
        cols[0].markdown("**Point**")
        cols[1].markdown("**X Coordinate**")
        cols[2].markdown("**Y Coordinate**")
        cols[3].markdown("**Actions**")
        cols[4].markdown("**Remove**")

        for idx, pt in enumerate(st.session_state.calibration_points):
            c_idx, c_x, c_y, c_act, c_del = st.columns([1, 2, 2, 2, 1])

            is_editing = idx == st.session_state.editing_point_idx

            c_idx.write(f"**{pt.get('name', f'P{idx+1}')}**")

            new_x = c_x.number_input(
                f"X P{idx+1}",
                value=float(pt["x"]),
                key=f"x_val_{idx}",
                label_visibility="collapsed",
            )
            new_y = c_y.number_input(
                f"Y P{idx+1}",
                value=float(pt["y"]),
                key=f"y_val_{idx}",
                label_visibility="collapsed",
            )

            # Update if manually edited in numeric box
            st.session_state.calibration_points[idx]["x"] = new_x
            st.session_state.calibration_points[idx]["y"] = new_y

            # FIX #2: Edit Button Toggle Logic
            btn_label = "🎯 Target Mode" if is_editing else "✏️ Click & Reposition"
            if c_act.button(
                btn_label,
                key=f"edit_btn_{idx}",
                use_container_width=True,
                type="primary" if is_editing else "secondary",
            ):
                if is_editing:
                    st.session_state.editing_point_idx = None  # Toggle off
                else:
                    st.session_state.editing_point_idx = idx  # Toggle on
                st.session_state.processed_click_sig = None
                st.rerun()

            # Remove point button
            if c_del.button("🗑️", key=f"del_btn_{idx}", use_container_width=True):
                st.session_state.calibration_points.pop(idx)
                st.session_state.editing_point_idx = None
                st.rerun()

        # Download Config Export
        st.markdown("---")
        export_payload = {
            "calibration_points": st.session_state.calibration_points,
            "vga_grid": (
                st.session_state.vga_grid_df.to_dict(orient="records")
                if (
                    st.session_state.vga_grid_df is not None
                    and not st.session_state.vga_grid_df.empty
                )
                else []
            ),
        }
        st.download_button(
            "💾 Download Calibration Setup (JSON)",
            data=json.dumps(export_payload, indent=2),
            file_name="calibration_config.json",
            mime="application/json",
        )


# ------------------------------------------------------------------------------
# TAB 3: TRACKING VIEW (FIX #3: Safeguard Empty DataFrame Checks)
# ------------------------------------------------------------------------------
with tab_tracking:
    st.subheader("Real-Time Tracking & Homography Projection View")

    vga_df = st.session_state.get("vga_grid_df")

    # FIX #3: Empty DataFrame check before passing to view
    if vga_df is not None and vga_df.empty:
        vga_df = None

    if len(st.session_state.calibration_points) < 4:
        st.warning(
            f"Homography calculation requires at least 4 point correspondences. Current points: {len(st.session_state.calibration_points)}/4"
        )

    t_col1, t_col2 = st.columns([2, 1])

    with t_col1:
        st.markdown("#### Projected CAD World View")
        track_fig = create_base_cad_figure(st.session_state.dxf_walls, vga_df)

        # Plot active tracking points
        if st.session_state.calibration_points:
            pts_x = [p["x"] for p in st.session_state.calibration_points]
            pts_y = [p["y"] for p in st.session_state.calibration_points]
            track_fig.add_trace(
                go.Scatter(
                    x=pts_x,
                    y=pts_y,
                    mode="markers+lines",
                    line=dict(color="#ED8936", dash="dot"),
                    marker=dict(size=10, color="#ED8936"),
                    name="Calibrated Quadrilateral",
                )
            )

        track_fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            margin=dict(l=10, r=10, t=30, b=10),
            uirevision="TRACKING_CANVAS_LOCK",
        )
        st.plotly_chart(track_fig, use_container_width=True)

    with t_col2:
        st.markdown("#### Alignment Status")
        st.metric(
            label="Walls Loaded",
            value=len(st.session_state.dxf_walls),
        )
        st.metric(
            label="Active Points",
            value=len(st.session_state.calibration_points),
        )
        st.metric(
            label="VGA Nodes",
            value=(
                len(st.session_state.vga_grid_df)
                if (
                    st.session_state.vga_grid_df is not None
                    and not st.session_state.vga_grid_df.empty
                )
                else 0
            ),
        )