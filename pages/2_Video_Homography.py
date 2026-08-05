# pages/2_Video_Homography.py
import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Polygon, Point
from utils.tracking_engine import HomographyCalibrator
from views.tracking_view import render_tracking_view

st.set_page_config(page_title="Module 2: Video Homography & Tracking", layout="wide")

st.title("📹 Module 2: Video Homography & Region Selection")

# Initialize Session State
if "dxf_walls" not in st.session_state:
    st.session_state.dxf_walls = []
if "vga_grid_df" not in st.session_state:
    st.session_state.vga_grid_df = None
if "selected_polygon_pts" not in st.session_state:
    st.session_state.selected_polygon_pts = []
if "homography_matrix" not in st.session_state:
    st.session_state.homography_matrix = None

# Navigation Tabs
tab_import, tab_region, tab_tracking = st.tabs([
    "📂 2.1 Import DXF / JSON", 
    "📐 2.2 Define Polygon Region", 
    "🔥 2.3 Occupancy Analytics"
])

# ==========================================
# TAB 1: FILE IMPORT (DXF or Exported JSON)
# ==========================================
with tab_import:
    st.subheader("Step 2.1: Load Floorplan & VGA Data")
    st.markdown("Upload your previously exported **JSON data** or raw **DXF Floorplan** file.")

    col_json, col_dxf = st.columns(2)

    with col_json:
        st.markdown("### 📄 Option A: Import Exported JSON")
        uploaded_json = st.file_uploader("Upload JSON Export", type=["json"], key="json_uploader")
        
        if uploaded_json is not None:
            try:
                data = json.load(uploaded_json)
                
                # Extract VGA Grid if available
                if "vga_grid" in data:
                    st.session_state.vga_grid_df = pd.DataFrame(data["vga_grid"])
                    st.success(f"✅ Loaded VGA Grid with {len(st.session_state.vga_grid_df)} nodes.")
                
                # Extract Calibration / Polygon points if present
                if "polygon_points" in data:
                    st.session_state.selected_polygon_pts = data["polygon_points"]
                    st.success(f"✅ Loaded {len(data['polygon_points'])} saved polygon points.")

                if "homography_matrix" in data:
                    st.session_state.homography_matrix = np.array(data["homography_matrix"])
                    st.success("✅ Loaded pre-saved Homography Matrix.")
                    
            except Exception as e:
                st.error(f"Error parsing JSON file: {e}")

    with col_dxf:
        st.markdown("### 📐 Option B: Import DXF File")
        uploaded_dxf = st.file_uploader("Upload DXF File", type=["dxf"], key="dxf_uploader")
        
        if uploaded_dxf is not None:
            # Here you can hook in your existing ezdxf parsing engine:
            # st.session_state.dxf_walls = parse_dxf_walls(uploaded_dxf)
            st.info("DXF file uploaded successfully. Wall polylines ready for rendering.")

# ==========================================
# TAB 2: POLYGON REGION SELECTION
# ==========================================
with tab_region:
    st.subheader("Step 2.2: Define Analysis Polygon on Floorplan")
    
    st.markdown(
        "Specify the vertices (X, Y in meters) of your Region of Interest (ROI) polygon. "
        "You can enter coordinates manually or modify imported points below."
    )

    col_controls, col_plot = st.columns([1, 2])

    with col_controls:
        st.markdown("#### Polygon Vertices (CAD World m)")
        
        # Interactive Point Table Editor
        default_df = pd.DataFrame(
            st.session_state.selected_polygon_pts if st.session_state.selected_polygon_pts 
            else [{"X (m)": 0.0, "Y (m)": 0.0}, {"X (m)": 10.0, "Y (m)": 0.0}, 
                  {"X (m)": 10.0, "Y (m)": 10.0}, {"X (m)": 0.0, "Y (m)": 10.0}]
        )

        edited_df = st.data_editor(
            default_df,
            num_rows="dynamic",
            use_container_width=True,
            key="poly_editor"
        )

        # Update Session State with edited points
        poly_pts = edited_df.values.tolist()
        st.session_state.selected_polygon_pts = poly_pts

        st.markdown("---")
        
        # Export updated configuration back to JSON
        export_payload = {
            "polygon_points": poly_pts,
            "vga_grid": st.session_state.vga_grid_df.to_dict(orient="records") if st.session_state.vga_grid_df is not None else [],
            "homography_matrix": st.session_state.homography_matrix.tolist() if st.session_state.homography_matrix is not None else None
        }

        st.download_button(
            label="💾 Export Updated JSON Config",
            data=json.dumps(export_payload, indent=2),
            file_name="floorplan_homography_config.json",
            mime="application/json",
            use_container_width=True
        )

    with col_plot:
        st.markdown("#### Live Floorplan Preview")
        
        fig = go.Figure()

        # 1. Render DXF Walls if present
        for wall in st.session_state.get("dxf_walls", []):
            wx, wy = wall.exterior.xy
            fig.add_trace(go.Scatter(x=list(wx), y=list(wy), mode="lines", line=dict(color="black", width=1), showlegend=False))

        # 2. Render VGA Nodes if present
        if st.session_state.vga_grid_df is not None:
            fig.add_trace(go.Scatter(
                x=st.session_state.vga_grid_df["x"],
                y=st.session_state.vga_grid_df["y"],
                mode="markers",
                marker=dict(size=4, color="lightgray"),
                name="VGA Grid Nodes"
            ))

        # 3. Render Polygon Outline
        if len(poly_pts) >= 3:
            px = [p[0] for p in poly_pts] + [poly_pts[0][0]]
            py = [p[1] for p in poly_pts] + [poly_pts[0][1]]

            fig.add_trace(go.Scatter(
                x=px, y=py,
                mode="lines+markers+text",
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.2)",
                line=dict(color="red", width=2.5),
                marker=dict(size=8, color="red"),
                text=[f"P{i+1}" for i in range(len(poly_pts))] + [""],
                textposition="top right",
                name="ROI Polygon"
            ))

        fig.update_layout(
            height=500,
            xaxis=dict(title="X (meters)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y (meters)"),
            margin=dict(l=10, r=10, t=10, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 3: OCCUPANCY TRACKING VIEW
# ==========================================
with tab_tracking:
    render_tracking_view(
        st.session_state.get("dxf_walls", []),
        st.session_state.get("vga_grid_df", None)
    )