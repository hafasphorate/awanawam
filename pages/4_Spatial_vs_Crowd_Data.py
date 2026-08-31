import json
import re
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo  # Built-in Python 3.9+ timezone library

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from supabase import Client, create_client

# Page Configuration
st.set_page_config(
    page_title="VGA & Crowd Metrics Correlation Analysis", layout="wide"
)

st.title("VGA & Crowd Metrics Pairwise Correlation Analysis")
st.write(
    "Upload your combined dataset (JSON) to compute pairwise correlations and "
    "analyze relationships across all spatial, visibility, and crowd metrics."
)


# Initialize Supabase Client
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


try:
    supabase = init_supabase()
except Exception:
    st.sidebar.warning("Supabase credentials not configured in secrets.toml.")
    supabase = None

# -----------------------------------------------------------------------------
# 1. File Upload & Metadata Section
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Data File (JSON)", type=["json"])


def load_and_parse_json(file):
    """Parse JSON dataset into a pandas DataFrame, flattening nested structures if present."""
    data = json.load(file)
    raw_nodes = data

    if isinstance(data, dict):
        for key in (
            "grid_nodes_correlation_data",
            "crowd_metrics_by_grid",
            "vga_floorplan_nodes",
            "vga_results",
            "vga_grid",
            "node_data",
            "nodes",
            "data",
            "results",
            "items",
        ):
            value = data.get(key)
            if isinstance(value, list):
                raw_nodes = value
                break
            if isinstance(value, dict) and isinstance(value.get("nodes"), list):
                raw_nodes = value["nodes"]
                break

        has_crowd_metrics = any(
            any(
                metric in str(column).lower()
                for metric in ("volume", "density", "crowd", "pedestrian")
            )
            for column in pd.json_normalize(raw_nodes).columns
        )
        trajectories = data.get("trajectories")
        if not has_crowd_metrics and isinstance(raw_nodes, list) and isinstance(trajectories, list):
            node_df = pd.json_normalize(raw_nodes)
            trajectory_df = pd.json_normalize(trajectories)
            grid_column = "grid_node_idx"
            if grid_column in node_df.columns and grid_column in trajectory_df.columns:
                id_column = next(
                    (column for column in ("track_id", "id") if column in trajectory_df.columns),
                    None,
                )
                if id_column:
                    crowd_df = trajectory_df.groupby(grid_column).agg(
                        volume=(id_column, "count"),
                        density=(id_column, "count"),
                        pedestrian_count=(id_column, "count"),
                        unique_pedestrians=(id_column, "nunique"),
                    ).reset_index()
                    raw_nodes = node_df.merge(crowd_df, on=grid_column, how="left").fillna(0).to_dict(
                        orient="records"
                    )

    # pd.json_normalize flattens nested dicts into dot-notation columns
    df = pd.json_normalize(raw_nodes)
    return df


# -----------------------------------------------------------------------------
# 2. Pairs Plot Matrix Rendering Engine
# -----------------------------------------------------------------------------
def plot_vga_pairs_matrix(df, selected_cols):
    sub_df = df[selected_cols].apply(pd.to_numeric, errors="coerce").dropna()
    n_vars = len(selected_cols)

    corr_matrix = sub_df.corr(method="pearson")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_bwr", ["#2b5c8f", "#f7f7f7", "#d73027"]
    )

    cell_size = max(2.0, min(3.2, 20.0 / n_vars))
    font_size = max(6, min(11, int(14 - 0.6 * n_vars)))
    text_val_size = max(7, min(13, int(15 - 0.7 * n_vars)))

    fig, axes = plt.subplots(
        n_vars, n_vars, figsize=(cell_size * n_vars, cell_size * n_vars)
    )
    plt.subplots_adjust(wspace=0.18, hspace=0.18)

    clean_labels = [
        col.replace("isovist_", "").replace("_", " ").title()
        for col in selected_cols
    ]

    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j] if n_vars > 1 else axes
            col_x = selected_cols[j]
            col_y = selected_cols[i]

            if i == j:
                sns.kdeplot(
                    data=sub_df[col_x],
                    ax=ax,
                    color="#d73027",
                    fill=True,
                    alpha=0.25,
                    linewidth=1.5,
                )
                ax.set_ylabel("")
                ax.set_xlabel("")
            elif i > j:
                ax.scatter(
                    sub_df[col_x],
                    sub_df[col_y],
                    alpha=0.5,
                    edgecolor="none",
                    s=max(10, int(30 - 1.5 * n_vars)),
                    color="#2c3e50",
                )
            else:
                r_val = corr_matrix.loc[col_y, col_x]
                norm_val = (r_val + 1) / 2 if not np.isnan(r_val) else 0.5
                sq_color = cmap(norm_val)

                abs_r = abs(r_val) if not np.isnan(r_val) else 0
                sq_size = 0.25 + (0.70 * abs_r)
                alpha_val = 0.25 + (0.75 * abs_r)

                rect = plt.Rectangle(
                    (0.5 - sq_size / 2, 0.5 - sq_size / 2),
                    sq_size,
                    sq_size,
                    facecolor=sq_color,
                    alpha=alpha_val,
                    edgecolor="none",
                )
                ax.add_patch(rect)

                text_str = f"{r_val:.2f}" if not np.isnan(r_val) else "N/A"
                text_color = "white" if abs_r > 0.6 else "#111111"
                ax.text(
                    0.5,
                    0.5,
                    text_str,
                    ha="center",
                    va="center",
                    fontsize=text_val_size,
                    weight="bold",
                    color=text_color,
                )

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")

            if i < n_vars - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(
                    clean_labels[j], fontsize=font_size, fontweight="bold"
                )
                ax.tick_params(axis="x", rotation=45, labelsize=font_size - 1)

            if j > 0 and i != j:
                ax.set_yticklabels([])
            if j == 0:
                ax.set_ylabel(
                    clean_labels[i], fontsize=font_size, fontweight="bold"
                )
                ax.tick_params(axis="y", labelsize=font_size - 1)

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 3. Streamlit App Execution & Upload Workflow
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        df_nodes = load_and_parse_json(uploaded_file)

        coord_or_id_cols = {"x", "y", "z", "node_id", "id", "index", "floor"}
        all_numeric_cols = [
            col
            for col in df_nodes.columns
            if col.lower() not in coord_or_id_cols
        ]

        metric_options_signature = tuple(all_numeric_cols)
        if st.session_state.get("metric_options_signature") != metric_options_signature:
            st.session_state["selected_metrics"] = all_numeric_cols
            st.session_state["metric_selector"] = all_numeric_cols
            st.session_state["metric_options_signature"] = metric_options_signature

        st.sidebar.header("Data Filter Settings")

        col_btn1, col_btn2 = st.sidebar.columns(2)
        if col_btn1.button("Select All"):
            st.session_state["selected_metrics"] = all_numeric_cols
            st.session_state["metric_selector"] = all_numeric_cols
        if col_btn2.button("Clear All"):
            st.session_state["selected_metrics"] = []
            st.session_state["metric_selector"] = []

        if "selected_metrics" not in st.session_state:
            st.session_state["selected_metrics"] = all_numeric_cols

        selected_metrics = st.sidebar.multiselect(
            "Select Metrics for Correlation Matrix:",
            options=all_numeric_cols,
            default=st.session_state["selected_metrics"],
            key="metric_selector",
        )

        st.sidebar.markdown("---")

        # ---------------------------------------------------------------------
        # Metadata Labeling Form (Sidebar)
        # ---------------------------------------------------------------------
        st.sidebar.header("🏷️ Dataset Labeling Metadata")
        meta_location = st.sidebar.text_input(
            "Location", placeholder="e.g., Main Concourse Floor 1"
        )

        # Get local Singapore time by default for inputs
        sgt_now = datetime.now(ZoneInfo("Asia/Singapore"))

        meta_date = st.sidebar.date_input("Date", value=sgt_now.date())

        # Auto-compute Day of the Week
        meta_day = meta_date.strftime("%A")
        st.sidebar.text_input(
            "Day of Week (Auto)", value=meta_day, disabled=True
        )

        # Direct 24-hour key-in input field
        default_24h_time = sgt_now.strftime("%H:%M")
        meta_time_str = st.sidebar.text_input(
            "Time (24-Hour Format)",
            value=default_24h_time,
            placeholder="e.g., 14:30 or 08:15",
            help="Type in the video/recording time in HH:MM format (24-hour clock).",
        )

        # Validate 24-hour time format
        time_pattern = r"^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$"
        is_valid_time = bool(re.match(time_pattern, meta_time_str.strip()))

        if not is_valid_time:
            st.sidebar.error(
                "⚠️ Invalid time format. Please use **HH:MM** (e.g., 08:30 or 17:45)."
            )

        meta_comments = st.sidebar.text_area(
            "Comments", placeholder="e.g., Recorded during peak morning rush."
        )

        st.sidebar.markdown("---")
        run_matrix = st.sidebar.button(
            "Calculate Correlation Matrix",
            type="primary",
            use_container_width=True,
        )

        if len(selected_metrics) < 2:
            st.warning(
                "Please select at least **2 metrics** to generate the matrix."
            )
        else:
            if run_matrix:
                st.subheader(
                    f"Correlation Matrix ({len(selected_metrics)} Metrics Analyzed)"
                )
                fig = plot_vga_pairs_matrix(df_nodes, selected_metrics)
                st.pyplot(fig)
            elif selected_metrics:
                st.info(
                    "Selected metrics are ready. Press the button above to calculate the matrix."
                )

            corr_df = (
                df_nodes[selected_metrics]
                .apply(pd.to_numeric, errors="coerce")
                .corr(method="pearson")
            )

            with st.expander("View Numerical Pearson Correlation Matrix Table"):
                st.dataframe(
                    corr_df.style.background_gradient(
                        cmap="coolwarm", vmin=-1, vmax=1
                    ).format("{:.3f}")
                )

            with st.expander("View Full Dataset Table"):
                st.dataframe(df_nodes)

            # -----------------------------------------------------------------
            # 4. Cloud Upload Section (Targeted Omission & Metadata Packaging)
            # -----------------------------------------------------------------
            st.markdown("---")
            st.subheader("🌐 Store Dataset to Cloud Repository")

            # Standard VGA Settings Checklist / Reminder Box
            with st.expander(
                "⚠️ Standard VGA Protocol Checklist (Verify Before Upload)",
                expanded=True,
            ):
                st.markdown("""
                Please ensure your analysis followed standard VGA protocols prior to upload:
                - **Grid Dimension:** Set between at **1000mm**.
                - **Ray Angle Step (Degrees):** Fixed at **2.00** standard.
                - **Isovist Radius:** **360° unobstructed field** (or consistent truncation limit).
                - **Data Cleanliness:** Ensure crowd numbers match node timestamps accurately.
                """)

            # Convert selected columns to numeric (forces non-numeric strings/invalid types to NaN)
            numeric_df = df_nodes[selected_metrics].apply(
                pd.to_numeric, errors="coerce"
            )

            # 1. Drop rows containing NaNs across any selected metrics
            valid_df = numeric_df.dropna(subset=selected_metrics)

            # 2. Identify crowd metrics specifically
            crowd_cols = [
                col
                for col in selected_metrics
                if any(
                    k in col.lower()
                    for k in ["crowd", "pedestrian", "count", "density", "people"]
                )
            ]

            # 3. Target crowd metrics for zero checks (allows VGA = 0 to pass through)
            if crowd_cols:
                valid_df = valid_df[(valid_df[crowd_cols] > 0).all(axis=1)]
            else:
                # Fallback if no specific crowd column name matched: drop rows where ALL metrics are 0
                valid_df = valid_df[~(valid_df == 0).all(axis=1)]

            omitted_count = len(df_nodes) - len(valid_df)

            st.info(
                f"**Data Audit:** {len(valid_df)} nodes contain full spatial and crowd data across "
                f"selected metrics. ({omitted_count} incomplete nodes will be omitted)."
            )

            if st.button("Publish Data to Aggregated Cloud Database"):
                if not is_valid_time:
                    st.error(
                        "Cannot upload: Please correct the time format (HH:MM) in the sidebar."
                    )
                elif supabase is None:
                    st.error(
                        "Database connection not available. Please configure your secrets.toml."
                    )
                elif len(valid_df) == 0:
                    st.error("No valid completed rows to upload.")
                else:
                    batch_id = str(uuid.uuid4())

                    # Package record with metadata labels
                    records = [
                        {
                            "upload_batch_id": batch_id,
                            "location": meta_location,
                            "date": str(meta_date),
                            "day_of_week": meta_day,
                            "time": meta_time_str.strip(),
                            "comments": meta_comments,
                            "metrics_data": row.to_dict(),
                        }
                        for _, row in valid_df.iterrows()
                    ]

                    # Insert to Supabase table
                    res = (
                        supabase.table("vga_crowd_records")
                        .insert(records)
                        .execute()
                    )

                    if res.data:
                        st.success(
                            f"Successfully uploaded {len(records)} data points with metadata to global repository!"
                        )
                    else:
                        st.error("Failed to upload data points.")

    except Exception as e:
        st.error(f"Error parsing JSON file: {e}")
else:
    st.info("👆 Please upload your VGA & Crowd `.json` file to run analysis.")