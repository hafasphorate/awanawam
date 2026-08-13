import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from supabase import Client, create_client

st.set_page_config(page_title="Aggregated Insights", layout="wide")

st.title("🌐 Global Aggregated Spatial & Crowd Insights")
st.write(
    "This page synthesizes all user-contributed node data stored in the central database "
    "to compute collective correlation trends."
)


# -----------------------------------------------------------------------------
# 1. Supabase Initialization & Data Fetching
# -----------------------------------------------------------------------------
@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


try:
    supabase = init_supabase()
except Exception:
    st.error("Supabase credentials missing in secrets.toml configuration.")
    st.stop()


@st.cache_data(ttl=600)  # Refresh cache every 10 mins
def fetch_aggregated_records():
    """Fetch all raw records including metadata from Supabase."""
    response = supabase.table("vga_crowd_records").select("*").execute()
    if response.data:
        return pd.DataFrame(response.data)
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# 2. Interactive Tile Matrix Engine (Scatter, Distributions & Correlation Tiles)
# -----------------------------------------------------------------------------
def build_custom_tile_matrix(df: pd.DataFrame, selected_cols: list):
    """
    Recreates the custom pairs plot format in Plotly:
    - Diagonal: Histograms / Distribution curves
    - Lower Triangle: Scatter Plots
    - Upper Triangle: Correlation Tiles with Pearson 'r'
    """
    sub_df = df[selected_cols].apply(pd.to_numeric, errors="coerce").dropna()
    n_vars = len(selected_cols)
    corr_matrix = sub_df.corr(method="pearson")

    clean_labels = [
        col.replace("isovist_", "").replace("_", " ").title()
        for col in selected_cols
    ]

    # Create empty subplot grid with shared axes across rows/columns
    fig = make_subplots(
        rows=n_vars,
        cols=n_vars,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.015,
        vertical_spacing=0.015,
    )

    for i in range(n_vars):
        for j in range(n_vars):
            col_x = selected_cols[j]
            col_y = selected_cols[i]
            lbl_x = clean_labels[j]
            lbl_y = clean_labels[i]

            row_idx = i + 1
            col_idx = j + 1

            # --- DIAGONAL: Distribution Histogram ---
            if i == j:
                fig.add_trace(
                    go.Histogram(
                        x=sub_df[col_x],
                        marker_color="#d73027",
                        opacity=0.6,
                        name=f"{lbl_x} Distribution",
                        showlegend=False,
                        hovertemplate=f"<b>{lbl_x}</b><br>Value: %{{x}}<br>Count: %{{y}}<extra></extra>",
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            # --- LOWER TRIANGLE: Scatter Plots ---
            elif i > j:
                fig.add_trace(
                    go.Scatter(
                        x=sub_df[col_x],
                        y=sub_df[col_y],
                        mode="markers",
                        marker=dict(
                            size=3,
                            color="#2b5c8f",
                            opacity=0.5,
                        ),
                        name=f"{lbl_y} vs {lbl_x}",
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{lbl_x}</b>: %{{x:.2f}}<br>"
                            f"<b>{lbl_y}</b>: %{{y:.2f}}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            # --- UPPER TRIANGLE: Correlation Tiles ---
            else:
                r_val = corr_matrix.loc[col_y, col_x]
                r_str = f"{r_val:.2f}" if not np.isnan(r_val) else "N/A"

                # Single-cell heatmap tile representation
                fig.add_trace(
                    go.Heatmap(
                        z=[[r_val]],
                        colorscale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        showscale=False,
                        hoverongaps=False,
                        hovertemplate=(
                            f"<b>Correlation Pair:</b><br>"
                            f"{lbl_y} ↔ {lbl_x}<br>"
                            f"<b>Pearson r:</b> {r_str}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                # Overlay big legible text on top of the tile
                fig.add_annotation(
                    text=f"<b>{r_str}</b>",
                    x=0,
                    y=0,
                    xref=f"x{col_idx if col_idx > 1 else ''}",
                    yref=f"y{row_idx if row_idx > 1 else ''}",
                    showarrow=False,
                    font=dict(
                        size=max(8, int(16 - 0.6 * n_vars)),
                        color="white" if abs(r_val) > 0.5 else "#111111",
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            # Hide tick marks on internal cells to keep tiles clean
            if i < n_vars - 1:
                fig.update_xaxes(showticklabels=False, row=row_idx, col=col_idx)
            else:
                fig.update_xaxes(
                    title_text=lbl_x,
                    title_font=dict(size=max(7, int(11 - 0.3 * n_vars))),
                    tickangle=-45,
                    row=row_idx,
                    col=col_idx,
                )

            if j > 0:
                fig.update_yaxes(showticklabels=False, row=row_idx, col=col_idx)
            else:
                fig.update_yaxes(
                    title_text=lbl_y,
                    title_font=dict(size=max(7, int(11 - 0.3 * n_vars))),
                    row=row_idx,
                    col=col_idx,
                )

    # Dynamic canvas sizing so tiles stay square and readable
    grid_size = max(650, min(1400, 85 * n_vars))
    fig.update_layout(
        height=grid_size,
        width=grid_size,
        margin=dict(l=70, r=20, t=30, b=70),
        plot_bgcolor="#fcfcfc",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def render_pair_focused_inspector(
    df: pd.DataFrame, col_x: str, col_y: str, r_val: float
):
    """
    Renders an isolated high-res card displaying:
    1. Scatter Plot for Col X vs Col Y
    2. Large Correlation Score Box
    3. Distribution Histograms for both metrics
    """
    clean_x = col_x.replace("isovist_", "").replace("_", " ").title()
    clean_y = col_y.replace("isovist_", "").replace("_", " ").title()

    c1, c2, c3 = st.columns([2, 1, 2])

    with c1:
        st.markdown(f"##### 📉 Scatter Plot: `{clean_y}` vs `{clean_x}`")
        fig_scatter = px.scatter(
            df,
            x=col_x,
            y=col_y,
            trendline="ols",
            opacity=0.6,
            labels={col_x: clean_x, col_y: clean_y},
            color_discrete_sequence=["#2b5c8f"],
        )
        fig_scatter.update_layout(
            height=320, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        st.markdown("##### 🧮 Pearson Correlation")
        # Color coding correlation severity
        bg_color = (
            "#d73027"
            if r_val > 0.5
            else ("#2b5c8f" if r_val < -0.5 else "#888888")
        )
        st.markdown(
            f"""
            <div style="
                background-color: {bg_color};
                padding: 25px;
                border-radius: 12px;
                text-align: center;
                color: white;
                margin-top: 15px;">
                <h1 style="margin: 0; font-size: 42px;">{r_val:.3f}</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">r-value</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"Shows linear relation between **{clean_x}** and **{clean_y}**."
        )

    with c3:
        st.markdown(f"##### 📊 Distribution Comparison")
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=df[col_x],
                name=clean_x,
                opacity=0.6,
                marker_color="#2b5c8f",
            )
        )
        fig_dist.add_trace(
            go.Histogram(
                x=df[col_y],
                name=clean_y,
                opacity=0.6,
                marker_color="#d73027",
            )
        )
        fig_dist.update_layout(
            barmode="overlay",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# -----------------------------------------------------------------------------
# 3. Main Data Extraction & Analysis
# -----------------------------------------------------------------------------
raw_db_df = fetch_aggregated_records()

if raw_db_df.empty:
    st.warning(
        "No aggregated data found in the cloud repository yet. Upload records first."
    )
else:
    metrics_list = raw_db_df["metrics_data"].tolist()
    df_global = pd.DataFrame(metrics_list)

    st.success(
        f"**Repository Active:** Loaded **{len(df_global)}** global node records."
    )

    coord_or_id_cols = {"x", "y", "z", "node_id", "id", "index", "floor"}
    numeric_cols = [
        col
        for col in df_global.columns
        if pd.api.types.is_numeric_dtype(df_global[col])
        and col.lower() not in coord_or_id_cols
    ]

    if len(numeric_cols) < 2:
        st.error(
            "Insufficient numeric metrics in the database to form a matrix."
        )
    else:
        st.sidebar.header("Global Analysis Filters")
        selected_metrics = st.sidebar.multiselect(
            "Select Metrics for Correlation Analysis:",
            options=numeric_cols,
            default=numeric_cols,
        )

        if len(selected_metrics) >= 2:
            st.subheader(
                f"Global Spatial Matrix ({len(selected_metrics)} Metrics)"
            )

            # --- Matrix Rendering ---
            st.caption(
                "💡 **Tile Grid Overview:** Scatter Plots (bottom-left), Histograms (diagonal), Correlation Tiles (top-right)."
            )
            fig_matrix = build_custom_tile_matrix(df_global, selected_metrics)
            st.plotly_chart(fig_matrix, use_container_width=True)

            # --- Interactive Pair Inspector Section ---
            st.markdown("---")
            st.subheader("🔍 Focused Tile Pair Inspector")
            st.write(
                "Select any pair of variables below to expand and inspect their **Scatter Plot**, **Correlation Box**, and **Distributions** side-by-side:"
            )

            clean_options = {
                col.replace("isovist_", "").replace("_", " ").title(): col
                for col in selected_metrics
            }

            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                var1_label = st.selectbox("Select Variable X (Horizontal Tile):", options=list(clean_options.keys()), index=0)
            with col_sel2:
                var2_label = st.selectbox("Select Variable Y (Vertical Tile):", options=list(clean_options.keys()), index=min(1, len(clean_options)-1))

            var1 = clean_options[var1_label]
            var2 = clean_options[var2_label]

            if var1 == var2:
                st.info("Select two different metrics to inspect correlation.")
            else:
                r_val = df_global[[var1, var2]].apply(pd.to_numeric, errors="coerce").corr().iloc[0, 1]
                render_pair_focused_inspector(df_global, var1, var2, r_val)

        else:
            st.warning("Please select at least **2 metrics** to plot.")

        st.markdown("---")
        if st.button("Refresh Global Repository Cache"):
            st.cache_data.clear()
            st.rerun()

# -----------------------------------------------------------------------------
# 4. Admin Management Section (Password Protected Dataset Deletion)
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔒 Admin Portal")

admin_password = st.secrets.get("ADMIN_PASSWORD", "admin123")
input_pass = st.sidebar.text_input("Admin Password", type="password")

if input_pass == admin_password:
    st.sidebar.success("Admin Access Granted")

    st.markdown("---")
    st.header("🔑 Admin Dataset Management")
    st.write(
        "Inspect individual batch uploads and delete datasets that contain anomalies or errors."
    )

    if not raw_db_df.empty and "upload_batch_id" in raw_db_df.columns:
        meta_cols = [
            "upload_batch_id",
            "location",
            "date",
            "day_of_week",
            "time",
            "comments",
        ]
        existing_meta_cols = [c for c in meta_cols if c in raw_db_df.columns]

        batch_summary = (
            raw_db_df.groupby(existing_meta_cols, dropna=False)
            .size()
            .reset_index(name="node_count")
        )

        st.subheader("Uploaded Datasets Overview")
        st.dataframe(batch_summary, use_container_width=True)

        batch_options = {}
        for _, row in batch_summary.iterrows():
            b_id = row["upload_batch_id"]
            loc = row.get("location", "N/A")
            dt = row.get("date", "N/A")
            count = row["node_count"]
            label = f"ID: {b_id[:8]}... | Loc: {loc} | Date: {dt} | ({count} nodes)"
            batch_options[label] = b_id

        selected_label = st.selectbox(
            "Select Dataset Batch to Inspect or Remove:",
            options=list(batch_options.keys()),
        )

        if selected_label:
            selected_batch_id = batch_options[selected_label]
            batch_records = raw_db_df[
                raw_db_df["upload_batch_id"] == selected_batch_id
            ]

            single_ds_metrics = pd.DataFrame(
                batch_records["metrics_data"].tolist()
            )
            single_numeric_cols = [
                col
                for col in single_ds_metrics.columns
                if pd.api.types.is_numeric_dtype(single_ds_metrics[col])
                and col.lower() not in coord_or_id_cols
            ]

            with st.expander(f"Inspect Dataset Matrix ({selected_label})"):
                if len(single_numeric_cols) >= 2:
                    fig_single = build_custom_tile_matrix(
                        single_ds_metrics, single_numeric_cols
                    )
                    st.plotly_chart(fig_single, use_container_width=True)
                else:
                    st.info(
                        "Not enough numeric variables in this dataset to generate a plot."
                    )

            if st.button("❌ Delete Selected Dataset Batch", type="primary"):
                del_res = (
                    supabase.table("vga_crowd_records")
                    .delete()
                    .eq("upload_batch_id", selected_batch_id)
                    .execute()
                )

                if del_res.data:
                    st.success(
                        f"Successfully deleted batch `{selected_batch_id}` from the database."
                    )
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Failed to delete dataset batch from database.")
    else:
        st.info("No structured dataset batches found to manage.")
elif input_pass:
    st.sidebar.error("Incorrect Password")