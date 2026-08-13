import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
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
# 2. Matplotlib Precision Pairs Matrix Engine
# -----------------------------------------------------------------------------
def plot_vga_pairs_matrix(df: pd.DataFrame, selected_cols: list):
    """
    Recreates the exact visual matrix:
    - Diagonal: Red KDE Filled Density Distribution Curves
    - Lower Triangle: Scatter Plots
    - Upper Triangle: Dynamically Scaled & Sized Correlation Squares
    """
    sub_df = df[selected_cols].apply(pd.to_numeric, errors="coerce").dropna()
    n_vars = len(selected_cols)

    corr_matrix = sub_df.corr(method="pearson")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_bwr", ["#2b5c8f", "#f7f7f7", "#d73027"]
    )

    # Scale overall figure dimensions so 14+ variables remain readable
    cell_size = max(1.8, min(3.0, 24.0 / n_vars))
    fig_size = cell_size * n_vars

    fig, axes = plt.subplots(n_vars, n_vars, figsize=(fig_size, fig_size))
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    clean_labels = [
        col.replace("isovist_", "").replace("_", " ").title()
        for col in selected_cols
    ]

    font_size = max(5, int(13 - 0.45 * n_vars))
    text_val_size = max(6, int(14 - 0.55 * n_vars))

    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j] if n_vars > 1 else axes
            col_x = selected_cols[j]
            col_y = selected_cols[i]

            # --- DIAGONAL: Red Filled KDE Density Curve ---
            if i == j:
                sns.kdeplot(
                    data=sub_df[col_x],
                    ax=ax,
                    color="#d73027",
                    fill=True,
                    alpha=0.35,
                    linewidth=1.2,
                )
                ax.set_ylabel("")
                ax.set_xlabel("")
                ax.set_xlim(sub_df[col_x].min(), sub_df[col_x].max())

            # --- LOWER TRIANGLE: Scatter Plot ---
            elif i > j:
                ax.scatter(
                    sub_df[col_x],
                    sub_df[col_y],
                    alpha=0.4,
                    edgecolor="none",
                    s=max(4, int(22 - 1.1 * n_vars)),
                    color="#2c3e50",
                )

            # --- UPPER TRIANGLE: Sized Correlation Squares ---
            else:
                ax.axis("off")  # Clear axes background
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                r_val = corr_matrix.loc[col_y, col_x]

                if not np.isnan(r_val):
                    # Normalized color mapping (-1 to 1 mapped to 0 to 1)
                    norm_val = (r_val + 1) / 2
                    sq_color = cmap(norm_val)

                    abs_r = abs(r_val)
                    # Dynamic sizing: bigger |r| = bigger square
                    sq_size = 0.20 + (0.75 * abs_r)
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

                    # Text label placed exactly inside the square
                    text_str = f"{r_val:.2f}"
                    text_color = "white" if abs_r > 0.65 else "#111111"
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

            # Axis Label formatting
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
# 3. Focused Pair Inspector (Fixed Trendline Error)
# -----------------------------------------------------------------------------
def render_pair_focused_inspector(
    df: pd.DataFrame, col_x: str, col_y: str, r_val: float
):
    """Isolated high-resolution card display with numpy linear fit to prevent statsmodels error."""
    clean_x = col_x.replace("isovist_", "").replace("_", " ").title()
    clean_y = col_y.replace("isovist_", "").replace("_", " ").title()

    sub_df = df[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()

    c1, c2, c3 = st.columns([2.2, 1, 2.2])

    with c1:
        st.markdown(f"##### 📉 Scatter Plot: `{clean_y}` vs `{clean_x}`")
        fig_scatter = go.Figure()

        # Raw scatter points
        fig_scatter.add_trace(
            go.Scatter(
                x=sub_df[col_x],
                y=sub_df[col_y],
                mode="markers",
                marker=dict(color="#2b5c8f", opacity=0.6, size=6),
                name="Data Points",
            )
        )

        # Pure NumPy OLS Trendline fit (No statsmodels dependency needed!)
        if len(sub_df) > 1:
            x_vals = sub_df[col_x].values
            y_vals = sub_df[col_y].values
            m, b = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = m * x_line + b

            fig_scatter.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="#d73027", width=2),
                    name="Trendline",
                )
            )

        fig_scatter.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title=clean_x,
            yaxis_title=clean_y,
            showlegend=False,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        st.markdown("##### 🧮 Pearson Correlation")
        bg_color = (
            "#d73027"
            if r_val > 0.5
            else ("#2b5c8f" if r_val < -0.5 else "#888888")
        )
        st.markdown(
            f"""
            <div style="
                background-color: {bg_color};
                padding: 25px 15px;
                border-radius: 12px;
                text-align: center;
                color: white;
                margin-top: 15px;">
                <h1 style="margin: 0; font-size: 40px;">{r_val:.3f}</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Pearson r</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(f"##### 📊 Metrics Distribution Skew")
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=sub_df[col_x],
                name=clean_x,
                opacity=0.55,
                marker_color="#2b5c8f",
            )
        )
        fig_dist.add_trace(
            go.Histogram(
                x=sub_df[col_y],
                name=clean_y,
                opacity=0.55,
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
# 4. Main Data Extraction & Analysis
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
                f"Global Spatial Analysis ({len(selected_metrics)} Metrics)"
            )

            # Render Matplotlib matrix
            fig = plot_vga_pairs_matrix(df_global, selected_metrics)
            st.pyplot(fig)

            # Focused Inspector Tool
            st.markdown("---")
            st.subheader("🔍 Focused Pair Inspector")
            st.write(
                "Select any variable pair below to inspect their detailed Scatter Plot, Pearson score, and Skew Distributions:"
            )

            clean_options = {
                col.replace("isovist_", "").replace("_", " ").title(): col
                for col in selected_metrics
            }

            c_select1, c_select2 = st.columns(2)
            with c_select1:
                var1_lbl = st.selectbox(
                    "Select Variable X:",
                    options=list(clean_options.keys()),
                    index=0,
                )
            with c_select2:
                var2_lbl = st.selectbox(
                    "Select Variable Y:",
                    options=list(clean_options.keys()),
                    index=min(1, len(clean_options) - 1),
                )

            var1 = clean_options[var1_lbl]
            var2 = clean_options[var2_lbl]

            if var1 == var2:
                st.info("Select two different metrics to inspect.")
            else:
                pair_df = df_global[[var1, var2]].apply(
                    pd.to_numeric, errors="coerce"
                )
                r_val = pair_df.corr().iloc[0, 1]
                render_pair_focused_inspector(df_global, var1, var2, r_val)

            # Numerical table expander
            with st.expander("View Numerical Pearson Correlation Matrix Table"):
                agg_corr = df_global[selected_metrics].corr(method="pearson")
                st.dataframe(
                    agg_corr.style.background_gradient(
                        cmap="coolwarm", vmin=-1, vmax=1
                    ).format("{:.3f}"),
                    use_container_width=True,
                )
        else:
            st.warning("Please select at least **2 metrics** to plot.")

        st.markdown("---")
        if st.button("Refresh Global Repository Cache"):
            st.cache_data.clear()
            st.rerun()

# -----------------------------------------------------------------------------
# 5. Admin Management Section
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔒 Admin Portal")

admin_password = st.secrets.get("ADMIN_PASSWORD", "admin123")
input_pass = st.sidebar.text_input("Admin Password", type="password")

if input_pass == admin_password:
    st.sidebar.success("Admin Access Granted")

    st.markdown("---")
    st.header("🔑 Admin Dataset Management")

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
                    fig_single = plot_vga_pairs_matrix(
                        single_ds_metrics, single_numeric_cols
                    )
                    st.pyplot(fig_single)
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