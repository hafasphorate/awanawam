import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
# 3. Main Data Extraction & Analysis
# -----------------------------------------------------------------------------
raw_db_df = fetch_aggregated_records()

if raw_db_df.empty:
    st.warning(
        "No aggregated data found in the cloud repository yet. Upload records first."
    )
else:
    # Extract embedded metrics_data into a flat DataFrame
    metrics_list = raw_db_df["metrics_data"].tolist()
    df_global = pd.DataFrame(metrics_list)

    st.success(
        f"**Repository Active:** Loaded **{len(df_global)}** global node records."
    )

    # Exclude basic index / coordinate columns from metrics selection
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
                f"Global Correlation Matrix ({len(selected_metrics)} Metrics)"
            )

            # Generate and display custom matrix figure
            fig = plot_vga_pairs_matrix(df_global, selected_metrics)
            st.pyplot(fig)

            # Expander for exact numerical table
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

        # Cache clear / Refresh button
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
        # Group records by upload batch metadata
        meta_cols = [
            "upload_batch_id",
            "location",
            "date",
            "day_of_week",
            "time",
            "comments",
        ]
        existing_meta_cols = [
            c for c in meta_cols if c in raw_db_df.columns
        ]

        # Summarize batches
        batch_summary = (
            raw_db_df.groupby(existing_meta_cols, dropna=False)
            .size()
            .reset_index(name="node_count")
        )

        st.subheader("Uploaded Datasets Overview")
        st.dataframe(batch_summary, use_container_width=True)

        # Batch Selection Dropdown for Inspection & Deletion
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

            # Parse and view matrix for this specific dataset
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

            # Delete Batch Action
            if st.button(
                "❌ Delete Selected Dataset Batch", type="primary"
            ):
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