import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# 2. Plotly Interactive Matrix Rendering Functions
# -----------------------------------------------------------------------------
def render_plotly_pairs_matrix(df: pd.DataFrame, selected_cols: list):
    """Generates an interactive Plotly Scatter Matrix (Pairs Plot)."""
    sub_df = df[selected_cols].apply(pd.to_numeric, errors="coerce").dropna()

    # Create user-friendly display labels
    rename_dict = {
        col: col.replace("isovist_", "").replace("_", " ").title()
        for col in selected_cols
    }
    sub_df = sub_df.rename(columns=rename_dict)
    clean_cols = list(rename_dict.values())

    # Dynamically scale height based on column count (min 600px, max 1400px)
    calculated_height = max(600, min(1400, 75 * len(clean_cols)))

    fig = px.scatter_matrix(
        sub_df,
        dimensions=clean_cols,
        height=calculated_height,
        opacity=0.65,
    )

    # Customize plot styling
    fig.update_traces(
        diagonal_visible=True,  # Shows histograms on the diagonal axis
        showupperhalf=True,
        marker=dict(size=4, color="#2b5c8f"),
    )

    fig.update_layout(
        margin=dict(l=60, r=20, t=30, b=60),
        font=dict(size=max(8, int(13 - 0.3 * len(clean_cols)))),
        dragmode="zoom",
    )

    # Clean up axis labels rotation
    fig.for_each_xaxis(
        lambda axis: axis.update(tickangle=-45, title_font=dict(size=10))
    )
    fig.for_each_yaxis(lambda axis: axis.update(title_font=dict(size=10)))

    return fig


def render_plotly_correlation_heatmap(df: pd.DataFrame, selected_cols: list):
    """Generates an interactive Heatmap for numerical correlations."""
    sub_df = df[selected_cols].apply(pd.to_numeric, errors="coerce").dropna()
    corr_matrix = sub_df.corr(method="pearson")

    clean_labels = [
        col.replace("isovist_", "").replace("_", " ").title()
        for col in selected_cols
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=clean_labels,
            y=clean_labels,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=max(8, int(14 - 0.4 * len(clean_labels)))),
            hoverongaps=False,
        )
    )

    calculated_height = max(500, min(1000, 60 * len(clean_labels)))

    fig.update_layout(
        height=calculated_height,
        margin=dict(l=80, r=20, t=30, b=80),
        xaxis=dict(tickangle=-45),
    )

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
                f"Global Spatial Analysis ({len(selected_metrics)} Metrics)"
            )

            # Tabbed View for Pairs Plot Matrix vs. Annotated Heatmap
            tab_matrix, tab_heatmap, tab_data = st.tabs(
                [
                    "📈 Pairs Plot Matrix",
                    "🔥 Correlation Heatmap",
                    "🔢 Pearson Correlation Table",
                ]
            )

            with tab_matrix:
                st.caption(
                    "💡 *Hover over points to inspect individual spatial nodes. Click and drag on any subplot to zoom.*"
                )
                fig_pairs = render_plotly_pairs_matrix(
                    df_global, selected_metrics
                )
                st.plotly_chart(fig_pairs, use_container_width=True)

            with tab_heatmap:
                st.caption(
                    "💡 *Clean, uncluttered view showing exact linear relationship values.*"
                )
                fig_heatmap = render_plotly_correlation_heatmap(
                    df_global, selected_metrics
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

            with tab_data:
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
        existing_meta_cols = [c for c in meta_cols if c in raw_db_df.columns]

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
                    fig_single = render_plotly_pairs_matrix(
                        single_ds_metrics, single_numeric_cols
                    )
                    st.plotly_chart(fig_single, use_container_width=True)
                else:
                    st.info(
                        "Not enough numeric variables in this dataset to generate a plot."
                    )

            # Delete Batch Action
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