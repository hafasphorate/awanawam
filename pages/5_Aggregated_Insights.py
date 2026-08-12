import pandas as pd
import streamlit as st
from supabase import Client, create_client

st.set_page_config(page_title="Aggregated Insights", layout="wide")

st.title("🌐 Global Aggregated Spatial & Crowd Insights")
st.write(
    "This page synthesizes all user-contributed node data stored in the central database "
    "to compute collective correlation trends."
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
    st.error("Supabase credentials missing in secrets.toml configuration.")
    st.stop()


@st.cache_data(ttl=600)  # Refresh cache every 10 mins
def fetch_aggregated_data():
    """Fetch all stored metric records across all user uploads."""
    response = (
        supabase.table("vga_crowd_records").select("metrics_data").execute()
    )
    if response.data:
        # Extract nested metrics JSON from each row into a DataFrame
        extracted = [row["metrics_data"] for row in response.data]
        return pd.DataFrame(extracted)
    return pd.DataFrame()


df_global = fetch_aggregated_data()

if df_global.empty:
    st.warning(
        "No aggregated data found in the cloud repository yet. Run Page 4 to upload records."
    )
else:
    st.success(
        f"**Repository Active:** Loaded **{len(df_global)}** global node records."
    )

    numeric_cols = df_global.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error(
            "Insufficient numeric metrics in the database to form a matrix."
        )
    else:
        selected_metrics = st.multiselect(
            "Select Metrics for Aggregated Analysis:",
            options=numeric_cols,
            default=numeric_cols,
        )

        if len(selected_metrics) >= 2:
            st.subheader("Global Pearson Correlation Matrix")

            agg_corr = df_global[selected_metrics].corr(method="pearson")

            st.dataframe(
                agg_corr.style.background_gradient(
                    cmap="coolwarm", vmin=-1, vmax=1
                ).format("{:.3f}"),
                use_container_width=True,
            )

            # Option to clean/refresh data
            if st.button("Refresh Global Data"):
                st.cache_data.clear()
                st.rerun()