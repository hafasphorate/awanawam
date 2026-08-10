import json
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="VGA & Crowd Metrics Correlation Analysis",
    layout="wide"
)

st.title("VGA & Crowd Metrics Pairwise Correlation Analysis")
st.write(
    "Upload your combined dataset (JSON) to compute pairwise correlations and "
    "analyze relationships across all spatial, visibility, and crowd metrics."
)

# -----------------------------------------------------------------------------
# 1. File Upload Section
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Data File (JSON)", type=["json"])


def load_and_parse_json(file):
    """Parse JSON dataset into a pandas DataFrame."""
    data = json.load(file)
    if isinstance(data, dict) and "vga_floorplan_nodes" in data:
        df = pd.DataFrame(data["vga_floorplan_nodes"])
    elif isinstance(data, dict) and "nodes" in data:
        df = pd.DataFrame(data["nodes"])
    else:
        df = pd.DataFrame(data)
    return df


# -----------------------------------------------------------------------------
# 2. Pairs Plot Matrix Rendering Engine (Dynamically Scaled)
# -----------------------------------------------------------------------------
def plot_vga_pairs_matrix(df, selected_cols):
    """
    Generates a custom pairplot matrix for all selected variables:
    - Diagonal: Density (KDE) distribution curve (Red)
    - Lower Triangle: Scatter plot of spatial/crowd data points
    - Upper Triangle: Pearson R magnitude box overlay with numeric overlay
    """
    sub_df = df[selected_cols].apply(pd.to_numeric, errors="coerce").dropna()
    n_vars = len(selected_cols)

    # Compute Pearson correlation matrix
    corr_matrix = sub_df.corr(method="pearson")

    # Diverging colormap: Blue (-1) -> White (0) -> Red (+1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_bwr", ["#2b5c8f", "#f7f7f7", "#d73027"]
    )

    # Dynamic plot dimension and font sizing based on grid size
    cell_size = max(2.0, min(3.2, 20.0 / n_vars))
    font_size = max(6, min(11, int(14 - 0.6 * n_vars)))
    text_val_size = max(7, min(13, int(15 - 0.7 * n_vars)))

    fig, axes = plt.subplots(
        n_vars, n_vars, figsize=(cell_size * n_vars, cell_size * n_vars)
    )
    plt.subplots_adjust(wspace=0.18, hspace=0.18)

    # Clean label formatting (strips prefixes and formats names)
    clean_labels = [
        col.replace("isovist_", "").replace("_", " ").title()
        for col in selected_cols
    ]

    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j] if n_vars > 1 else axes

            col_x = selected_cols[j]
            col_y = selected_cols[i]

            # -----------------------------------------------------------------
            # DIAGONAL: Variable Density (KDE) Curve
            # -----------------------------------------------------------------
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

            # -----------------------------------------------------------------
            # LOWER TRIANGLE: Scatter Plots
            # -----------------------------------------------------------------
            elif i > j:
                ax.scatter(
                    sub_df[col_x],
                    sub_df[col_y],
                    alpha=0.5,
                    edgecolor="none",
                    s=max(10, int(30 - 1.5 * n_vars)),
                    color="#2c3e50",
                )

            # -----------------------------------------------------------------
            # UPPER TRIANGLE: Pearson R Box Overlays
            # -----------------------------------------------------------------
            else:
                r_val = corr_matrix.loc[col_y, col_x]

                # Map [-1, 1] correlation value to [0, 1] color range
                norm_val = (r_val + 1) / 2 if not np.isnan(r_val) else 0.5
                sq_color = cmap(norm_val)

                # Scale box size and transparency proportional to correlation magnitude |r|
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

                # Numeric text overlay
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

            # -----------------------------------------------------------------
            # Axis Tick & Label Formatting
            # -----------------------------------------------------------------
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
# 3. Streamlit App Execution
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        df_nodes = load_and_parse_json(uploaded_file)

        # Identify numeric columns while excluding 2D/3D spatial position coordinates
        coord_or_id_cols = {"x", "y", "z", "node_id", "id", "index", "floor"}
        all_numeric_cols = [
            col
            for col in df_nodes.select_dtypes(include=[np.number]).columns
            if col.lower() not in coord_or_id_cols
        ]

        st.sidebar.header("Data Filter Settings")

        # Quick selection buttons in sidebar
        col_btn1, col_btn2 = st.sidebar.columns(2)
        if col_btn1.button("Select All"):
            st.session_state["selected_metrics"] = all_numeric_cols
        if col_btn2.button("Clear All"):
            st.session_state["selected_metrics"] = []

        if "selected_metrics" not in st.session_state:
            st.session_state["selected_metrics"] = all_numeric_cols

        # Multiselect input defaulting to ALL detected metrics (VGA + Crowd)
        selected_metrics = st.sidebar.multiselect(
            "Select Metrics for Correlation Matrix:",
            options=all_numeric_cols,
            default=st.session_state["selected_metrics"],
            key="metric_selector",
        )

        if len(selected_metrics) < 2:
            st.warning("Please select at least **2 metrics** to generate the matrix.")
        else:
            st.subheader(f"Correlation Matrix ({len(selected_metrics)} Metrics Analyzed)")
            
            # Render custom pairplot
            fig = plot_vga_pairs_matrix(df_nodes, selected_metrics)
            st.pyplot(fig)

            # Display numeric correlation values in tabular view
            with st.expander("View Numerical Pearson Correlation Matrix Table"):
                corr_df = df_nodes[selected_metrics].corr(method="pearson")
                st.dataframe(
                    corr_df.style.background_gradient(
                        cmap="coolwarm", vmin=-1, vmax=1
                    ).format("{:.3f}")
                )

            # Raw Data Table View
            with st.expander("View Full Dataset Table"):
                st.dataframe(df_nodes)

    except Exception as e:
        st.error(f"Error parsing JSON file: {e}")
else:
    st.info("👆 Please upload your VGA & Crowd `.json` file to run analysis.")