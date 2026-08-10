import json
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Set Streamlit page layout
st.set_page_config(
    page_title="VGA & Crowd Metrics Correlation Analysis", layout="wide"
)

st.title("VGA Metrics Pairwise Correlation Analysis")
st.write(
    "Upload your VGA JSON dataset to compute pairwise correlations and visualize "
    "the relationship between space syntax metrics."
)

# -----------------------------------------------------------------------------
# 1. File Upload Section
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload VGA JSON File", type=["json"])


def load_and_parse_json(file):
    """Parse VGA JSON data into a pandas DataFrame."""
    data = json.load(file)
    if "vga_floorplan_nodes" in data:
        df = pd.DataFrame(data["vga_floorplan_nodes"])
    else:
        # Fallback if top-level list or different key structure
        df = pd.DataFrame(data)
    return df


# -----------------------------------------------------------------------------
# 2. Pairs Plot Matrix Rendering Engine
# -----------------------------------------------------------------------------
def plot_vga_pairs_matrix(df, selected_cols):
    """Generates a custom pairplot:

    - Diagonal: Red Density (KDE) curve
    - Lower Triangle: Scatter Plot
    - Upper Triangle: Square Heatmap with Pearson R overlay
    """
    sub_df = df[selected_cols].dropna()
    n_vars = len(selected_cols)

    # Calculate Pearson Correlation Matrix
    corr_matrix = sub_df.corr(method="pearson")

    # Custom Diverging Colormap (Blue = -1, White = 0, Red = 1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_bwr", ["#2b5c8f", "#f7f7f7", "#d73027"]
    )

    fig, axes = plt.subplots(
        n_vars, n_vars, figsize=(3.2 * n_vars, 3.2 * n_vars)
    )
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    # Format axis tick labels clean display name
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
            # DIAGONAL: Red Density Curve
            # -----------------------------------------------------------------
            if i == j:
                sns.kdeplot(
                    data=sub_df[col_x],
                    ax=ax,
                    color="#d73027",
                    fill=True,
                    alpha=0.25,
                    linewidth=2,
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
                    alpha=0.6,
                    edgecolor="none",
                    s=25,
                    color="#333333",
                )

            # -----------------------------------------------------------------
            # UPPER TRIANGLE: Pearson R Squares with Opacity Scaling
            # -----------------------------------------------------------------
            else:
                r_val = corr_matrix.loc[col_y, col_x]

                # Map [-1, 1] correlation value to colormap [0, 1]
                norm_val = (r_val + 1) / 2
                sq_color = cmap(norm_val)

                # Size & Opacity proportional to magnitude |r|
                abs_r = abs(r_val) if not np.isnan(r_val) else 0
                sq_size = 0.2 + (0.75 * abs_r)  # Box dimension scale
                alpha_val = 0.3 + (0.7 * abs_r)  # Box transparency

                # Draw Correlation Square
                rect = plt.Rectangle(
                    (0.5 - sq_size / 2, 0.5 - sq_size / 2),
                    sq_size,
                    sq_size,
                    facecolor=sq_color,
                    alpha=alpha_val,
                    edgecolor="none",
                )
                ax.add_patch(rect)

                # Overlay Text Value
                text_str = f"{r_val:.2f}" if not np.isnan(r_val) else "N/A"
                text_color = "white" if abs_r > 0.65 else "black"
                ax.text(
                    0.5,
                    0.5,
                    text_str,
                    ha="center",
                    va="center",
                    fontsize=12 + n_vars,
                    weight="bold",
                    color=text_color,
                )

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")

            # -----------------------------------------------------------------
            # Subplot Axis Formatting & Ticks
            # -----------------------------------------------------------------
            if i < n_vars - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(clean_labels[j], fontsize=10, fontweight="bold")

            if j > 0 and i != j:
                ax.set_yticklabels([])
            if j == 0:
                ax.set_ylabel(clean_labels[i], fontsize=10, fontweight="bold")

            ax.tick_params(labelsize=8)

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 3. Streamlit Workflow Execution
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        df_nodes = load_and_parse_json(uploaded_file)

        # Exclude positional spatial coordinates from automatic selection
        numeric_cols = df_nodes.select_dtypes(include=[np.number]).columns.tolist()
        default_vga_cols = [
            col for col in numeric_cols if col not in ["x", "y"]
        ]

        st.sidebar.header("Plot Configurations")
        selected_metrics = st.sidebar.multiselect(
            "Select Metrics to Correlate:",
            options=numeric_cols,
            default=(
                default_vga_cols[:5]
                if len(default_vga_cols) >= 5
                else default_vga_cols
            ),
        )

        if len(selected_metrics) < 2:
            st.warning("Please select at least two metrics to plot pairs.")
        else:
            st.subheader("Pairwise Matrix & Pearson Correlation Plot")
            fig = plot_vga_pairs_matrix(df_nodes, selected_metrics)
            st.pyplot(fig)

            # Optional: Display raw data preview inside an expander
            with st.expander("View Uploaded Raw VGA Nodes Data"):
                st.dataframe(df_nodes)

    except Exception as e:
        st.error(f"Error parsing JSON dataset: {e}")
else:
    st.info("👆 Please upload a `.json` file to begin analysis.")