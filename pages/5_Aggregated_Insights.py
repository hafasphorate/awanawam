import io
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from supabase import Client, create_client
from utils.correlation_metrics import fisher_average_batch_correlations
from utils.dataset_locks import is_locked, unlocked_records
from utils.navigation import render_home_button

st.set_page_config(page_title="Module 5: Aggregated Insights", layout="wide")
render_home_button()

st.title("Module 5: Aggregated Insights")
st.write(
    "This page synthesizes all user-contributed node data stored in the central database "
    "to compute collective correlation trends."
)

DARK_THEME = st.get_option("theme.base") == "dark"
PLOTLY_TEMPLATE = "plotly_dark" if DARK_THEME else "plotly_white"
FIGURE_BACKGROUND = "#15191f" if DARK_THEME else "#ffffff"
AXIS_BACKGROUND = "#1e242c" if DARK_THEME else "#ffffff"
TEXT_COLOR = "#f3f4f6" if DARK_THEME else "#20242a"
SPINE_COLOR = "#626b78" if DARK_THEME else "#a7adb5"


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


@st.cache_data  # The full table is refreshed explicitly to avoid repeated bulk reads.
def fetch_aggregated_records():
    """Fetch all raw records including metadata from Supabase."""
    page_size = 1000
    offset = 0
    records = []

    while True:
        response = (
            supabase.table("vga_crowd_records")
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        page = response.data or []
        records.extend(page)

        if len(page) < page_size:
            break
        offset += page_size

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 2. Matplotlib Precision Pairs Matrix Engine (High-Res Ready)
# -----------------------------------------------------------------------------
def add_log_density_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p-transformed crowd and peak density metrics when available."""
    normalized_columns = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in df.columns
    }
    crowd_density_col = normalized_columns.get(
        "crowd_density", normalized_columns.get("density")
    )
    peak_density_col = normalized_columns.get("peak_density")

    if crowd_density_col is not None:
        df["log_crowd_density"] = np.log1p(
            pd.to_numeric(df[crowd_density_col], errors="coerce")
        )
    if peak_density_col is not None:
        df["log_peak_density"] = np.log1p(
            pd.to_numeric(df[peak_density_col], errors="coerce")
        )

    return df


def exclude_zero_analysis_rows(df: pd.DataFrame) -> pd.DataFrame:
    zero_filter_metrics = {"crowd_density", "density", "peak_density", "speed"}
    zero_filter_cols = [
        column
        for column in df.columns
        if str(column).strip().lower().replace(" ", "_") in zero_filter_metrics
    ]

    if not zero_filter_cols:
        return df

    has_zero_metric = pd.Series(False, index=df.index)
    for column in zero_filter_cols:
        has_zero_metric |= pd.to_numeric(df[column], errors="coerce").eq(0)

    return df.loc[~has_zero_metric]


def plot_vga_pairs_matrix(df: pd.DataFrame, selected_cols: list, dpi_val: int = 100):
    """
    Recreates the exact visual matrix:
    - Diagonal: Red KDE Density Curves
    - Lower Triangle: Scatter Plots
    - Upper Triangle: Dynamically Scaled Correlation Squares
    """
    matrix_data = exclude_zero_analysis_rows(df)
    sub_df = matrix_data[selected_cols].apply(pd.to_numeric, errors="coerce").dropna()
    n_vars = len(selected_cols)

    corr_matrix = sub_df.corr(method="pearson")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_bwr", ["#2b5c8f", "#f7f7f7", "#d73027"]
    )

    cell_size = max(2.2, min(3.5, 26.0 / n_vars))
    fig_size = cell_size * n_vars

    fig, axes = plt.subplots(n_vars, n_vars, figsize=(fig_size, fig_size), dpi=dpi_val)
    fig.patch.set_facecolor(FIGURE_BACKGROUND)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    clean_labels = [
        col.replace("isovist_", "").replace("_", " ").title()
        for col in selected_cols
    ]

    font_size = max(6, int(14 - 0.45 * n_vars))
    text_val_size = max(7, int(15 - 0.55 * n_vars))

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
                    s=max(6, int(26 - 1.1 * n_vars)),
                    color="#2c3e50",
                )

            # --- UPPER TRIANGLE: Sized Correlation Squares ---
            else:
                ax.axis("off")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                r_val = corr_matrix.loc[col_y, col_x]

                if not np.isnan(r_val):
                    r_val = np.clip(r_val, -1.0, 1.0)
                    norm_val = (r_val + 1) / 2
                    sq_color = cmap(norm_val)

                    abs_r = abs(r_val)
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

            if i < n_vars - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(clean_labels[j], fontsize=font_size, fontweight="bold")
                ax.tick_params(axis="x", rotation=45, labelsize=font_size - 1)

            if j > 0 and i != j:
                ax.set_yticklabels([])
            if j == 0:
                ax.set_ylabel(clean_labels[i], fontsize=font_size, fontweight="bold")
                ax.tick_params(axis="y", labelsize=font_size - 1)

            ax.set_facecolor(AXIS_BACKGROUND)
            ax.tick_params(colors=TEXT_COLOR)
            ax.xaxis.label.set_color(TEXT_COLOR)
            ax.yaxis.label.set_color(TEXT_COLOR)
            for spine in ax.spines.values():
                spine.set_color(SPINE_COLOR)

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 3. Focused Pair Inspector (Expanded Graph + Non-Linear Trendlines)
# -----------------------------------------------------------------------------
def render_pair_focused_inspector(df: pd.DataFrame, col_x: str, col_y: str):
    """Large scatter plot with both linear (Pearson) and non-linear/monotonic (Spearman) trendlines."""
    clean_x = col_x.replace("isovist_", "").replace("_", " ").title()
    clean_y = col_y.replace("isovist_", "").replace("_", " ").title()

    sub_df = df[[col_x, col_y]].apply(pd.to_numeric, errors="coerce").dropna()

    # Calculate all three correlation metrics
    p_r = sub_df.corr(method="pearson").iloc[0, 1]
    s_rho = sub_df.corr(method="spearman").iloc[0, 1]
    k_tau = sub_df.corr(method="kendall").iloc[0, 1]

    c1, c2 = st.columns([1.6, 1])

    with c1:
        st.markdown(f"#####  Pair Scatter Analysis: `{clean_y}` vs `{clean_x}`")
        fig_scatter = go.Figure()

        # Scatter points
        fig_scatter.add_trace(
            go.Scatter(
                x=sub_df[col_x],
                y=sub_df[col_y],
                mode="markers",
                marker=dict(color="#2b5c8f", opacity=0.5, size=7),
                name="Data Points",
            )
        )

        if len(sub_df) > 2:
            x_vals = sub_df[col_x].values
            y_vals = sub_df[col_y].values
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)

            # 1. Linear OLS Fit (Pearson r)
            m, b = np.polyfit(x_vals, y_vals, 1)
            y_linear = m * x_line + b

            fig_scatter.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_linear,
                    mode="lines",
                    line=dict(color="#d73027", width=2.5),
                    name="Linear Trend (Pearson)",
                )
            )

            # 2. Polynomial 2nd Degree Curved Fit (Monotonic / Spearman curve alignment)
            p_coefs = np.polyfit(x_vals, y_vals, 2)
            p_func = np.poly1d(p_coefs)
            y_curved = p_func(x_line)

            fig_scatter.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_curved,
                    mode="lines",
                    line=dict(color="#2b5c8f", width=2.5, dash="dash"),
                    name="Curved/Monotonic Fit (Spearman/Kendall)",
                )
            )

        fig_scatter.update_layout(
            height=480,
            template=PLOTLY_TEMPLATE,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title=clean_x,
            yaxis_title=clean_y,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        st.markdown("#####  Statistical Relationship Comparison")

        # Metric Display Cards
        st.markdown(
            f"""
            <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                <div style="background: #f8f9fa; border-left: 5px solid #d73027; padding: 12px; flex: 1; border-radius: 4px;">
                    <span style="font-size: 12px; color: #666;">Pearson (r)</span>
                    <h2 style="margin:0; color:#111;">{p_r:.3f}</h2>
                </div>
                <div style="background: #f8f9fa; border-left: 5px solid #2b5c8f; padding: 12px; flex: 1; border-radius: 4px;">
                    <span style="font-size: 12px; color: #666;">Spearman (ρ)</span>
                    <h2 style="margin:0; color:#111;">{s_rho:.3f}</h2>
                </div>
                <div style="background: #f8f9fa; border-left: 5px solid #27ae60; padding: 12px; flex: 1; border-radius: 4px;">
                    <span style="font-size: 12px; color: #666;">Kendall (τ)</span>
                    <h2 style="margin:0; color:#111;">{k_tau:.3f}</h2>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(" Understanding these metrics & trendlines", expanded=True):
            st.markdown(
                """
                * **Red Solid Line — Pearson ($r$):** Fits a straight-line model. Evaluates pure **linear** proportionality.
                * **Blue Dashed Line — Spearman ($\rho$) & Kendall ($\tau$):** Fits a non-linear polynomial curve to track monotonic trends (data bending smoothly upwards/downwards).
                * **Comparing metrics:** If Spearman ($\rho$) is higher than Pearson ($r$), the relationship is strong but **curved** rather than strictly straight.
                """
            )


# -----------------------------------------------------------------------------
# 4. Multivariate Feature Driver Regression Analysis (Linear / Ridge / Lasso)
# -----------------------------------------------------------------------------
def render_multivariate_regression(df: pd.DataFrame, numeric_cols: list):
    """Pure NumPy implementation of Standardized Linear, Ridge, and Lasso Regression."""
    st.markdown("---")
    st.subheader(" Drivers Analysis (Multivariate Regression)")
    st.write(
        "Identify which specific spatial properties exert the strongest influence on a key target metric. "
        "Variables are **$Z$-score standardized** so coefficient magnitudes can be directly compared."
    )

    clean_options = {
        col.replace("isovist_", "").replace("_", " ").title(): col
        for col in numeric_cols
    }

    reg_col1, reg_col2, reg_col3 = st.columns([2, 2, 1])

    with reg_col1:
        target_lbl = st.selectbox(
            "Select Target Metric (Dependent Variable Y):",
            options=list(clean_options.keys()),
            index=0,
        )
    target_var = clean_options[target_lbl]

    feature_cols = [c for c in numeric_cols if c != target_var]

    with reg_col2:
        selected_features = st.multiselect(
            "Select Predictor Properties (X):",
            options=[
                k for k, v in clean_options.items() if v in feature_cols
            ],
            default=[
                k for k, v in clean_options.items() if v in feature_cols
            ],
        )

    with reg_col3:
        model_type = st.selectbox(
            "Regression Model:",
            options=["Standard OLS", "Ridge (L2)", "Lasso (L1)"],
        )

    if not selected_features:
        st.warning("Select at least one predictor feature to build the model.")
        return

    chosen_feature_vars = [clean_options[k] for k in selected_features]

    # Prepare Data
    reg_df = df[[target_var] + chosen_feature_vars].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(reg_df) < len(chosen_feature_vars) + 1:
        st.error("Not enough valid data points to perform regression.")
        return

    Y = reg_df[target_var].values
    X_raw = reg_df[chosen_feature_vars].values

    # Z-score standardization: (X - mu) / sigma
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1.0  # avoid division by zero
    X_scaled = (X_raw - X_mean) / X_std

    Y_mean = Y.mean()
    Y_std = Y.std() if Y.std() != 0 else 1.0
    Y_scaled = (Y - Y_mean) / Y_std

    # Add Intercept
    N = len(Y_scaled)
    X_design = np.hstack([np.ones((N, 1)), X_scaled])

    alpha = 1.0  # Regularization parameter
    coefs = None

    if model_type == "Standard OLS":
        # (X^T X)^(-1) X^T Y
        try:
            weights = np.linalg.lstsq(X_design, Y_scaled, rcond=None)[0]
            coefs = weights[1:]
        except np.linalg.LinAlgError:
            st.error("Matrix singular. Try removing collinear features.")
            return

    elif model_type == "Ridge (L2)":
        # (X^T X + alpha*I)^(-1) X^T Y
        I = np.eye(X_design.shape[1])
        I[0, 0] = 0  # Do not penalize intercept
        try:
            weights = np.linalg.inv(X_design.T @ X_design + alpha * I) @ X_design.T @ Y_scaled
            coefs = weights[1:]
        except np.linalg.LinAlgError:
            st.error("Computation error during Ridge estimation.")
            return

    elif model_type == "Lasso (L1)":
        # Iterative Coordinate Descent for Lasso
        w = np.zeros(X_design.shape[1])
        for _ in range(200):
            for j in range(X_design.shape[1]):
                X_j = X_design[:, j]
                y_pred = X_design @ w
                r = Y_scaled - y_pred + w[j] * X_j
                rho = np.dot(X_j, r)
                if j == 0:
                    w[j] = rho / N
                else:
                    # Soft thresholding
                    lam = alpha * 0.1
                    if rho < -lam:
                        w[j] = (rho + lam) / N
                    elif rho > lam:
                        w[j] = (rho - lam) / N
                    else:
                        w[j] = 0.0
        coefs = w[1:]

    # Calculate R-Squared
    y_pred_scaled = X_design @ np.hstack(
        [
            [
                Y_scaled.mean()
                if model_type != "Standard OLS"
                else weights[0]
            ],
            coefs,
        ]
    )
    ss_res = np.sum((Y_scaled - y_pred_scaled) ** 2)
    ss_tot = np.sum((Y_scaled - Y_scaled.mean()) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Display Results
    res_df = pd.DataFrame(
        {
            "Property Feature": selected_features,
            "Standardized Coefficient Weight": coefs,
            "Impact Direction": [
                "Positive (+)" if c > 0 else ("Negative (-)" if c < 0 else "Neutral")
                for c in coefs
            ],
            "Absolute Importance": np.abs(coefs),
        }
    ).sort_values(by="Absolute Importance", ascending=False)

    rc1, rc2 = st.columns([1.5, 1])

    with rc1:
        fig_bar = px.bar(
            res_df,
            x="Standardized Coefficient Weight",
            y="Property Feature",
            orientation="h",
            color="Standardized Coefficient Weight",
            color_continuous_scale="RdBu_r",
            title=f"Relative Driver Weights on '{target_lbl}' (R² = {r2_score:.3f})",
        )
        fig_bar.update_layout(
            height=380,
            template=PLOTLY_TEMPLATE,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with rc2:
        st.markdown("#####  Ranked Feature Impact")
        st.dataframe(
            res_df[
                [
                    "Property Feature",
                    "Standardized Coefficient Weight",
                    "Impact Direction",
                ]
            ].style.format({"Standardized Coefficient Weight": "{:.3f}"}),
            use_container_width=True,
            height=320,
        )


def render_mediation_analysis(df: pd.DataFrame, numeric_cols: list):
    """Estimate a simple mediation model with a percentile bootstrap interval."""
    st.markdown("---")
    st.subheader(" Mediation Analysis")
    st.write(
        "Test whether a mediator helps explain the relationship between a predictor "
        "and an outcome. The indirect effect is the product of paths $a$ and $b$."
    )

    clean_options = {
        col.replace("isovist_", "").replace("_", " ").title(): col
        for col in numeric_cols
    }
    labels = list(clean_options.keys())
    x_col, mediator_col, y_col = st.columns(3)

    with x_col:
        predictor_lbl = st.selectbox("Predictor (X):", labels, key="mediation_predictor")
    with mediator_col:
        mediator_lbl = st.selectbox("Mediator (M):", labels, index=min(1, len(labels) - 1), key="mediation_mediator")
    with y_col:
        outcome_lbl = st.selectbox("Outcome (Y):", labels, index=min(2, len(labels) - 1), key="mediation_outcome")

    predictor = clean_options[predictor_lbl]
    mediator = clean_options[mediator_lbl]
    outcome = clean_options[outcome_lbl]
    if len({predictor, mediator, outcome}) < 3:
        st.info("Select three different metrics for the predictor, mediator, and outcome.")
        return

    mediation_df = df[[predictor, mediator, outcome]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(mediation_df) < 10:
        st.warning("At least 10 complete records are recommended for mediation analysis.")
        return
    if any(mediation_df[column].nunique() < 2 for column in [predictor, mediator, outcome]):
        st.warning("Each selected metric needs more than one distinct value.")
        return

    x = mediation_df[predictor].to_numpy()
    m = mediation_df[mediator].to_numpy()
    y = mediation_df[outcome].to_numpy()

    def ols(response, design):
        return np.linalg.lstsq(
            np.column_stack([np.ones(len(response)), design]), response, rcond=None
        )[0]

    a = ols(m, x[:, None])[1]
    total_effect = ols(y, x[:, None])[1]
    direct_effect = ols(y, np.column_stack([x, m]))[1]
    b = ols(y, np.column_stack([x, m]))[2]
    indirect_effect = a * b

    rng = np.random.default_rng(42)
    bootstrap_effects = np.empty(1000)
    for index in range(len(bootstrap_effects)):
        sample_indices = rng.integers(0, len(mediation_df), len(mediation_df))
        sample_x = x[sample_indices]
        sample_m = m[sample_indices]
        sample_y = y[sample_indices]
        bootstrap_a = ols(sample_m, sample_x[:, None])[1]
        bootstrap_b = ols(sample_y, np.column_stack([sample_x, sample_m]))[2]
        bootstrap_effects[index] = bootstrap_a * bootstrap_b
    lower_ci, upper_ci = np.percentile(bootstrap_effects, [2.5, 97.5])

    results = pd.DataFrame(
        {
            "Effect": ["Path a (X → M)", "Path b (M → Y | X)", "Total effect c", "Direct effect c'", "Indirect effect a × b"],
            "Estimate": [a, b, total_effect, direct_effect, indirect_effect],
            "95% Bootstrap CI": ["", "", "", "", f"[{lower_ci:.4f}, {upper_ci:.4f}]"],
        }
    )
    st.dataframe(
        results.style.format({"Estimate": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    if lower_ci <= 0 <= upper_ci:
        st.info(
            "The bootstrap interval for the indirect effect includes zero; "
            "the mediated effect is not statistically significant at the 95% level."
        )
    else:
        direction = "positive" if indirect_effect > 0 else "negative"
        st.success(
            "The bootstrap interval for the indirect effect excludes zero; "
            f"the {direction} mediated effect is statistically significant at the 95% level."
        )

    # Draw one triangle with X, M, and Y as distinct nodes.
    node_x = [0, 1, 2]
    node_y = [0, 1.2, 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[node_x[0], node_x[1]], y=[node_y[0], node_y[1]],
        mode="lines", line=dict(color="#2b5c8f", width=3), name="Path a",
    ))
    fig.add_trace(go.Scatter(
        x=[node_x[1], node_x[2]], y=[node_y[1], node_y[2]],
        mode="lines", line=dict(color="#d73027", width=3), name="Path b",
    ))
    fig.add_trace(go.Scatter(
        x=[node_x[0], node_x[2]], y=[node_y[0] - 0.08, node_y[2] - 0.08],
        mode="lines", line=dict(color="#27ae60", width=3), name="Total effect c",
    ))
    fig.add_trace(go.Scatter(
        x=[node_x[0], node_x[2]], y=[node_y[0] + 0.08, node_y[2] + 0.08],
        mode="lines", line=dict(color="#8e44ad", width=2, dash="dash"), name="Direct effect c'",
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[predictor_lbl, mediator_lbl, outcome_lbl],
        textposition=["middle left", "top center", "middle right"],
        marker=dict(size=18, color=["#2b5c8f", "#f39c12", "#d73027"], line=dict(width=2, color="white")),
        name="Variables",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=360, showlegend=False, xaxis=dict(visible=False, range=[-0.35, 2.35]),
        yaxis=dict(visible=False, range=[-0.3, 1.55]), margin=dict(l=20, r=20, t=20, b=20),
        annotations=[
            dict(x=0.5, y=0.68, text=f"a = {a:.3f}", showarrow=False, font=dict(color="#2b5c8f")),
            dict(x=1.5, y=0.68, text=f"b = {b:.3f}", showarrow=False, font=dict(color="#d73027")),
            dict(x=1.0, y=-0.16, text=f"c = {total_effect:.3f}", showarrow=False, font=dict(color="#27ae60")),
            dict(x=1.0, y=0.16, text=f"c' = {direct_effect:.3f}", showarrow=False, font=dict(color="#8e44ad")),
        ],
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# 5. Main Application Workflow
# -----------------------------------------------------------------------------
raw_db_df = fetch_aggregated_records()
if st.sidebar.button("Refresh Global Repository Cache"):
    st.cache_data.clear()
    st.rerun()

unlocked_db_df = unlocked_records(raw_db_df)
location_values = (
    sorted(
        {
            str(location).strip()
            for location in unlocked_db_df.get("location", pd.Series(dtype=object)).dropna()
            if str(location).strip()
        },
        key=str.casefold,
    )
    if not unlocked_db_df.empty
    else []
)
selected_location = st.sidebar.selectbox(
    "Filter by mall/location",
    options=["All locations", *location_values],
)
dataset_labels = {}
dataset_ids = []
if "upload_batch_id" in unlocked_db_df.columns:
    for _, record in unlocked_db_df.dropna(subset=["upload_batch_id"]).drop_duplicates(
        "upload_batch_id"
    ).iterrows():
        batch_id = str(record["upload_batch_id"])
        dataset_ids.append(batch_id)
        dataset_labels[batch_id] = (
            f"{batch_id[:8]}... | "
            f"{record.get('location', 'N/A')} | "
            f"{record.get('date', 'N/A')} | "
            f"{record.get('time', 'N/A')}"
        )

selected_dataset_ids = st.sidebar.multiselect(
    "Or select specific datasets",
    options=dataset_ids,
    format_func=lambda batch_id: dataset_labels[batch_id],
    help="Selecting one or more datasets takes precedence over the mall/location filter above.",
)
if selected_dataset_ids:
    active_db_df = unlocked_db_df[
        unlocked_db_df["upload_batch_id"].astype(str).isin(selected_dataset_ids)
    ]
else:
    active_db_df = (
        unlocked_db_df
        if selected_location == "All locations"
        else unlocked_db_df[unlocked_db_df["location"] == selected_location]
    )
coord_or_id_cols = {"x", "y", "z", "node_id", "id", "index", "floor"}

if active_db_df.empty:
    if selected_dataset_ids:
        st.info("No unlocked records found for the selected datasets.")
    elif not unlocked_db_df.empty and selected_location != "All locations":
        st.info(f"No unlocked records found for **{selected_location}**.")
    elif not raw_db_df.empty:
        st.info("All uploaded datasets are temporarily locked. Unlock a dataset in the Admin Portal to include it in analysis.")
    else:
        st.warning("No aggregated data found in the cloud repository yet. Upload records first.")
else:
    metrics_list = active_db_df["metrics_data"].tolist()
    df_global = pd.DataFrame(metrics_list)
    df_global = exclude_zero_analysis_rows(df_global)
    df_global = add_log_density_metrics(df_global)
    analysis_db_df = active_db_df.iloc[df_global.index].reset_index(drop=True)
    df_global = df_global.reset_index(drop=True)
    dataset_count = (
        active_db_df["upload_batch_id"].nunique()
        if "upload_batch_id" in active_db_df.columns
        else 0
    )

    st.success(
        f"**Repository Active:** Loaded **{len(df_global)}** analysis records after excluding rows "
        f"with zero crowd density, peak density, or speed "
        f"(from **{len(metrics_list)}** unlocked node records) "
        f"from **{dataset_count}** data sets."
    )

    numeric_cols = [
        col
        for col in df_global.columns
        if pd.api.types.is_numeric_dtype(df_global[col])
        and col.lower() not in coord_or_id_cols
    ]

    if df_global.empty:
        st.warning(
            "No rows remain after excluding records with zero crowd density, peak density, or speed."
        )
    elif len(numeric_cols) < 2:
        st.error("Insufficient numeric metrics in the database to form a matrix.")
    else:
        crowd_words = (
            "crowd", "pedestrian", "density", "people", "count", "volume",
            "person", "frame", "speed", "direction", "deviation", "occupancy",
            "trajectory", "spine",
        )
        crowd_metrics = [
            column for column in numeric_cols
            if any(word in column.lower() for word in crowd_words)
        ]
        default_crowd_selection = list(
            st.session_state.get("global_crowd_selection", crowd_metrics)
        )
        default_crowd_selection.extend(
            metric
            for metric in ("log_crowd_density", "log_peak_density")
            if metric in crowd_metrics and metric not in default_crowd_selection
        )
        vga_metrics = [column for column in numeric_cols if column not in crowd_metrics]

        st.sidebar.header("Global Analysis Filters")
        with st.sidebar.form("global_analysis_filters"):
            crowd_selection = st.multiselect(
                "Crowd metrics",
                options=crowd_metrics,
                default=default_crowd_selection,
                key="global_crowd_filter",
            )
            vga_selection = st.multiselect(
                "VGA metrics",
                options=vga_metrics,
                default=st.session_state.get("global_vga_selection", vga_metrics),
                key="global_vga_filter",
            )
            generate_analysis = st.form_submit_button("Generate Analysis", type="primary")

        active_batches = (
            tuple(sorted(active_db_df["upload_batch_id"].dropna().astype(str).unique()))
            if "upload_batch_id" in active_db_df.columns
            else (str(len(active_db_df)),)
        )
        if generate_analysis:
            st.session_state.global_crowd_selection = crowd_selection
            st.session_state.global_vga_selection = vga_selection
            st.session_state.global_analysis_batches = active_batches
            st.session_state.global_analysis_generated = len(set(crowd_selection + vga_selection)) >= 2

        if st.session_state.get("global_analysis_batches") != active_batches:
            st.session_state.global_analysis_generated = False

        selected_metrics = (
            st.session_state.get("global_crowd_selection", [])
            + st.session_state.get("global_vga_selection", [])
            if st.session_state.get("global_analysis_generated", False)
            else []
        )

        if len(selected_metrics) >= 2:
            st.subheader(
                f"Global Spatial Analysis ({len(selected_metrics)} Metrics)"
            )

            # Matplotlib Grid Display
            fig = plot_vga_pairs_matrix(df_global, selected_metrics, dpi_val=100)
            st.pyplot(fig, use_container_width=True)

            # High-Res Export Buffer
            buffer = io.BytesIO()
            fig_highres = plot_vga_pairs_matrix(
                df_global, selected_metrics, dpi_val=300
            )
            fig_highres.savefig(buffer, format="png", bbox_inches="tight")
            plt.close(fig_highres)

            st.download_button(
                label=" Download High-Res Matrix Image (300 DPI PNG)",
                data=buffer.getvalue(),
                file_name="spatial_correlation_matrix_300dpi.png",
                mime="image/png",
            )

            # Focused Inspector Tool
            st.markdown("---")
            st.subheader(" Focused Pair Inspector")
            st.write(
                "Select any variable pair below to inspect their detailed scatter plot and multi-correlation metrics:"
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
                render_pair_focused_inspector(df_global, var1, var2)

            # Multivariate Regression Feature Drivers Section
            render_multivariate_regression(df_global, selected_metrics)

            # Mediation Analysis Section
            render_mediation_analysis(df_global, selected_metrics)

            # Treat each mall/date combination as one independent correlation group.
            with st.expander("View Mall-and-Date-Averaged Pearson Correlations"):
                group_columns = ["location", "date"]
                if all(column in analysis_db_df.columns for column in group_columns):
                    correlation_data = df_global.copy()
                    for column in group_columns:
                        correlation_data[column] = analysis_db_df[column].to_numpy()
                    agg_corr = fisher_average_batch_correlations(
                        correlation_data,
                        selected_metrics,
                        batch_column=group_columns,
                    )
                    triangle_corr = agg_corr.copy()
                    for metric_index, metric in enumerate(selected_metrics):
                        triangle_corr.loc[
                            metric, selected_metrics[metric_index:]
                        ] = np.nan

                    st.caption(
                        "Each mall-and-date combination contributes one within-group Pearson correlation per pair, "
                        "regardless of how many upload batches contain that combination. "
                        "Coefficients are averaged equally across mall/date groups using Fisher z-transformation; "
                        "pairs with fewer than 3 complete observations or no variation in a batch are excluded."
                    )
                    st.dataframe(
                        triangle_corr.style.background_gradient(
                            cmap="coolwarm", vmin=-1, vmax=1
                        ).format("{:.3f}", na_rep=""),
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "Mall-and-date-averaged correlations are unavailable because location or date metadata is missing."
                    )
        else:
            st.info("Select at least two metrics across the Crowd and VGA filters, then press **Generate Analysis**.")

# -----------------------------------------------------------------------------
# 6. Admin Management Section
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header(" Admin Portal")

admin_password = st.secrets.get("ADMIN_PASSWORD", "admin123")
input_pass = st.sidebar.text_input("Admin Password", type="password")

if input_pass == admin_password:
    st.sidebar.success("Admin Access Granted")

    st.markdown("---")
    st.header(" Admin Dataset Management")

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
        if "is_locked" in raw_db_df.columns:
            locked_by_batch = raw_db_df.groupby("upload_batch_id")["is_locked"].agg(
                lambda values: any(is_locked(value) for value in values)
            )
            batch_summary["temporarily_locked"] = batch_summary["upload_batch_id"].map(locked_by_batch).fillna(False)

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
            "Select Dataset Batch to Inspect, Lock, or Remove:",
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
            single_ds_metrics = exclude_zero_analysis_rows(single_ds_metrics)
            single_ds_metrics = add_log_density_metrics(single_ds_metrics)
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

            batch_locked = (
                "is_locked" in batch_records.columns
                and any(is_locked(value) for value in batch_records["is_locked"])
            )
            lock_label = "Unlock Dataset Batch" if batch_locked else "Temporarily Lock Dataset Batch"
            if "is_locked" not in raw_db_df.columns:
                st.warning("Dataset locking requires applying `database/lock_datasets.sql` to Supabase first.")
            elif st.button(lock_label, key=f"toggle_dataset_lock_{selected_batch_id}"):
                try:
                    lock_res = (
                        supabase.table("vga_crowd_records")
                        .update({"is_locked": not batch_locked})
                        .eq("upload_batch_id", selected_batch_id)
                        .select("upload_batch_id")
                        .execute()
                    )
                    if lock_res.data:
                        st.cache_data.clear()
                        st.rerun()
                    st.error("No dataset rows were updated. Check Supabase update permissions.")
                except Exception as error:
                    st.error(f"Could not update dataset lock: {error}")

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