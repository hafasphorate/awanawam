import numpy as np
import pandas as pd


def fisher_average_batch_correlations(
    data: pd.DataFrame,
    metric_columns: list[str],
    batch_column: str | list[str] = "upload_batch_id",
    min_observations: int = 3,
) -> pd.DataFrame:
    """Average within-group Pearson correlations equally using Fisher z values."""
    batch_columns = [batch_column] if isinstance(batch_column, str) else batch_column
    missing_columns = [column for column in batch_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing batch identifier columns: {missing_columns}")

    correlations = pd.DataFrame(
        np.nan, index=metric_columns, columns=metric_columns, dtype=float
    )
    for metric in metric_columns:
        correlations.loc[metric, metric] = 1.0
    numeric_data = data[metric_columns].apply(pd.to_numeric, errors="coerce")
    batch_groups = [
        numeric_data.iloc[positions]
        for positions in data.groupby(
            batch_columns, sort=False, dropna=True
        ).indices.values()
    ]

    for metric_index, row_metric in enumerate(metric_columns):
        for column_metric in metric_columns[:metric_index]:
            fisher_z_values = []
            for batch_data in batch_groups:
                paired = batch_data[[row_metric, column_metric]].dropna()
                if len(paired) < min_observations:
                    continue
                if paired[row_metric].nunique() < 2 or paired[column_metric].nunique() < 2:
                    continue

                correlation = paired[row_metric].corr(paired[column_metric])
                if not np.isfinite(correlation):
                    continue
                bounded_correlation = np.clip(
                    correlation, -1.0 + 1e-12, 1.0 - 1e-12
                )
                fisher_z_values.append(np.arctanh(bounded_correlation))

            if fisher_z_values:
                average_correlation = np.tanh(np.mean(fisher_z_values))
                correlations.loc[row_metric, column_metric] = average_correlation
                correlations.loc[column_metric, row_metric] = average_correlation

    return correlations
