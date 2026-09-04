"""
Model evaluation with chronological train/test splitting.

All evaluation uses time-ordered splits to prevent temporal data leakage.
Metrics are computed only on held-out test data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def chronological_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    date_col: str = 'date',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train and test sets.

    The split is based on unique dates: the last `test_ratio` fraction of
    dates form the test set. This prevents future data from leaking into
    training.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with a date column.
    test_ratio : float
        Fraction of dates to use for testing. Default: 0.2
    date_col : str
        Name of the date column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    dates = sorted(df[date_col].unique())
    n_test = max(1, int(len(dates) * test_ratio))
    split_date = dates[-(n_test)]

    train = df[df[date_col] < split_date].copy()
    test = df[df[date_col] >= split_date].copy()

    return train, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.

    Metrics computed:
    - MAE: Mean Absolute Error (units)
    - RMSE: Root Mean Squared Error (units)
    - R2: Coefficient of determination
    - MAPE: Mean Absolute Percentage Error (%), only if all y_true > 0

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    Dict[str, float]
        Dictionary of metric name -> value.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    metrics = {
        'MAE': round(mean_absolute_error(y_true, y_pred), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'R2': round(r2_score(y_true, y_pred), 4),
    }

    # MAPE: only compute if no zero actuals (division by zero)
    if np.all(y_true > 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['MAPE'] = round(mape, 2)

    return metrics


def compute_baseline_metrics(y_true: np.ndarray, y_baseline: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for a baseline (naive) forecast.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_baseline : array-like
        Baseline predicted values.

    Returns
    -------
    Dict[str, float]
        Same format as compute_metrics.
    """
    return compute_metrics(y_true, y_baseline)


def compute_improvement(baseline_metrics: Dict, model_metrics: Dict) -> Dict[str, str]:
    """
    Compute improvement of model over baseline.

    For MAE, RMSE, MAPE: improvement = (baseline - model) / baseline * 100
    For R2: improvement = model_R2 - baseline_R2

    Returns
    -------
    Dict[str, str]
        Formatted improvement strings.
    """
    improvements = {}

    for metric in ['MAE', 'RMSE', 'MAPE']:
        if metric in baseline_metrics and metric in model_metrics:
            if baseline_metrics[metric] > 0:
                pct = (baseline_metrics[metric] - model_metrics[metric]) / baseline_metrics[metric] * 100
                improvements[metric] = f"{pct:+.1f}%"
            else:
                improvements[metric] = "N/A"

    if 'R2' in baseline_metrics and 'R2' in model_metrics:
        diff = model_metrics['R2'] - baseline_metrics['R2']
        improvements['R2'] = f"{diff:+.4f}"

    return improvements


def compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute residuals (actual - predicted) for prediction interval estimation.

    Returns
    -------
    np.ndarray
        Array of residuals.
    """
    return np.array(y_true) - np.array(y_pred)
