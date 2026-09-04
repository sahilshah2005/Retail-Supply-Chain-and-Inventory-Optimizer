"""
Data loading, validation, and quality reporting for retail sales data.

Performs comprehensive data quality checks and returns a validated DataFrame
along with a quality report summarizing any issues found.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from .config import DEFAULT_DATA_FILE


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load retail sales CSV and compute derived columns.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV file. Defaults to DEFAULT_DATA_FILE from config.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed dates and computed revenue column.
    """
    if filepath is None:
        filepath = DEFAULT_DATA_FILE

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['revenue'] = df['units_sold'] * df['unit_price']
    return df


def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and clean a retail sales DataFrame.

    Checks for:
    - Missing values
    - Duplicate rows
    - Invalid dates
    - Negative units_sold
    - Invalid (zero/negative) prices
    - Missing product identifiers
    - Missing categories

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (cleaned_df, quality_report)
        The quality report documents all checks and any transformations applied.
    """
    report = {
        'total_rows': len(df),
        'issues': [],
        'transformations': [],
    }
    df_clean = df.copy()

    # --- Check required columns ---
    required_cols = ['date', 'product', 'category', 'units_sold', 'unit_price']
    missing_cols = [c for c in required_cols if c not in df_clean.columns]
    if missing_cols:
        report['issues'].append(f"Missing required columns: {missing_cols}")
        # Cannot proceed without required columns
        report['rows_after_cleaning'] = len(df_clean)
        return df_clean, report

    # --- Missing values ---
    missing_counts = df_clean[required_cols].isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        report['issues'].append(
            f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}"
        )
        rows_before = len(df_clean)
        df_clean = df_clean.dropna(subset=required_cols)
        dropped = rows_before - len(df_clean)
        if dropped > 0:
            report['transformations'].append(
                f"Dropped {dropped} rows with missing required values"
            )

    # --- Duplicate rows ---
    n_duplicates = df_clean.duplicated().sum()
    if n_duplicates > 0:
        report['issues'].append(f"Found {n_duplicates} duplicate rows")
        df_clean = df_clean.drop_duplicates()
        report['transformations'].append(
            f"Removed {n_duplicates} duplicate rows"
        )

    # --- Invalid dates ---
    invalid_dates = df_clean['date'].isna().sum()
    if invalid_dates > 0:
        report['issues'].append(f"Found {invalid_dates} invalid/unparseable dates")
        df_clean = df_clean.dropna(subset=['date'])
        report['transformations'].append(
            f"Removed {invalid_dates} rows with invalid dates"
        )

    # --- Negative units_sold ---
    if 'units_sold' in df_clean.columns:
        negative_units = (df_clean['units_sold'] < 0).sum()
        if negative_units > 0:
            report['issues'].append(
                f"Found {negative_units} rows with negative units_sold"
            )
            df_clean = df_clean[df_clean['units_sold'] >= 0]
            report['transformations'].append(
                f"Removed {negative_units} rows with negative units_sold"
            )

    # --- Invalid prices (zero or negative) ---
    if 'unit_price' in df_clean.columns:
        invalid_prices = (df_clean['unit_price'] <= 0).sum()
        if invalid_prices > 0:
            report['issues'].append(
                f"Found {invalid_prices} rows with zero or negative unit_price"
            )
            df_clean = df_clean[df_clean['unit_price'] > 0]
            report['transformations'].append(
                f"Removed {invalid_prices} rows with invalid prices"
            )

    # --- Missing product identifiers ---
    empty_products = df_clean['product'].astype(str).str.strip().eq('').sum()
    if empty_products > 0:
        report['issues'].append(
            f"Found {empty_products} rows with empty product names"
        )

    # --- Missing categories ---
    empty_categories = df_clean['category'].astype(str).str.strip().eq('').sum()
    if empty_categories > 0:
        report['issues'].append(
            f"Found {empty_categories} rows with empty categories"
        )

    # --- Recompute revenue after cleaning ---
    if 'units_sold' in df_clean.columns and 'unit_price' in df_clean.columns:
        df_clean['revenue'] = df_clean['units_sold'] * df_clean['unit_price']

    # --- Summary statistics ---
    report['rows_after_cleaning'] = len(df_clean)
    report['rows_removed'] = report['total_rows'] - len(df_clean)
    report['n_products'] = df_clean['product'].nunique() if len(df_clean) > 0 else 0
    report['n_categories'] = df_clean['category'].nunique() if len(df_clean) > 0 else 0
    report['date_range'] = (
        (str(df_clean['date'].min().date()), str(df_clean['date'].max().date()))
        if len(df_clean) > 0 else ('N/A', 'N/A')
    )
    report['n_weeks'] = df_clean['date'].nunique() if len(df_clean) > 0 else 0

    if not report['issues']:
        report['issues'].append("No data quality issues found")
    if not report['transformations']:
        report['transformations'].append("No transformations needed")

    return df_clean, report


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a high-level summary of the dataset.

    Returns
    -------
    Dict with keys: n_rows, n_products, n_categories, n_weeks, date_range,
    products, categories
    """
    return {
        'n_rows': len(df),
        'n_products': df['product'].nunique(),
        'n_categories': df['category'].nunique(),
        'n_weeks': df['date'].nunique(),
        'date_range': (
            str(df['date'].min().date()),
            str(df['date'].max().date()),
        ),
        'products': sorted(df['product'].unique().tolist()),
        'categories': sorted(df['category'].unique().tolist()),
    }
