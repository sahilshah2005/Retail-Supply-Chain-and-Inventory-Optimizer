"""
Feature engineering pipeline for demand forecasting.

Creates temporal, lag, rolling, and trend features from retail sales data.
All features are designed to prevent data leakage: lag and rolling features
only use information available at prediction time (prior periods).
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def create_product_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw sales data into weekly product-level time series.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales data with columns: date, product, category, units_sold, unit_price.

    Returns
    -------
    pd.DataFrame
        Weekly aggregated data with one row per product per week, sorted by
        (product, date). Includes columns: date, product, category, units_sold,
        unit_price, revenue.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    agg = df.groupby(['date', 'product', 'category']).agg(
        units_sold=('units_sold', 'sum'),
        unit_price=('unit_price', 'mean'),
        revenue=('revenue', 'sum'),
    ).reset_index()

    agg = agg.sort_values(['product', 'date']).reset_index(drop=True)
    return agg


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features derived from the date column.

    Features added:
    - week_of_year: ISO week number (1-53)
    - month: month of year (1-12)
    - quarter: quarter of year (1-4)
    - year: calendar year

    These features capture seasonal and cyclical demand patterns.
    """
    df = df.copy()
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    return df


def add_lag_features(df: pd.DataFrame, lag_periods: List[int] = None) -> pd.DataFrame:
    """
    Add lag features for units_sold, computed per product.

    Lag features represent past demand values. For example, lag_1 is the
    demand from 1 period (week) ago. These are strictly backward-looking
    to prevent data leakage.

    Parameters
    ----------
    lag_periods : list of int
        Number of periods to lag. Default: [1, 2, 4]

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns: lag_1, lag_2, lag_4, etc.
        Early rows will have NaN where lag data is unavailable.
    """
    if lag_periods is None:
        lag_periods = [1, 2, 4]

    df = df.copy()
    df = df.sort_values(['product', 'date']).reset_index(drop=True)

    for lag in lag_periods:
        df[f'lag_{lag}'] = df.groupby('product')['units_sold'].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """
    Add rolling statistics for units_sold, computed per product.

    Rolling features summarize recent demand trends. The rolling window
    uses shift(1) to ensure only past data is used (no leakage).

    Parameters
    ----------
    windows : list of int
        Rolling window sizes in periods. Default: [4]

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns: rolling_mean_4, rolling_std_4, etc.
    """
    if windows is None:
        windows = [4]

    df = df.copy()
    df = df.sort_values(['product', 'date']).reset_index(drop=True)

    for w in windows:
        # shift(1) ensures we only use past values (not current period)
        shifted = df.groupby('product')['units_sold'].shift(1)
        df[f'rolling_mean_{w}'] = shifted.groupby(df['product']).transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        df[f'rolling_std_{w}'] = shifted.groupby(df['product']).transform(
            lambda x: x.rolling(window=w, min_periods=2).std()
        )

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend features capturing recent demand momentum.

    Features added:
    - demand_growth: percentage change from lag_2 to lag_1
      Positive values indicate growing demand; negative indicates declining.
      Uses lag values (already shifted) so no leakage.
    """
    df = df.copy()

    # Demand growth: change from 2 periods ago to 1 period ago
    if 'lag_1' in df.columns and 'lag_2' in df.columns:
        df['demand_growth'] = np.where(
            df['lag_2'] > 0,
            (df['lag_1'] - df['lag_2']) / df['lag_2'],
            0.0
        )
    else:
        # If lag features haven't been added yet, compute inline
        lag_1 = df.groupby('product')['units_sold'].shift(1)
        lag_2 = df.groupby('product')['units_sold'].shift(2)
        df['demand_growth'] = np.where(
            lag_2 > 0,
            (lag_1 - lag_2) / lag_2,
            0.0
        )

    return df


def encode_products(df: pd.DataFrame, product_mapping: dict = None) -> Tuple[pd.DataFrame, dict]:
    """
    Encode product names as integer IDs.

    Parameters
    ----------
    product_mapping : dict, optional
        Existing mapping of product name -> integer ID.
        If None, a new mapping is created from the data.
        If provided, unknown products are assigned -1.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (DataFrame with product_encoded column, mapping dict)
    """
    df = df.copy()

    if product_mapping is None:
        products = sorted(df['product'].unique())
        product_mapping = {p: i for i, p in enumerate(products)}

    df['product_encoded'] = df['product'].map(product_mapping).fillna(-1).astype(int)
    return df, product_mapping


def build_feature_matrix(
    df: pd.DataFrame,
    lag_periods: List[int] = None,
    rolling_windows: List[int] = None,
    product_mapping: dict = None,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Complete feature engineering pipeline.

    Applies all feature engineering steps in order:
    1. Aggregate to weekly product series
    2. Add temporal features
    3. Add lag features
    4. Add rolling features
    5. Add trend features
    6. Encode products
    7. Optionally drop rows with NaN (from lag/rolling)

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales data.
    lag_periods : list of int, optional
        Lag periods for lag features. Default: [1, 2, 4]
    rolling_windows : list of int, optional
        Window sizes for rolling features. Default: [4]
    product_mapping : dict, optional
        Existing product encoding mapping. None = create new.
    drop_na : bool
        Whether to drop rows with NaN values from lag/rolling features.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (feature_matrix, product_mapping)
    """
    if lag_periods is None:
        lag_periods = [1, 2, 4]
    if rolling_windows is None:
        rolling_windows = [4]

    result = create_product_weekly_series(df)
    result = add_temporal_features(result)
    result = add_lag_features(result, lag_periods)
    result = add_rolling_features(result, rolling_windows)
    result = add_trend_features(result)
    result, mapping = encode_products(result, product_mapping)

    if drop_na:
        result = result.dropna().reset_index(drop=True)

    return result, mapping


def get_feature_columns(
    lag_periods: List[int] = None,
    rolling_windows: List[int] = None,
) -> List[str]:
    """
    Return the list of feature column names used by the model.

    This ensures consistency between training and prediction.
    """
    if lag_periods is None:
        lag_periods = [1, 2, 4]
    if rolling_windows is None:
        rolling_windows = [4]

    features = [
        'week_of_year', 'month', 'quarter',
        'product_encoded',
    ]
    features += [f'lag_{l}' for l in lag_periods]
    features += [f'rolling_mean_{w}' for w in rolling_windows]
    features += [f'rolling_std_{w}' for w in rolling_windows]
    features += ['demand_growth']

    return features
