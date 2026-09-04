"""
ABC inventory classification analysis.

Classifies products using the Pareto principle based on sales value:
- A items: High-value items contributing to ~70% of total value
- B items: Medium-value items contributing to next ~20% (70-90% cumulative)
- C items: Low-value items contributing to remaining ~10% (90-100%)

Threshold assumptions:
- A: cumulative value contribution <= 70%
- B: cumulative value contribution <= 90%
- C: everything else

These thresholds follow the classic 70/20/10 ABC split commonly used
in inventory management. They are configurable.
"""
import pandas as pd
import numpy as np
from typing import Tuple

from ..utils.config import ABC_A_THRESHOLD, ABC_B_THRESHOLD


def compute_abc_classification(
    df: pd.DataFrame,
    a_threshold: float = ABC_A_THRESHOLD,
    b_threshold: float = ABC_B_THRESHOLD,
) -> pd.DataFrame:
    """
    Classify products into ABC categories based on sales value.

    The classification uses total revenue (units_sold * unit_price) as the
    value measure. Products are ranked by value, and cumulative contribution
    determines the class.

    Parameters
    ----------
    df : pd.DataFrame
        Sales data with columns: product, units_sold, unit_price, revenue.
    a_threshold : float
        Cumulative value threshold for A class (default: 0.70).
    b_threshold : float
        Cumulative value threshold for B class (default: 0.90).

    Returns
    -------
    pd.DataFrame
        ABC classification with columns: Product, Category, Total Revenue,
        Value Percentage, Cumulative Percentage, ABC Class.
    """
    if 'revenue' not in df.columns:
        df = df.copy()
        df['revenue'] = df['units_sold'] * df['unit_price']

    # Aggregate by product
    product_value = (
        df.groupby(['product', 'category'])
        .agg(
            total_revenue=('revenue', 'sum'),
            total_units=('units_sold', 'sum'),
            avg_price=('unit_price', 'mean'),
        )
        .reset_index()
        .sort_values('total_revenue', ascending=False)
        .reset_index(drop=True)
    )

    # Calculate percentage and cumulative percentage
    total = product_value['total_revenue'].sum()
    if total > 0:
        product_value['value_pct'] = product_value['total_revenue'] / total
    else:
        product_value['value_pct'] = 0.0

    product_value['cumulative_pct'] = product_value['value_pct'].cumsum()

    # Assign ABC class
    def assign_class(cum_pct):
        if cum_pct <= a_threshold:
            return 'A'
        elif cum_pct <= b_threshold:
            return 'B'
        else:
            return 'C'

    product_value['abc_class'] = product_value['cumulative_pct'].apply(assign_class)

    # Format output
    result = pd.DataFrame({
        'Product': product_value['product'],
        'Category': product_value['category'],
        'Total Revenue': product_value['total_revenue'].round(0),
        'Total Units': product_value['total_units'],
        'Avg Price': product_value['avg_price'].round(0),
        'Value Percentage': (product_value['value_pct'] * 100).round(2),
        'Cumulative Percentage': (product_value['cumulative_pct'] * 100).round(2),
        'ABC Class': product_value['abc_class'],
    })

    return result


def get_abc_class_map(abc_df: pd.DataFrame) -> dict:
    """
    Extract a product -> ABC class mapping from classification results.

    Returns
    -------
    dict
        {product_name: 'A'/'B'/'C'}
    """
    return dict(zip(abc_df['Product'], abc_df['ABC Class']))


def get_abc_summary(abc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize ABC classification: count and value per class.

    Returns
    -------
    pd.DataFrame
        Summary with columns: ABC Class, Product Count, Total Revenue,
        Value Share (%).
    """
    summary = (
        abc_df.groupby('ABC Class')
        .agg(
            product_count=('Product', 'count'),
            total_revenue=('Total Revenue', 'sum'),
        )
        .reset_index()
    )
    total_rev = summary['total_revenue'].sum()
    if total_rev > 0:
        summary['value_share_pct'] = (summary['total_revenue'] / total_rev * 100).round(1)
    else:
        summary['value_share_pct'] = 0.0

    summary.columns = ['ABC Class', 'Product Count', 'Total Revenue', 'Value Share (%)']
    return summary.sort_values('ABC Class').reset_index(drop=True)
