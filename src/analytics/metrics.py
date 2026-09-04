"""
Business KPI computation for the executive dashboard.

Computes aggregate business metrics from sales data and model outputs.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


def compute_business_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute top-level business KPIs from sales data.

    Parameters
    ----------
    df : pd.DataFrame
        Validated sales data with columns: date, product, category,
        units_sold, unit_price, revenue.

    Returns
    -------
    Dict with keys:
        total_revenue, total_units, n_products, n_categories,
        avg_weekly_revenue, avg_weekly_units, avg_unit_price,
        date_range, n_weeks
    """
    weekly_revenue = df.groupby('date')['revenue'].sum()
    weekly_units = df.groupby('date')['units_sold'].sum()

    return {
        'total_revenue': df['revenue'].sum(),
        'total_units': df['units_sold'].sum(),
        'n_products': df['product'].nunique(),
        'n_categories': df['category'].nunique(),
        'avg_weekly_revenue': weekly_revenue.mean(),
        'avg_weekly_units': weekly_units.mean(),
        'avg_unit_price': df['unit_price'].mean(),
        'date_range': (
            str(df['date'].min().date()),
            str(df['date'].max().date()),
        ),
        'n_weeks': df['date'].nunique(),
    }


def compute_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute revenue and units by category.

    Returns
    -------
    pd.DataFrame
        Columns: Category, Total Revenue, Total Units, Avg Price, Revenue Share (%)
    """
    cat_agg = (
        df.groupby('category')
        .agg(
            total_revenue=('revenue', 'sum'),
            total_units=('units_sold', 'sum'),
            avg_price=('unit_price', 'mean'),
        )
        .reset_index()
        .sort_values('total_revenue', ascending=False)
    )

    total = cat_agg['total_revenue'].sum()
    cat_agg['revenue_share'] = (cat_agg['total_revenue'] / total * 100).round(1) if total > 0 else 0

    cat_agg.columns = ['Category', 'Total Revenue', 'Total Units', 'Avg Price', 'Revenue Share (%)']
    return cat_agg.reset_index(drop=True)


def compute_product_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute revenue, units, and demand stats by product.

    Returns
    -------
    pd.DataFrame
        Columns: Product, Category, Total Revenue, Total Units,
        Avg Weekly Demand, Demand Std Dev, Avg Price
    """
    product_agg = []

    for product, grp in df.groupby('product'):
        weekly = grp.groupby('date')['units_sold'].sum()
        category = grp['category'].iloc[0]

        product_agg.append({
            'Product': product,
            'Category': category,
            'Total Revenue': grp['revenue'].sum(),
            'Total Units': grp['units_sold'].sum(),
            'Avg Weekly Demand': round(weekly.mean(), 1),
            'Demand Std Dev': round(weekly.std(), 1) if len(weekly) > 1 else 0,
            'Avg Price': round(grp['unit_price'].mean(), 0),
        })

    return (
        pd.DataFrame(product_agg)
        .sort_values('Total Revenue', ascending=False)
        .reset_index(drop=True)
    )


def compute_weekly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly revenue and units trends.

    Returns
    -------
    pd.DataFrame
        Columns: date, revenue, units_sold
    """
    return (
        df.groupby('date')
        .agg(
            revenue=('revenue', 'sum'),
            units_sold=('units_sold', 'sum'),
        )
        .reset_index()
        .sort_values('date')
    )
