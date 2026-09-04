"""
Automated insights and alert generation.

Analyzes inventory risk reports and forecast data to surface
actionable findings. All insights are derived from actual
calculations - nothing is fabricated.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def generate_alerts(risk_report: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate prioritized alerts from the risk report.

    Alert types:
    - stockout_risk: Products with stockout risk
    - reorder_needed: Products that need reordering
    - high_demand: Products with above-average demand
    - volatile_demand: Products with high demand variability
    - overstock: Products with excess inventory
    - critical_abc: A-class products at risk

    Parameters
    ----------
    risk_report : pd.DataFrame
        Output from build_risk_report().

    Returns
    -------
    List[Dict]
        List of alert dictionaries with keys:
        type, severity, icon, title, message, products
    """
    alerts = []

    # Stockout risk
    stockout = risk_report[risk_report['Risk'] == 'Stockout Risk']
    if len(stockout) > 0:
        alerts.append({
            'type': 'stockout_risk',
            'severity': 'critical',
            'icon': '🔴',
            'title': 'Stockout Risk Detected',
            'message': f"{len(stockout)} product(s) at risk of stockout: {', '.join(stockout['Product'].tolist())}",
            'products': stockout['Product'].tolist(),
        })

    # Reorder needed
    reorder = risk_report[risk_report['Risk'] == 'Reorder Soon']
    if len(reorder) > 0:
        alerts.append({
            'type': 'reorder_needed',
            'severity': 'warning',
            'icon': '🟡',
            'title': 'Reorder Recommended',
            'message': f"{len(reorder)} product(s) approaching reorder point: {', '.join(reorder['Product'].tolist())}",
            'products': reorder['Product'].tolist(),
        })

    # Critical: A-class items at risk
    if 'ABC Class' in risk_report.columns:
        critical_abc = risk_report[
            (risk_report['ABC Class'] == 'A') &
            (risk_report['Risk'].isin(['Stockout Risk', 'Reorder Soon']))
        ]
        if len(critical_abc) > 0:
            alerts.append({
                'type': 'critical_abc',
                'severity': 'critical',
                'icon': '⚠️',
                'title': 'High-Value Products at Risk',
                'message': f"{len(critical_abc)} A-class (high-value) product(s) need attention: {', '.join(critical_abc['Product'].tolist())}",
                'products': critical_abc['Product'].tolist(),
            })

    # Overstock
    overstock = risk_report[risk_report['Risk'] == 'Overstock']
    if len(overstock) > 0:
        alerts.append({
            'type': 'overstock',
            'severity': 'info',
            'icon': '📦',
            'title': 'Potential Overstock',
            'message': f"{len(overstock)} product(s) may be overstocked: {', '.join(overstock['Product'].tolist())}",
            'products': overstock['Product'].tolist(),
        })

    # If no alerts
    if not alerts:
        alerts.append({
            'type': 'all_clear',
            'severity': 'success',
            'icon': '✅',
            'title': 'All Products Healthy',
            'message': 'No immediate inventory risks detected. Continue monitoring.',
            'products': [],
        })

    return alerts


def generate_demand_insights(
    product_performance: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Generate demand-related insights from product performance data.

    Parameters
    ----------
    product_performance : pd.DataFrame
        Output from compute_product_performance().

    Returns
    -------
    List[Dict]
        List of insight dictionaries.
    """
    insights = []

    if len(product_performance) == 0:
        return insights

    # Highest demand product
    top = product_performance.iloc[0]
    insights.append({
        'type': 'top_product',
        'icon': '🏆',
        'title': 'Top Revenue Product',
        'message': f"{top['Product']} leads with \u20b9{top['Total Revenue']:,.0f} total revenue.",
    })

    # High variability products (CV > 0.15)
    if 'Demand Std Dev' in product_performance.columns and 'Avg Weekly Demand' in product_performance.columns:
        pp = product_performance.copy()
        pp['cv'] = pp['Demand Std Dev'] / pp['Avg Weekly Demand'].replace(0, np.nan)
        volatile = pp[pp['cv'] > 0.15]
        if len(volatile) > 0:
            insights.append({
                'type': 'volatile_demand',
                'icon': '📊',
                'title': 'Variable Demand Products',
                'message': f"{len(volatile)} product(s) show notable demand variability (CV > 15%): {', '.join(volatile['Product'].tolist())}. Consider higher safety stock.",
            })

    return insights


def get_risk_summary(risk_report: pd.DataFrame) -> Dict[str, int]:
    """
    Count products by risk category.

    Returns
    -------
    Dict[str, int]
        {risk_category: count}
    """
    return risk_report['Risk'].value_counts().to_dict()
