"""
Inventory optimization engine: Safety Stock, Reorder Point, EOQ.

Implements classical inventory management formulas:
- Safety Stock: buffer inventory to protect against demand variability
- Reorder Point (ROP): inventory level at which to trigger a new order
- Economic Order Quantity (EOQ): optimal order size minimizing total costs

Formulas are documented inline. All parameters are configurable.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from ..utils.config import (
    DEFAULT_LEAD_TIME_WEEKS,
    DEFAULT_SERVICE_LEVEL,
    DEFAULT_ORDERING_COST,
    DEFAULT_HOLDING_COST_PCT,
    SERVICE_LEVEL_Z,
)


def get_z_score(service_level: float) -> float:
    """
    Get the z-score for a given service level.

    Uses lookup table for common values, interpolates for others.

    Parameters
    ----------
    service_level : float
        Desired service level (0 to 1). E.g., 0.95 for 95%.

    Returns
    -------
    float
        Corresponding z-score from the standard normal distribution.
    """
    if service_level in SERVICE_LEVEL_Z:
        return SERVICE_LEVEL_Z[service_level]

    # Linear interpolation between known values
    levels = sorted(SERVICE_LEVEL_Z.keys())
    for i in range(len(levels) - 1):
        if levels[i] <= service_level <= levels[i + 1]:
            ratio = (service_level - levels[i]) / (levels[i + 1] - levels[i])
            return SERVICE_LEVEL_Z[levels[i]] + ratio * (
                SERVICE_LEVEL_Z[levels[i + 1]] - SERVICE_LEVEL_Z[levels[i]]
            )

    # Fallback: use 1.645 (95%)
    return 1.645


def compute_safety_stock(
    std_demand: float,
    lead_time_weeks: float,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> float:
    """
    Compute safety stock.

    Formula: SS = z * sigma_d * sqrt(LT)
    where:
      z = z-score for desired service level
      sigma_d = standard deviation of weekly demand
      LT = lead time in weeks

    Parameters
    ----------
    std_demand : float
        Standard deviation of weekly demand.
    lead_time_weeks : float
        Supplier lead time in weeks.
    service_level : float
        Desired service level (0-1).

    Returns
    -------
    float
        Safety stock in units.
    """
    z = get_z_score(service_level)
    return round(z * std_demand * np.sqrt(lead_time_weeks), 1)


def compute_reorder_point(
    avg_demand: float,
    std_demand: float,
    lead_time_weeks: float,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> float:
    """
    Compute reorder point.

    Formula: ROP = d_avg * LT + SS
    where:
      d_avg = average weekly demand
      LT = lead time in weeks
      SS = safety stock

    The reorder point is the inventory level at which a new order should
    be placed. When inventory drops to or below this level, order.

    Returns
    -------
    float
        Reorder point in units.
    """
    safety_stock = compute_safety_stock(std_demand, lead_time_weeks, service_level)
    lead_time_demand = avg_demand * lead_time_weeks
    return round(lead_time_demand + safety_stock, 1)


def compute_eoq(
    annual_demand: float,
    ordering_cost: float = DEFAULT_ORDERING_COST,
    holding_cost_pct: float = DEFAULT_HOLDING_COST_PCT,
    unit_cost: float = 1000.0,
) -> float:
    """
    Compute Economic Order Quantity.

    Formula: EOQ = sqrt(2 * D * S / H)
    where:
      D = annual demand (units per year)
      S = ordering/setup cost per order (INR)
      H = annual holding cost per unit = holding_cost_pct * unit_cost

    EOQ minimizes the sum of ordering costs and holding costs.

    Returns
    -------
    float
        Optimal order quantity in units.
    """
    holding_cost = holding_cost_pct * unit_cost
    if holding_cost <= 0 or annual_demand <= 0:
        return 0.0
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    return round(eoq, 1)


def analyze_inventory(
    df: pd.DataFrame,
    lead_time_weeks: float = DEFAULT_LEAD_TIME_WEEKS,
    service_level: float = DEFAULT_SERVICE_LEVEL,
    ordering_cost: float = DEFAULT_ORDERING_COST,
    holding_cost_pct: float = DEFAULT_HOLDING_COST_PCT,
) -> pd.DataFrame:
    """
    Compute inventory optimization metrics for all products.

    Parameters
    ----------
    df : pd.DataFrame
        Sales data with columns: date, product, category, units_sold, unit_price.
    lead_time_weeks : float
        Supplier lead time in weeks.
    service_level : float
        Desired service level.
    ordering_cost : float
        Cost per order placement.
    holding_cost_pct : float
        Annual holding cost as fraction of unit cost.

    Returns
    -------
    pd.DataFrame
        Inventory analysis with columns: Product, Category, Avg Weekly Demand,
        Demand Std Dev, Safety Stock, Reorder Point, EOQ, Annual Demand Est.,
        Unit Price.
    """
    results = []

    for product, grp in df.groupby('product'):
        weekly = grp.groupby('date')['units_sold'].sum()
        avg_demand = weekly.mean()
        std_demand = weekly.std() if len(weekly) > 1 else avg_demand * 0.2
        unit_price = grp['unit_price'].mean()
        category = grp['category'].iloc[0]

        # Annualize demand (52 weeks)
        n_weeks = len(weekly)
        annual_demand = avg_demand * 52

        ss = compute_safety_stock(std_demand, lead_time_weeks, service_level)
        rop = compute_reorder_point(avg_demand, std_demand, lead_time_weeks, service_level)
        eoq = compute_eoq(annual_demand, ordering_cost, holding_cost_pct, unit_price)

        results.append({
            'Product': product,
            'Category': category,
            'Avg Weekly Demand': round(avg_demand, 1),
            'Demand Std Dev': round(std_demand, 1),
            'Safety Stock': ss,
            'Reorder Point': rop,
            'EOQ': eoq,
            'Annual Demand Est.': int(annual_demand),
            'Unit Price': unit_price,
        })

    return pd.DataFrame(results).sort_values('Avg Weekly Demand', ascending=False).reset_index(drop=True)
