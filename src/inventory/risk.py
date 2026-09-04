"""
Inventory risk classification engine.

Classifies each product's inventory position into risk categories
and generates prioritized recommended actions.

Risk classifications:
- Healthy: current inventory > reorder point + safety stock
- Monitor: current inventory > reorder point but within safety stock buffer
- Reorder Soon: current inventory near reorder point
- Stockout Risk: current inventory < safety stock
- Overstock: current inventory > 3 * EOQ (significantly above normal levels)

Assumptions:
- Current inventory is a user-provided simulation input (not from historical data)
- Risk thresholds are based on inventory optimization calculations
- Priority combines ABC class (if available) with risk severity
"""
import pandas as pd
import numpy as np
from typing import Optional


def classify_risk(
    current_inventory: float,
    safety_stock: float,
    reorder_point: float,
    eoq: float,
) -> str:
    """
    Classify inventory risk based on current position relative to key levels.

    Parameters
    ----------
    current_inventory : float
        Current inventory level (user-provided simulation input).
    safety_stock : float
        Computed safety stock.
    reorder_point : float
        Computed reorder point.
    eoq : float
        Economic order quantity.

    Returns
    -------
    str
        Risk category: 'Stockout Risk', 'Reorder Soon', 'Monitor',
        'Healthy', or 'Overstock'.
    """
    if current_inventory <= safety_stock:
        return "Stockout Risk"
    elif current_inventory <= reorder_point:
        return "Reorder Soon"
    elif current_inventory <= reorder_point + safety_stock:
        return "Monitor"
    elif eoq > 0 and current_inventory > 3 * eoq:
        return "Overstock"
    else:
        return "Healthy"


def recommend_action(
    risk: str,
    current_inventory: float,
    reorder_point: float,
    eoq: float,
    safety_stock: float,
) -> tuple:
    """
    Generate recommended action and order quantity based on risk.

    Returns
    -------
    tuple
        (action: str, recommended_order_qty: float)
    """
    if risk == "Stockout Risk":
        # Order immediately: bring inventory up to reorder point + EOQ
        order_qty = max(reorder_point + eoq - current_inventory, eoq)
        return "Order Immediately", round(order_qty, 0)
    elif risk == "Reorder Soon":
        return "Reorder Soon", round(eoq, 0)
    elif risk == "Monitor":
        return "Monitor", 0.0
    elif risk == "Overstock":
        return "No Action Required", 0.0
    else:  # Healthy
        return "No Action Required", 0.0


def assign_priority(risk: str, abc_class: str = "C") -> str:
    """
    Assign priority based on risk and ABC classification.

    Priority matrix:
    - Critical: A-class + Stockout Risk
    - High: A-class + Reorder Soon, or B-class + Stockout Risk
    - Medium: other risk combinations
    - Low: Healthy or Overstock with C-class

    Parameters
    ----------
    risk : str
        Risk classification.
    abc_class : str
        ABC classification (A, B, or C).

    Returns
    -------
    str
        Priority level: 'Critical', 'High', 'Medium', or 'Low'.
    """
    if abc_class == "A" and risk == "Stockout Risk":
        return "Critical"
    elif abc_class == "A" and risk == "Reorder Soon":
        return "High"
    elif abc_class == "B" and risk == "Stockout Risk":
        return "High"
    elif risk in ("Stockout Risk", "Reorder Soon"):
        return "Medium"
    elif risk == "Monitor":
        return "Medium"
    else:
        return "Low"


def build_risk_report(
    inventory_df: pd.DataFrame,
    current_inventory_map: Optional[dict] = None,
    default_inventory: float = 100.0,
    abc_classes: Optional[dict] = None,
    forecast_demand: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a comprehensive inventory risk report.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        Output from analyze_inventory() with columns:
        Product, Category, Avg Weekly Demand, Demand Std Dev,
        Safety Stock, Reorder Point, EOQ, etc.
    current_inventory_map : dict, optional
        {product: current_inventory}. If None, uses default_inventory.
    default_inventory : float
        Default current inventory value for simulation.
    abc_classes : dict, optional
        {product: 'A'/'B'/'C'} from ABC analysis.
    forecast_demand : dict, optional
        {product: forecasted_demand} from forecasting module.

    Returns
    -------
    pd.DataFrame
        Risk report with columns: Product, Category, Forecast Demand,
        Lead-Time Demand, Current Inventory, Safety Stock, Reorder Point,
        EOQ, ABC Class, Risk, Recommended Order Qty, Priority,
        Recommended Action.
    """
    if current_inventory_map is None:
        current_inventory_map = {}
    if abc_classes is None:
        abc_classes = {}
    if forecast_demand is None:
        forecast_demand = {}

    records = []

    for _, row in inventory_df.iterrows():
        product = row['Product']
        category = row['Category']
        avg_demand = row['Avg Weekly Demand']
        safety_stock = row['Safety Stock']
        reorder_point = row['Reorder Point']
        eoq = row['EOQ']

        current_inv = current_inventory_map.get(product, default_inventory)
        abc_class = abc_classes.get(product, 'C')
        fc_demand = forecast_demand.get(product, avg_demand)

        # Lead-time demand based on forecast or average
        lead_time_demand = row.get('Avg Weekly Demand', avg_demand) * 2  # Default 2 weeks

        risk = classify_risk(current_inv, safety_stock, reorder_point, eoq)
        action, order_qty = recommend_action(risk, current_inv, reorder_point, eoq, safety_stock)
        priority = assign_priority(risk, abc_class)

        records.append({
            'Product': product,
            'Category': category,
            'Forecast Demand': round(fc_demand, 1),
            'Lead-Time Demand': round(lead_time_demand, 1),
            'Current Inventory': round(current_inv, 0),
            'Safety Stock': safety_stock,
            'Reorder Point': reorder_point,
            'EOQ': eoq,
            'ABC Class': abc_class,
            'Risk': risk,
            'Recommended Order Qty': order_qty,
            'Priority': priority,
            'Recommended Action': action,
        })

    result = pd.DataFrame(records)

    # Sort by priority
    priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    result['_priority_sort'] = result['Priority'].map(priority_order)
    result = result.sort_values('_priority_sort').drop(columns='_priority_sort').reset_index(drop=True)

    return result
