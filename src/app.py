"""
Retail Supply Chain and Inventory Optimizer — Streamlit Application

Main entry point for the interactive dashboard providing:
1. Executive Dashboard — KPIs, trends, risk overview
2. Demand Forecasting — ML forecast with uncertainty
3. Inventory Optimization — EOQ, ROP, safety stock analysis
4. Inventory Risk — Risk classification and reorder recommendations
5. Product Intelligence — Single-SKU deep-dive
6. Scenario Simulator — What-if analysis
7. Sales Analytics — Trends and category comparisons
8. Model Evaluation — Comparison table, feature importance

Author: Sahil Shah
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
import os
import io

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROOT_DIR)

from src.utils.config import (
    APP_NAME, APP_ICON, DEFAULT_LEAD_TIME_WEEKS, DEFAULT_SERVICE_LEVEL,
    DEFAULT_ORDERING_COST, DEFAULT_HOLDING_COST_PCT, DEFAULT_CURRENT_INVENTORY,
    DEFAULT_FORECAST_HORIZON, MIN_FORECAST_HORIZON, MAX_FORECAST_HORIZON,
    PREDICTION_INTERVAL_Z, PREDICTION_INTERVAL_LEVEL,
)
from src.utils.data_loader import load_data, validate_data, get_data_summary
from src.forecasting.models import ForecastingPipeline
from src.inventory.optimizer import analyze_inventory
from src.inventory.risk import build_risk_report
from src.inventory.abc_analysis import (
    compute_abc_classification, get_abc_class_map, get_abc_summary,
)
from src.analytics.metrics import (
    compute_business_kpis, compute_category_performance,
    compute_product_performance, compute_weekly_trends,
)
from src.analytics.insights import generate_alerts, generate_demand_insights, get_risk_summary

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f2937;
        border-left: 4px solid #667eea;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def cached_load_and_validate():
    df = load_data()
    df_clean, quality_report = validate_data(df)
    return df_clean, quality_report


@st.cache_resource
def cached_train_pipeline(_df):
    pipeline = ForecastingPipeline()
    results = pipeline.train_and_evaluate(_df)
    return pipeline, results


df, quality_report = cached_load_and_validate()
pipeline, model_results = cached_train_pipeline(df)

# Precompute common analytics
kpis = compute_business_kpis(df)
abc_df = compute_abc_classification(df)
abc_map = get_abc_class_map(abc_df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{APP_ICON} {APP_NAME}")
    st.caption("AI-Powered Supply Chain Decision Support")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "📊 Executive Dashboard",
            "🔮 Demand Forecasting",
            "📦 Inventory Optimization",
            "⚠️ Inventory Risk",
            "🔍 Product Intelligence",
            "🎯 Scenario Simulator",
            "📈 Sales Analytics",
            "🤖 Model Evaluation",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    category_filter = st.multiselect(
        "Filter by Category",
        options=sorted(df['category'].unique().tolist()),
        default=sorted(df['category'].unique().tolist()),
    )

    st.markdown("---")
    st.markdown(
        "<small>Built with Python · Scikit-Learn · Streamlit<br>"
        "© 2024 Sahil Shah</small>",
        unsafe_allow_html=True,
    )

filtered_df = df[df['category'].isin(category_filter)] if category_filter else df


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    st.title("📊 Executive Dashboard")
    st.caption("Supply chain overview: What happened? What's expected? What needs attention?")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Revenue", f"₹{kpis['total_revenue']:,.0f}")
    with c2:
        st.metric("Units Sold", f"{kpis['total_units']:,}")
    with c3:
        st.metric("Products Tracked", kpis['n_products'])
    with c4:
        best_r2 = model_results['test_metrics'].get('R2', 'N/A')
        st.metric("Forecast R²", f"{best_r2}")
    with c5:
        best_mae = model_results['test_metrics'].get('MAE', 'N/A')
        st.metric("Forecast MAE", f"{best_mae} units")

    st.divider()

    # ── Risk Summary ──────────────────────────────────────────────────────────
    inv_df = analyze_inventory(filtered_df)
    risk_report = build_risk_report(
        inv_df, abc_classes=abc_map,
        default_inventory=DEFAULT_CURRENT_INVENTORY,
    )
    risk_counts = get_risk_summary(risk_report)
    alerts = generate_alerts(risk_report)

    cr1, cr2, cr3 = st.columns(3)
    with cr1:
        stockout_count = risk_counts.get('Stockout Risk', 0)
        st.metric("Stockout Risk", stockout_count,
                   delta=None if stockout_count == 0 else "⚠️ needs attention",
                   delta_color="inverse")
    with cr2:
        reorder_count = risk_counts.get('Reorder Soon', 0)
        st.metric("Reorder Needed", reorder_count)
    with cr3:
        overstock_count = risk_counts.get('Overstock', 0)
        st.metric("Overstock", overstock_count)

    # ── Alerts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Alerts & Recommendations</div>',
                unsafe_allow_html=True)
    for alert in alerts:
        if alert['severity'] == 'critical':
            st.error(f"{alert['icon']} **{alert['title']}**: {alert['message']}")
        elif alert['severity'] == 'warning':
            st.warning(f"{alert['icon']} **{alert['title']}**: {alert['message']}")
        elif alert['severity'] == 'success':
            st.success(f"{alert['icon']} **{alert['title']}**: {alert['message']}")
        else:
            st.info(f"{alert['icon']} **{alert['title']}**: {alert['message']}")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Weekly Revenue Trend</div>',
                    unsafe_allow_html=True)
        weekly = compute_weekly_trends(filtered_df)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.fill_between(weekly['date'], weekly['revenue'], alpha=0.3, color='#667eea')
        ax.plot(weekly['date'], weekly['revenue'], color='#667eea', linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue (₹)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K'))
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-header">Revenue by Category</div>',
                    unsafe_allow_html=True)
        cat_perf = compute_category_performance(filtered_df)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a'][:len(cat_perf)]
        wedges, texts, autotexts = ax2.pie(
            cat_perf['Total Revenue'].values,
            labels=cat_perf['Category'].values,
            autopct='%1.1f%%',
            colors=colors, startangle=90,
        )
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Top Products ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Top Products by Revenue</div>',
                unsafe_allow_html=True)
    prod_perf = compute_product_performance(filtered_df)
    display_df = prod_perf[['Product', 'Category', 'Total Revenue', 'Total Units', 'Avg Weekly Demand']].copy()
    display_df['Total Revenue'] = display_df['Total Revenue'].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMAND FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Demand Forecasting":
    st.title("🔮 Demand Forecasting")
    best_name = model_results['best_model']
    test_met = model_results['test_metrics']
    st.caption(
        f"Best Model: **{best_name}** | "
        f"Test MAE: **{test_met.get('MAE', 'N/A')}** | "
        f"Test R²: **{test_met.get('R2', 'N/A')}**"
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        product = st.selectbox("Select Product", sorted(df['product'].unique()))
        weeks = st.slider(
            "Forecast Horizon (weeks)",
            min_value=MIN_FORECAST_HORIZON,
            max_value=MAX_FORECAST_HORIZON,
            value=DEFAULT_FORECAST_HORIZON,
        )
        run = st.button("Generate Forecast", type="primary", use_container_width=True)

    if run:
        with col2:
            history, forecast_df = pipeline.get_product_forecast_with_history(
                product, df, weeks
            )

            st.markdown(
                f'<div class="section-header">Forecast: {product} — '
                f'Next {weeks} Weeks</div>',
                unsafe_allow_html=True,
            )

            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(history['date'], history['units_sold'],
                    label='Historical', color='#667eea', linewidth=2)

            forecast_dates = pd.to_datetime(forecast_df['date'])
            ax.plot(forecast_dates, forecast_df['forecast'],
                    label='Forecast', color='#f093fb', linewidth=2,
                    linestyle='--', marker='o')

            # Prediction interval
            ax.fill_between(
                forecast_dates,
                forecast_df['lower_bound'],
                forecast_df['upper_bound'],
                alpha=0.2, color='#f093fb',
                label=f'{int(PREDICTION_INTERVAL_LEVEL*100)}% Prediction Interval',
            )

            ax.axvline(x=history['date'].iloc[-1], color='gray',
                       linestyle=':', alpha=0.7, label='Forecast Start')
            ax.set_xlabel("Date")
            ax.set_ylabel("Units")
            ax.legend(fontsize=8)
            ax.tick_params(axis='x', rotation=30)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.caption(
                f"_Prediction intervals are {int(PREDICTION_INTERVAL_LEVEL*100)}% "
                f"residual-based intervals. They indicate the range within which "
                f"actual demand is expected to fall, based on model residuals from "
                f"training data._"
            )

            display_fc = forecast_df.copy()
            display_fc['date'] = display_fc['date'].dt.strftime('%Y-%m-%d')
            display_fc.columns = ['Week', 'Forecast', 'Lower Bound', 'Upper Bound']
            st.dataframe(display_fc, use_container_width=True, hide_index=True)
    else:
        with col2:
            st.info("👈 Select a product and click **Generate Forecast** to see predictions.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INVENTORY OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Inventory Optimization":
    st.title("📦 Inventory Optimization")
    st.caption("EOQ, Reorder Point, and Safety Stock analysis")

    inv_df = analyze_inventory(
        filtered_df,
        lead_time_weeks=DEFAULT_LEAD_TIME_WEEKS,
        service_level=DEFAULT_SERVICE_LEVEL,
        ordering_cost=DEFAULT_ORDERING_COST,
        holding_cost_pct=DEFAULT_HOLDING_COST_PCT,
    )

    st.markdown('<div class="section-header">Inventory Summary</div>',
                unsafe_allow_html=True)
    st.dataframe(inv_df, use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Reorder Points by Product</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(inv_df['Product'], inv_df['Reorder Point'], color='#667eea')
        ax.set_xlabel("Units")
        ax.bar_label(bars, padding=3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">EOQ by Product</div>',
                    unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars2 = ax2.barh(inv_df['Product'], inv_df['EOQ'], color='#f093fb')
        ax2.set_xlabel("Units (Economic Order Quantity)")
        ax2.bar_label(bars2, padding=3)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.info(
        "**EOQ** = Economic Order Quantity — optimal units to order per batch "
        "to minimize total inventory cost (ordering + holding). "
        "**Reorder Point** = inventory level at which a new order should be "
        "triggered, accounting for lead time demand and safety stock. "
        "**Safety Stock** = buffer to protect against demand variability "
        f"at {DEFAULT_SERVICE_LEVEL*100:.0f}% service level."
    )

    # ── Formulas ──────────────────────────────────────────────────────────────
    with st.expander("📐 Formulas Used"):
        st.markdown("""
        **Safety Stock** = z × σ_demand × √(Lead Time)

        **Reorder Point** = (Avg Demand × Lead Time) + Safety Stock

        **EOQ** = √(2 × Annual Demand × Order Cost / Holding Cost per Unit)

        Where:
        - z = z-score for the target service level
        - σ_demand = standard deviation of weekly demand
        - Holding Cost per Unit = Holding Cost % × Unit Price
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INVENTORY RISK
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Inventory Risk":
    st.title("⚠️ Inventory Risk & Reorder Recommendations")
    st.caption(
        "Risk classification based on current inventory position. "
        "Current inventory values are user-configurable simulation inputs."
    )

    # Current inventory input
    st.markdown('<div class="section-header">Current Inventory (Simulation Input)</div>',
                unsafe_allow_html=True)
    st.warning(
        "⚠️ Current inventory values are **assumed/scenario inputs** for simulation "
        "purposes. They are not from historical data. Adjust below to model different scenarios."
    )

    products = sorted(df['product'].unique())
    inv_inputs = {}
    cols = st.columns(min(len(products), 5))
    for i, prod in enumerate(products):
        with cols[i % len(cols)]:
            inv_inputs[prod] = st.number_input(
                f"{prod}", min_value=0, value=DEFAULT_CURRENT_INVENTORY,
                key=f"inv_{prod}",
            )

    inv_df = analyze_inventory(filtered_df)

    # Generate forecast demand for risk report
    forecast_demand = {}
    for prod in products:
        try:
            fc = pipeline.forecast(prod, weeks=DEFAULT_FORECAST_HORIZON, df=df)
            forecast_demand[prod] = fc['forecast'].mean()
        except Exception:
            forecast_demand[prod] = 0

    risk_report = build_risk_report(
        inv_df,
        current_inventory_map=inv_inputs,
        abc_classes=abc_map,
        forecast_demand=forecast_demand,
    )

    st.markdown('<div class="section-header">Risk Report</div>',
                unsafe_allow_html=True)

    # Color-code risk
    def color_risk(val):
        colors = {
            'Stockout Risk': 'background-color: #fee2e2',
            'Reorder Soon': 'background-color: #fef3c7',
            'Monitor': 'background-color: #dbeafe',
            'Healthy': 'background-color: #d1fae5',
            'Overstock': 'background-color: #ede9fe',
        }
        return colors.get(val, '')

    styled = risk_report.style.applymap(color_risk, subset=['Risk'])
    st.dataframe(risk_report, use_container_width=True, hide_index=True)

    # Risk distribution chart
    st.divider()
    risk_counts = get_risk_summary(risk_report)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Risk Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        risk_colors = {
            'Stockout Risk': '#ef4444', 'Reorder Soon': '#f59e0b',
            'Monitor': '#3b82f6', 'Healthy': '#10b981', 'Overstock': '#8b5cf6',
        }
        cats = list(risk_counts.keys())
        vals = list(risk_counts.values())
        bar_colors = [risk_colors.get(c, '#667eea') for c in cats]
        ax.bar(cats, vals, color=bar_colors)
        ax.set_ylabel("Products")
        ax.tick_params(axis='x', rotation=20)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Action Summary</div>',
                    unsafe_allow_html=True)
        action_counts = risk_report['Recommended Action'].value_counts().reset_index()
        action_counts.columns = ['Action', 'Count']
        st.dataframe(action_counts, use_container_width=True, hide_index=True)

    # CSV Export
    st.divider()
    st.markdown('<div class="section-header">Export Report</div>',
                unsafe_allow_html=True)
    csv_data = risk_report.to_csv(index=False)
    st.download_button(
        "📥 Download Risk Report (CSV)",
        data=csv_data,
        file_name="inventory_risk_report.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PRODUCT INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Product Intelligence":
    st.title("🔍 Product Intelligence")
    st.caption("Complete SKU-level decision view")

    product = st.selectbox("Select Product", sorted(df['product'].unique()))
    prod_data = df[df['product'] == product]
    weekly = prod_data.groupby('date')['units_sold'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"₹{prod_data['revenue'].sum():,.0f}")
    with col2:
        st.metric("Avg Weekly Demand", f"{weekly.mean():.1f}")
    with col3:
        st.metric("Demand Std Dev", f"{weekly.std():.1f}" if len(weekly) > 1 else "N/A")
    with col4:
        st.metric("ABC Class", abc_map.get(product, 'N/A'))

    st.divider()

    # Forecast
    history, forecast_fc = pipeline.get_product_forecast_with_history(product, df, 4)

    st.markdown('<div class="section-header">Demand History & Forecast</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history['date'], history['units_sold'],
            label='Historical', color='#667eea', linewidth=2, marker='o', markersize=4)
    fc_dates = pd.to_datetime(forecast_fc['date'])
    ax.plot(fc_dates, forecast_fc['forecast'],
            label='Forecast', color='#f093fb', linewidth=2, linestyle='--', marker='s')
    ax.fill_between(fc_dates, forecast_fc['lower_bound'], forecast_fc['upper_bound'],
                    alpha=0.2, color='#f093fb', label='Prediction Interval')
    ax.axvline(x=history['date'].iloc[-1], color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Inventory metrics
    st.divider()
    inv_single = analyze_inventory(prod_data)
    if len(inv_single) > 0:
        row = inv_single.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Safety Stock", f"{row['Safety Stock']:.0f} units")
        with c2:
            st.metric("Reorder Point", f"{row['Reorder Point']:.0f} units")
        with c3:
            st.metric("EOQ", f"{row['EOQ']:.0f} units")
        with c4:
            st.metric("Annual Demand", f"{row['Annual Demand Est.']:,}")

    # Recommendation
    st.divider()
    risk_single = build_risk_report(
        inv_single,
        default_inventory=DEFAULT_CURRENT_INVENTORY,
        abc_classes=abc_map,
    )
    if len(risk_single) > 0:
        rec = risk_single.iloc[0]
        st.markdown('<div class="section-header">Recommendation</div>',
                    unsafe_allow_html=True)
        risk_val = rec['Risk']
        action_val = rec['Recommended Action']
        order_val = rec['Recommended Order Qty']

        if risk_val == 'Stockout Risk':
            st.error(f"🔴 **{risk_val}** — {action_val}. Recommended order: **{order_val:.0f} units**")
        elif risk_val == 'Reorder Soon':
            st.warning(f"🟡 **{risk_val}** — {action_val}. Recommended order: **{order_val:.0f} units**")
        else:
            st.success(f"🟢 **{risk_val}** — {action_val}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SCENARIO SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Scenario Simulator":
    st.title("🎯 What-If Scenario Simulator")
    st.caption(
        "Explore how changes in supply chain parameters affect inventory decisions. "
        "Adjust inputs to answer questions like: \"What if lead time increases?\" "
        "or \"What if we target a higher service level?\""
    )

    st.info(
        "💡 All values below are **scenario inputs** for simulation. "
        "Adjust them to see the impact on safety stock, reorder points, "
        "EOQ, and risk classifications."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Supply Chain Parameters**")
        sim_lead_time = st.slider(
            "Lead Time (weeks)", min_value=1, max_value=8,
            value=DEFAULT_LEAD_TIME_WEEKS,
        )
        sim_service_level = st.select_slider(
            "Service Level",
            options=[0.80, 0.85, 0.90, 0.95, 0.99],
            value=DEFAULT_SERVICE_LEVEL,
            format_func=lambda x: f"{x*100:.0f}%",
        )
        sim_order_cost = st.number_input(
            "Ordering Cost (₹)", min_value=100, max_value=5000,
            value=int(DEFAULT_ORDERING_COST), step=100,
        )
        sim_holding_pct = st.slider(
            "Holding Cost (%)", min_value=5, max_value=50,
            value=int(DEFAULT_HOLDING_COST_PCT * 100),
        ) / 100.0

    with col2:
        st.markdown("**Inventory Position (Simulation)**")
        sim_product = st.selectbox("Product", sorted(df['product'].unique()),
                                    key="sim_product")
        sim_current_inv = st.number_input(
            "Current Inventory (units)", min_value=0, max_value=1000,
            value=DEFAULT_CURRENT_INVENTORY, key="sim_inv",
        )
        sim_forecast_horizon = st.slider(
            "Forecast Horizon (weeks)",
            min_value=MIN_FORECAST_HORIZON,
            max_value=MAX_FORECAST_HORIZON,
            value=DEFAULT_FORECAST_HORIZON,
            key="sim_horizon",
        )

    if st.button("Run Scenario", type="primary", use_container_width=True):
        st.divider()

        # Compute with scenario parameters
        inv_scenario = analyze_inventory(
            df, lead_time_weeks=sim_lead_time,
            service_level=sim_service_level,
            ordering_cost=sim_order_cost,
            holding_cost_pct=sim_holding_pct,
        )

        # Focus on selected product
        prod_row = inv_scenario[inv_scenario['Product'] == sim_product]
        if len(prod_row) > 0:
            row = prod_row.iloc[0]

            st.markdown(f'<div class="section-header">Scenario Results: {sim_product}</div>',
                        unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Safety Stock", f"{row['Safety Stock']:.0f} units")
            with c2:
                st.metric("Reorder Point", f"{row['Reorder Point']:.0f} units")
            with c3:
                st.metric("EOQ", f"{row['EOQ']:.0f} units")
            with c4:
                st.metric("Annual Demand", f"{row['Annual Demand Est.']:,}")

            # Forecast with scenario horizon
            fc = pipeline.forecast(sim_product, sim_forecast_horizon, df)
            fc_demand = fc['forecast'].sum()

            st.markdown(f'<div class="section-header">Projected Demand ({sim_forecast_horizon} weeks)</div>',
                        unsafe_allow_html=True)
            st.metric("Total Projected Demand", f"{fc_demand:.0f} units")

            # Risk with scenario inventory
            risk_sc = build_risk_report(
                prod_row,
                current_inventory_map={sim_product: sim_current_inv},
                abc_classes=abc_map,
            )
            if len(risk_sc) > 0:
                rec = risk_sc.iloc[0]
                st.markdown('<div class="section-header">Risk Assessment</div>',
                            unsafe_allow_html=True)
                risk_val = rec['Risk']
                action_val = rec['Recommended Action']
                order_val = rec['Recommended Order Qty']

                if risk_val == 'Stockout Risk':
                    st.error(f"🔴 **{risk_val}** — {action_val}. Order: **{order_val:.0f} units**")
                elif risk_val == 'Reorder Soon':
                    st.warning(f"🟡 **{risk_val}** — {action_val}. Order: **{order_val:.0f} units**")
                elif risk_val == 'Overstock':
                    st.info(f"📦 **{risk_val}** — {action_val}")
                else:
                    st.success(f"🟢 **{risk_val}** — {action_val}")

            # Compare default vs scenario
            st.divider()
            st.markdown('<div class="section-header">Default vs Scenario Comparison</div>',
                        unsafe_allow_html=True)
            default_row = analyze_inventory(df).set_index('Product').loc[sim_product]
            comparison = pd.DataFrame({
                'Metric': ['Safety Stock', 'Reorder Point', 'EOQ'],
                'Default': [
                    default_row['Safety Stock'],
                    default_row['Reorder Point'],
                    default_row['EOQ'],
                ],
                'Scenario': [
                    row['Safety Stock'],
                    row['Reorder Point'],
                    row['EOQ'],
                ],
            })
            comparison['Change'] = comparison['Scenario'] - comparison['Default']
            comparison['Change (%)'] = np.where(
                comparison['Default'] > 0,
                ((comparison['Scenario'] - comparison['Default']) / comparison['Default'] * 100).round(1),
                0,
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SALES ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Sales Analytics":
    st.title("📈 Sales Analytics")
    st.caption("Historical demand patterns across products and time")

    # Product trends
    product_sel = st.multiselect(
        "Select Products to Compare",
        options=sorted(df['product'].unique()),
        default=sorted(df['product'].unique())[:3],
    )

    if product_sel:
        trend_df = df[df['product'].isin(product_sel)]
        fig, ax = plt.subplots(figsize=(10, 4))
        colors_list = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a']
        for i, prod in enumerate(product_sel):
            p_df = trend_df[trend_df['product'] == prod].groupby('date')['units_sold'].sum()
            ax.plot(p_df.index, p_df.values, label=prod,
                    color=colors_list[i % len(colors_list)], linewidth=2,
                    marker='o', markersize=4)
        ax.set_xlabel("Date")
        ax.set_ylabel("Units Sold")
        ax.legend(loc='upper left')
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.markdown('<div class="section-header">Weekly Sales Table</div>',
                    unsafe_allow_html=True)
        pivot = trend_df.groupby(['date', 'product'])['units_sold'].sum().unstack(fill_value=0)
        pivot.index = pivot.index.strftime('%Y-%m-%d')
        st.dataframe(pivot, use_container_width=True)
    else:
        st.warning("Please select at least one product.")

    # Category performance
    st.divider()
    st.markdown('<div class="section-header">Category Performance</div>',
                unsafe_allow_html=True)
    cat_perf = compute_category_performance(filtered_df)
    st.dataframe(cat_perf, use_container_width=True, hide_index=True)

    # ABC Analysis
    st.divider()
    st.markdown('<div class="section-header">ABC Classification (Pareto Analysis)</div>',
                unsafe_allow_html=True)
    st.caption(
        "Products classified by revenue contribution: "
        "A (top ~70%), B (next ~20%), C (remaining ~10%). "
        "Based on the 70/20/10 Pareto principle."
    )
    st.dataframe(abc_df, use_container_width=True, hide_index=True)

    # Pareto chart
    fig, ax1 = plt.subplots(figsize=(8, 4))
    bar_colors = {'A': '#667eea', 'B': '#f093fb', 'C': '#cbd5e1'}
    colors = [bar_colors.get(c, '#667eea') for c in abc_df['ABC Class']]
    bars = ax1.bar(abc_df['Product'], abc_df['Value Percentage'], color=colors)
    ax1.set_ylabel("Value Contribution (%)")
    ax1.tick_params(axis='x', rotation=30)
    ax1.bar_label(bars, fmt='%.1f%%', padding=2, fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(abc_df['Product'], abc_df['Cumulative Percentage'],
             color='#ef4444', marker='D', linewidth=2, markersize=6)
    ax2.set_ylabel("Cumulative %")
    ax2.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='A/B boundary (70%)')
    ax2.axhline(y=90, color='gray', linestyle=':', alpha=0.5, label='B/C boundary (90%)')
    ax2.legend(fontsize=7, loc='center right')

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ABC Summary
    abc_sum = get_abc_summary(abc_df)
    st.dataframe(abc_sum, use_container_width=True, hide_index=True)

    # Demand insights
    prod_perf = compute_product_performance(df)
    demand_insights = generate_demand_insights(prod_perf)
    if demand_insights:
        st.divider()
        st.markdown('<div class="section-header">Demand Insights</div>',
                    unsafe_allow_html=True)
        for insight in demand_insights:
            st.info(f"{insight['icon']} **{insight['title']}**: {insight['message']}")

    # Data quality
    st.divider()
    with st.expander("📋 Data Quality Report"):
        st.json(quality_report)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Evaluation":
    st.title("🤖 Model Evaluation & Explainability")
    st.caption(
        "All metrics are computed on chronologically held-out test data. "
        "No training data was used for evaluation."
    )

    # Model comparison table
    st.markdown('<div class="section-header">Model Comparison</div>',
                unsafe_allow_html=True)
    st.caption(
        "Models evaluated using chronological train/test split. "
        "The last ~25% of dates form the test set."
    )
    comparison = model_results['comparison_table']
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.success(
        f"✅ **Best Model**: {model_results['best_model']} "
        f"(selected by lowest MAE on test set)"
    )

    # Improvement over baseline
    st.divider()
    st.markdown('<div class="section-header">Improvement over Naive Baseline</div>',
                unsafe_allow_html=True)
    improvement = model_results['improvement']
    if improvement:
        imp_df = pd.DataFrame([
            {'Metric': k, 'Improvement': v}
            for k, v in improvement.items()
        ])
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
        st.caption(
            "Improvement is calculated as: (Baseline - Model) / Baseline × 100 for error metrics. "
            "Positive values indicate the ML model performs better than the naive baseline."
        )
    else:
        st.info("Improvement data not available.")

    # Test metrics detail
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Best Model Test Metrics</div>',
                    unsafe_allow_html=True)
        test_met = model_results['test_metrics']
        for metric, value in test_met.items():
            st.metric(metric, value)

    with col2:
        st.markdown('<div class="section-header">Naive Baseline Metrics</div>',
                    unsafe_allow_html=True)
        base_met = model_results['baseline_metrics']
        for metric, value in base_met.items():
            st.metric(metric, value)

    # Feature importance
    st.divider()
    st.markdown('<div class="section-header">Feature Importance</div>',
                unsafe_allow_html=True)
    st.caption(
        "Feature importance indicates how much each feature contributes to the model's "
        "predictions. Higher importance means the model relies more heavily on that feature. "
        "**This does NOT establish causality** — it only reflects the model's learned associations."
    )

    fi = model_results['feature_importance']
    if fi is not None and len(fi) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(fi['Feature'], fi['Importance'], color='#667eea')
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(fi, use_container_width=True, hide_index=True)
    else:
        st.info("Feature importance not available for the selected model.")

    # Evaluation methodology
    st.divider()
    with st.expander("📖 Evaluation Methodology"):
        st.markdown(f"""
        **Chronological Validation**

        To prevent temporal data leakage, the dataset is split chronologically:
        - **Training set**: Earlier ~75% of dates
        - **Test set**: Later ~25% of dates

        No random shuffling is used — the model never sees future data during training.

        **Metrics**
        - **MAE** (Mean Absolute Error): Average prediction error in units
        - **RMSE** (Root Mean Squared Error): Penalizes larger errors more
        - **R²** (Coefficient of Determination): Proportion of variance explained
        - **MAPE** (Mean Absolute Percentage Error): Error as % of actual (when all actuals > 0)

        **Baseline**
        - Naive baseline: predicts the last known value for each product
        - If the ML model cannot beat this, it's not adding value

        **Prediction Intervals**
        - {int(PREDICTION_INTERVAL_LEVEL*100)}% residual-based prediction intervals
        - Computed from the standard deviation of training residuals
        - Formula: forecast ± z × residual_std (z = {PREDICTION_INTERVAL_Z})
        """)
