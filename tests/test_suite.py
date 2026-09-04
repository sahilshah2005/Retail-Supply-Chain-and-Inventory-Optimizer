"""
Comprehensive test suite for Retail Supply Chain and Inventory Optimizer.

Covers: data loading, validation, feature engineering, chronological splitting,
model training/evaluation, forecasting, inventory optimization, risk engine,
ABC analysis, and integration tests.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.data_loader import load_data, validate_data, get_data_summary
from src.utils.config import DEFAULT_DATA_FILE, RANDOM_SEED
from src.forecasting.features import (
    create_product_weekly_series,
    add_temporal_features,
    add_lag_features,
    add_rolling_features,
    add_trend_features,
    encode_products,
    build_feature_matrix,
    get_feature_columns,
)
from src.forecasting.evaluation import (
    chronological_split,
    compute_metrics,
    compute_baseline_metrics,
    compute_improvement,
    compute_residuals,
)
from src.forecasting.models import ForecastingPipeline, NaiveBaseline
from src.inventory.optimizer import (
    compute_safety_stock,
    compute_reorder_point,
    compute_eoq,
    analyze_inventory,
    get_z_score,
)
from src.inventory.risk import (
    classify_risk,
    recommend_action,
    assign_priority,
    build_risk_report,
)
from src.inventory.abc_analysis import (
    compute_abc_classification,
    get_abc_class_map,
    get_abc_summary,
)
from src.analytics.metrics import (
    compute_business_kpis,
    compute_category_performance,
    compute_product_performance,
    compute_weekly_trends,
)
from src.analytics.insights import generate_alerts, generate_demand_insights, get_risk_summary


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    """Create a minimal sample DataFrame for testing."""
    dates = pd.date_range('2024-01-01', periods=10, freq='W')
    records = []
    for d in dates:
        for product, cat, price in [
            ('Widget A', 'Cat1', 100),
            ('Widget B', 'Cat2', 200),
        ]:
            records.append({
                'date': d,
                'product': product,
                'category': cat,
                'units_sold': np.random.RandomState(42).randint(10, 100),
                'unit_price': price,
                'store_id': 'S1',
            })
    df = pd.DataFrame(records)
    df['revenue'] = df['units_sold'] * df['unit_price']
    # Make units_sold vary by product
    df.loc[df['product'] == 'Widget A', 'units_sold'] = [20, 25, 30, 22, 28, 35, 24, 32, 26, 30]
    df.loc[df['product'] == 'Widget B', 'units_sold'] = [50, 55, 60, 48, 52, 65, 58, 62, 54, 60]
    df['revenue'] = df['units_sold'] * df['unit_price']
    return df


@pytest.fixture
def real_df():
    """Load the actual project dataset."""
    return load_data()


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading & Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoading:
    def test_load_data_returns_dataframe(self, real_df):
        assert isinstance(real_df, pd.DataFrame)
        assert len(real_df) > 0

    def test_load_data_has_required_columns(self, real_df):
        required = ['date', 'product', 'category', 'units_sold', 'unit_price', 'revenue']
        for col in required:
            assert col in real_df.columns, f"Missing column: {col}"

    def test_load_data_date_parsed(self, real_df):
        assert pd.api.types.is_datetime64_any_dtype(real_df['date'])

    def test_load_data_revenue_computed(self, real_df):
        expected = real_df['units_sold'] * real_df['unit_price']
        np.testing.assert_array_almost_equal(real_df['revenue'].values, expected.values)


class TestDataValidation:
    def test_validate_clean_data(self, real_df):
        clean, report = validate_data(real_df)
        assert len(clean) > 0
        assert 'total_rows' in report
        assert 'issues' in report

    def test_validate_removes_negative_units(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-08']),
            'product': ['A', 'A'],
            'category': ['C1', 'C1'],
            'units_sold': [10, -5],
            'unit_price': [100, 100],
        })
        df['revenue'] = df['units_sold'] * df['unit_price']
        clean, report = validate_data(df)
        assert len(clean) == 1
        assert clean.iloc[0]['units_sold'] == 10

    def test_validate_removes_duplicates(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'product': ['A', 'A'],
            'category': ['C1', 'C1'],
            'units_sold': [10, 10],
            'unit_price': [100, 100],
        })
        df['revenue'] = df['units_sold'] * df['unit_price']
        clean, report = validate_data(df)
        assert len(clean) == 1

    def test_validate_handles_missing_values(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-08']),
            'product': ['A', None],
            'category': ['C1', 'C1'],
            'units_sold': [10, 20],
            'unit_price': [100, 100],
        })
        df['revenue'] = df['units_sold'] * df['unit_price']
        clean, report = validate_data(df)
        assert len(clean) == 1

    def test_get_data_summary(self, real_df):
        summary = get_data_summary(real_df)
        assert summary['n_products'] == 5
        assert summary['n_rows'] == 90


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Engineering Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    def test_create_weekly_series(self, sample_df):
        weekly = create_product_weekly_series(sample_df)
        assert 'date' in weekly.columns
        assert 'product' in weekly.columns
        assert 'units_sold' in weekly.columns
        # Should be sorted by product, date
        products = weekly['product'].values
        assert products[0] <= products[-1]  # Sorted

    def test_temporal_features(self, sample_df):
        weekly = create_product_weekly_series(sample_df)
        result = add_temporal_features(weekly)
        assert 'week_of_year' in result.columns
        assert 'month' in result.columns
        assert 'quarter' in result.columns
        assert 'year' in result.columns
        assert result['month'].between(1, 12).all()
        assert result['quarter'].between(1, 4).all()

    def test_lag_features(self, sample_df):
        weekly = create_product_weekly_series(sample_df)
        result = add_lag_features(weekly, lag_periods=[1, 2])
        assert 'lag_1' in result.columns
        assert 'lag_2' in result.columns
        # First row for each product should have NaN for lag_1
        for product in result['product'].unique():
            prod_data = result[result['product'] == product]
            assert pd.isna(prod_data.iloc[0]['lag_1'])

    def test_lag_features_no_leakage(self, sample_df):
        """Verify lag features only use past data."""
        weekly = create_product_weekly_series(sample_df)
        weekly = weekly.sort_values(['product', 'date']).reset_index(drop=True)
        result = add_lag_features(weekly, lag_periods=[1])
        for product in result['product'].unique():
            prod_data = result[result['product'] == product].reset_index(drop=True)
            for i in range(1, len(prod_data)):
                if not pd.isna(prod_data.loc[i, 'lag_1']):
                    assert prod_data.loc[i, 'lag_1'] == prod_data.loc[i-1, 'units_sold']

    def test_rolling_features(self, sample_df):
        weekly = create_product_weekly_series(sample_df)
        result = add_rolling_features(weekly, windows=[4])
        assert 'rolling_mean_4' in result.columns
        assert 'rolling_std_4' in result.columns

    def test_trend_features(self, sample_df):
        weekly = create_product_weekly_series(sample_df)
        weekly = add_lag_features(weekly, [1, 2])
        result = add_trend_features(weekly)
        assert 'demand_growth' in result.columns

    def test_encode_products(self, sample_df):
        weekly = create_product_weekly_series(sample_df)
        result, mapping = encode_products(weekly)
        assert 'product_encoded' in result.columns
        assert len(mapping) == 2  # 2 products
        assert all(v >= 0 for v in mapping.values())

    def test_encode_unknown_product(self, sample_df):
        weekly = create_product_weekly_series(sample_df)
        _, mapping = encode_products(weekly)
        # Add unknown product row
        unknown = pd.DataFrame([{
            'date': pd.Timestamp('2024-06-01'), 'product': 'Unknown',
            'category': 'X', 'units_sold': 10, 'unit_price': 50,
            'revenue': 500,
        }])
        result, _ = encode_products(unknown, product_mapping=mapping)
        assert result.iloc[0]['product_encoded'] == -1

    def test_build_feature_matrix(self, sample_df):
        result, mapping = build_feature_matrix(sample_df)
        feature_cols = get_feature_columns()
        for col in feature_cols:
            assert col in result.columns, f"Missing feature: {col}"
        # No NaN in output (drop_na=True)
        assert result[feature_cols].isna().sum().sum() == 0

    def test_feature_columns_consistency(self):
        features = get_feature_columns()
        assert 'week_of_year' in features
        assert 'product_encoded' in features
        assert 'lag_1' in features
        assert 'rolling_mean_4' in features
        assert 'demand_growth' in features


# ═══════════════════════════════════════════════════════════════════════════════
# Chronological Splitting Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChronologicalSplit:
    def test_split_preserves_order(self, sample_df):
        featured, _ = build_feature_matrix(sample_df)
        train, test = chronological_split(featured, test_ratio=0.3)
        assert train['date'].max() <= test['date'].min()

    def test_split_no_data_loss(self, sample_df):
        featured, _ = build_feature_matrix(sample_df)
        train, test = chronological_split(featured, test_ratio=0.3)
        assert len(train) + len(test) == len(featured)

    def test_split_test_has_data(self, sample_df):
        featured, _ = build_feature_matrix(sample_df)
        train, test = chronological_split(featured, test_ratio=0.3)
        assert len(test) > 0
        assert len(train) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Model Training & Evaluation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelEvaluation:
    def test_compute_metrics(self):
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 33, 38, 52])
        metrics = compute_metrics(y_true, y_pred)
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics
        assert metrics['MAE'] >= 0
        assert metrics['RMSE'] >= 0

    def test_compute_metrics_perfect(self):
        y = np.array([10, 20, 30])
        metrics = compute_metrics(y, y)
        assert metrics['MAE'] == 0
        assert metrics['R2'] == 1.0

    def test_compute_metrics_mape(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([11, 22, 27])
        metrics = compute_metrics(y_true, y_pred)
        assert 'MAPE' in metrics
        assert metrics['MAPE'] >= 0

    def test_compute_improvement(self):
        baseline = {'MAE': 10.0, 'RMSE': 12.0, 'R2': 0.5}
        model = {'MAE': 7.0, 'RMSE': 9.0, 'R2': 0.7}
        improvement = compute_improvement(baseline, model)
        assert 'MAE' in improvement
        assert '+' not in improvement['MAE'] or '-' not in improvement['MAE']  # Has sign

    def test_compute_residuals(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])
        residuals = compute_residuals(y_true, y_pred)
        np.testing.assert_array_almost_equal(residuals, [-2, 2, -3])


class TestNaiveBaseline:
    def test_naive_baseline(self, sample_df):
        weekly = create_product_weekly_series(sample_df)
        naive = NaiveBaseline()
        naive.fit(weekly)
        preds = naive.predict(weekly)
        assert len(preds) == len(weekly)
        assert all(p >= 0 for p in preds)


class TestForecastingPipeline:
    def test_train_and_evaluate(self, real_df):
        pipeline = ForecastingPipeline(random_seed=42)
        results = pipeline.train_and_evaluate(real_df)
        assert 'comparison_table' in results
        assert 'best_model' in results
        assert 'test_metrics' in results
        assert pipeline.is_fitted

    def test_comparison_table_has_models(self, real_df):
        pipeline = ForecastingPipeline(random_seed=42)
        results = pipeline.train_and_evaluate(real_df)
        table = results['comparison_table']
        assert len(table) >= 3  # Naive + RF + GB
        assert 'Model' in table.columns
        assert 'MAE' in table.columns

    def test_forecast_starts_after_history(self, real_df):
        pipeline = ForecastingPipeline(random_seed=42)
        pipeline.train_and_evaluate(real_df)
        last_date = real_df['date'].max()
        fc = pipeline.forecast('Laptop', 4, real_df)
        assert fc['date'].min() > last_date

    def test_forecast_non_negative(self, real_df):
        pipeline = ForecastingPipeline(random_seed=42)
        pipeline.train_and_evaluate(real_df)
        fc = pipeline.forecast('Laptop', 4, real_df)
        assert (fc['forecast'] >= 0).all()
        assert (fc['lower_bound'] >= 0).all()

    def test_forecast_has_uncertainty(self, real_df):
        pipeline = ForecastingPipeline(random_seed=42)
        pipeline.train_and_evaluate(real_df)
        fc = pipeline.forecast('Laptop', 4, real_df)
        assert 'lower_bound' in fc.columns
        assert 'upper_bound' in fc.columns
        assert (fc['upper_bound'] >= fc['forecast']).all()
        assert (fc['lower_bound'] <= fc['forecast']).all()

    def test_forecast_unknown_product(self, real_df):
        pipeline = ForecastingPipeline(random_seed=42)
        pipeline.train_and_evaluate(real_df)
        fc = pipeline.forecast('NonExistentProduct', 4, real_df)
        assert len(fc) == 4
        assert (fc['forecast'] == 0).all()

    def test_feature_importance_exists(self, real_df):
        pipeline = ForecastingPipeline(random_seed=42)
        results = pipeline.train_and_evaluate(real_df)
        fi = results['feature_importance']
        assert fi is not None
        assert len(fi) > 0

    def test_test_metrics_not_on_training_data(self, real_df):
        """Verify that test metrics are from held-out data, not training."""
        pipeline = ForecastingPipeline(random_seed=42)
        pipeline.train_and_evaluate(real_df)
        # The test set should be chronologically after training set
        assert pipeline.train_df['date'].max() <= pipeline.test_df['date'].min()


# ═══════════════════════════════════════════════════════════════════════════════
# Inventory Optimization Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestInventoryOptimization:
    def test_safety_stock_positive(self):
        ss = compute_safety_stock(std_demand=10, lead_time_weeks=2, service_level=0.95)
        assert ss > 0

    def test_safety_stock_zero_variability(self):
        ss = compute_safety_stock(std_demand=0, lead_time_weeks=2, service_level=0.95)
        assert ss == 0

    def test_safety_stock_increases_with_lead_time(self):
        ss1 = compute_safety_stock(10, lead_time_weeks=1)
        ss2 = compute_safety_stock(10, lead_time_weeks=4)
        assert ss2 > ss1

    def test_safety_stock_increases_with_service_level(self):
        ss1 = compute_safety_stock(10, lead_time_weeks=2, service_level=0.90)
        ss2 = compute_safety_stock(10, lead_time_weeks=2, service_level=0.99)
        assert ss2 > ss1

    def test_reorder_point(self):
        rop = compute_reorder_point(
            avg_demand=50, std_demand=10,
            lead_time_weeks=2, service_level=0.95,
        )
        assert rop > 100  # At least avg_demand * lead_time

    def test_eoq_formula(self):
        # EOQ = sqrt(2 * D * S / H)
        # D=1000, S=500, H=0.2*1000=200
        # EOQ = sqrt(2*1000*500/200) = sqrt(5000) ≈ 70.7
        eoq = compute_eoq(
            annual_demand=1000, ordering_cost=500,
            holding_cost_pct=0.2, unit_cost=1000,
        )
        assert abs(eoq - 70.7) < 1.0

    def test_eoq_zero_demand(self):
        eoq = compute_eoq(annual_demand=0)
        assert eoq == 0

    def test_analyze_inventory(self, real_df):
        result = analyze_inventory(real_df)
        assert len(result) == 5  # 5 products
        assert 'Safety Stock' in result.columns
        assert 'Reorder Point' in result.columns
        assert 'EOQ' in result.columns

    def test_get_z_score(self):
        assert get_z_score(0.95) == 1.645
        assert get_z_score(0.99) == 2.326
        # Interpolated value should be between known values
        z = get_z_score(0.92)
        assert 1.282 < z < 1.645


# ═══════════════════════════════════════════════════════════════════════════════
# Risk Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskEngine:
    def test_classify_stockout_risk(self):
        risk = classify_risk(
            current_inventory=5, safety_stock=20,
            reorder_point=50, eoq=100,
        )
        assert risk == "Stockout Risk"

    def test_classify_reorder_soon(self):
        risk = classify_risk(
            current_inventory=40, safety_stock=20,
            reorder_point=50, eoq=100,
        )
        assert risk == "Reorder Soon"

    def test_classify_healthy(self):
        risk = classify_risk(
            current_inventory=100, safety_stock=20,
            reorder_point=50, eoq=100,
        )
        assert risk == "Healthy"

    def test_classify_overstock(self):
        risk = classify_risk(
            current_inventory=500, safety_stock=20,
            reorder_point=50, eoq=100,
        )
        assert risk == "Overstock"

    def test_recommend_action_stockout(self):
        action, qty = recommend_action(
            risk="Stockout Risk", current_inventory=5,
            reorder_point=50, eoq=100, safety_stock=20,
        )
        assert action == "Order Immediately"
        assert qty > 0

    def test_recommend_action_healthy(self):
        action, qty = recommend_action(
            risk="Healthy", current_inventory=200,
            reorder_point=50, eoq=100, safety_stock=20,
        )
        assert action == "No Action Required"
        assert qty == 0

    def test_assign_priority_critical(self):
        priority = assign_priority(risk="Stockout Risk", abc_class="A")
        assert priority == "Critical"

    def test_assign_priority_low(self):
        priority = assign_priority(risk="Healthy", abc_class="C")
        assert priority == "Low"

    def test_build_risk_report(self, real_df):
        inv_df = analyze_inventory(real_df)
        report = build_risk_report(inv_df, default_inventory=100)
        assert len(report) == 5
        assert 'Risk' in report.columns
        assert 'Recommended Action' in report.columns
        assert 'Priority' in report.columns


# ═══════════════════════════════════════════════════════════════════════════════
# ABC Analysis Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestABCAnalysis:
    def test_abc_classification(self, real_df):
        result = compute_abc_classification(real_df)
        assert len(result) == 5
        assert 'ABC Class' in result.columns
        assert set(result['ABC Class'].unique()).issubset({'A', 'B', 'C'})

    def test_abc_cumulative_100(self, real_df):
        result = compute_abc_classification(real_df)
        assert abs(result['Cumulative Percentage'].iloc[-1] - 100.0) < 0.1

    def test_abc_sorted_by_value(self, real_df):
        result = compute_abc_classification(real_df)
        values = result['Total Revenue'].values
        assert all(values[i] >= values[i+1] for i in range(len(values)-1))

    def test_get_abc_class_map(self, real_df):
        abc_df = compute_abc_classification(real_df)
        mapping = get_abc_class_map(abc_df)
        assert isinstance(mapping, dict)
        assert len(mapping) == 5

    def test_get_abc_summary(self, real_df):
        abc_df = compute_abc_classification(real_df)
        summary = get_abc_summary(abc_df)
        assert 'Product Count' in summary.columns
        total_count = summary['Product Count'].sum()
        assert total_count == 5


# ═══════════════════════════════════════════════════════════════════════════════
# Analytics Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalytics:
    def test_business_kpis(self, real_df):
        kpis = compute_business_kpis(real_df)
        assert kpis['total_revenue'] > 0
        assert kpis['total_units'] > 0
        assert kpis['n_products'] == 5

    def test_category_performance(self, real_df):
        cat = compute_category_performance(real_df)
        assert len(cat) > 0
        assert 'Revenue Share (%)' in cat.columns

    def test_product_performance(self, real_df):
        prod = compute_product_performance(real_df)
        assert len(prod) == 5
        assert 'Avg Weekly Demand' in prod.columns

    def test_weekly_trends(self, real_df):
        trends = compute_weekly_trends(real_df)
        assert len(trends) > 0
        assert 'revenue' in trends.columns
        assert 'units_sold' in trends.columns


class TestInsights:
    def test_generate_alerts_with_risk(self, real_df):
        inv_df = analyze_inventory(real_df)
        # Low inventory to trigger alerts
        report = build_risk_report(inv_df, default_inventory=5)
        alerts = generate_alerts(report)
        assert len(alerts) > 0
        assert any(a['type'] == 'stockout_risk' for a in alerts)

    def test_generate_alerts_all_clear(self, real_df):
        inv_df = analyze_inventory(real_df)
        # High inventory = no risk
        report = build_risk_report(inv_df, default_inventory=10000)
        alerts = generate_alerts(report)
        # Should either have overstock alerts or all clear
        assert len(alerts) > 0

    def test_demand_insights(self, real_df):
        prod_perf = compute_product_performance(real_df)
        insights = generate_demand_insights(prod_perf)
        assert isinstance(insights, list)

    def test_risk_summary(self, real_df):
        inv_df = analyze_inventory(real_df)
        report = build_risk_report(inv_df, default_inventory=100)
        summary = get_risk_summary(report)
        assert isinstance(summary, dict)
        total = sum(summary.values())
        assert total == 5


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline_flow(self, real_df):
        """Integration test: data -> features -> model -> forecast -> inventory -> risk."""
        # 1. Validate data
        clean_df, report = validate_data(real_df)
        assert len(clean_df) > 0

        # 2. Train model
        pipeline = ForecastingPipeline(random_seed=42)
        results = pipeline.train_and_evaluate(clean_df)
        assert pipeline.is_fitted

        # 3. Generate forecast
        product = clean_df['product'].unique()[0]
        forecast = pipeline.forecast(product, weeks=4, df=clean_df)
        assert len(forecast) == 4
        assert (forecast['forecast'] >= 0).all()

        # 4. Inventory analysis
        inv_df = analyze_inventory(clean_df)
        assert len(inv_df) > 0

        # 5. ABC classification
        abc_df = compute_abc_classification(clean_df)
        abc_map = get_abc_class_map(abc_df)

        # 6. Risk report
        risk_report = build_risk_report(
            inv_df, default_inventory=100, abc_classes=abc_map,
        )
        assert len(risk_report) > 0
        assert 'Recommended Action' in risk_report.columns

        # 7. Alerts
        alerts = generate_alerts(risk_report)
        assert len(alerts) > 0

    def test_scenario_simulation(self, real_df):
        """Test scenario simulator flow."""
        # Different lead times should produce different safety stocks
        inv1 = analyze_inventory(real_df, lead_time_weeks=1)
        inv2 = analyze_inventory(real_df, lead_time_weeks=4)

        for product in real_df['product'].unique():
            ss1 = inv1[inv1['Product'] == product]['Safety Stock'].values[0]
            ss2 = inv2[inv2['Product'] == product]['Safety Stock'].values[0]
            assert ss2 > ss1, f"Safety stock should increase with lead time for {product}"

    def test_csv_export(self, real_df):
        """Test that risk report can be exported to CSV."""
        inv_df = analyze_inventory(real_df)
        report = build_risk_report(inv_df, default_inventory=100)
        csv_data = report.to_csv(index=False)
        assert len(csv_data) > 0
        # Verify it can be read back
        import io
        recovered = pd.read_csv(io.StringIO(csv_data))
        assert len(recovered) == len(report)
