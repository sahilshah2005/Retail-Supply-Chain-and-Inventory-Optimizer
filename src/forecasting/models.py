"""
Forecasting model training, comparison, and prediction.

Implements a model comparison framework with:
- Naive baseline (last known value)
- Random Forest Regressor
- Histogram-based Gradient Boosting Regressor

All models are evaluated using chronological validation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from .features import (
    build_feature_matrix,
    get_feature_columns,
    add_temporal_features,
    add_lag_features,
    add_rolling_features,
    add_trend_features,
    encode_products,
)
from .evaluation import (
    chronological_split,
    compute_metrics,
    compute_baseline_metrics,
    compute_improvement,
    compute_residuals,
)


class NaiveBaseline:
    """
    Naive forecast baseline: predicts the last known value for each product.

    This serves as a lower-bound benchmark. If ML models cannot beat this,
    they are not adding predictive value.
    """

    def __init__(self):
        self.last_values = {}

    def fit(self, df: pd.DataFrame):
        """Store the last known units_sold for each product."""
        for product, grp in df.groupby('product'):
            last_row = grp.sort_values('date').iloc[-1]
            self.last_values[product] = last_row['units_sold']

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using last known value per product."""
        preds = []
        for _, row in df.iterrows():
            product = row.get('product', '')
            preds.append(self.last_values.get(product, 0))
        return np.array(preds)


class ForecastingPipeline:
    """
    Complete forecasting pipeline with model comparison.

    Handles feature engineering, chronological train/test splitting,
    model training, evaluation, comparison, and future prediction
    with uncertainty estimation.

    Attributes
    ----------
    best_model_name : str
        Name of the best-performing model.
    comparison_results : pd.DataFrame
        Model comparison table with metrics.
    test_metrics : dict
        Metrics of the best model on test data.
    feature_importance : pd.DataFrame
        Feature importance of the best model.
    residual_std : float
        Standard deviation of residuals for prediction intervals.
    product_mapping : dict
        Product name to integer encoding.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.models = {}
        self.best_model_name = None
        self.best_model = None
        self.comparison_results = None
        self.test_metrics = None
        self.baseline_metrics = None
        self.improvement = None
        self.feature_importance = None
        self.residual_std = 0.0
        self.product_mapping = None
        self.feature_columns = None
        self.train_df = None
        self.test_df = None
        self.last_historical_date = None
        self.is_fitted = False

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict:
        """
        Full training pipeline:
        1. Feature engineering
        2. Chronological split
        3. Train multiple models
        4. Evaluate on test set
        5. Select best model
        6. Compute prediction interval statistics

        Parameters
        ----------
        df : pd.DataFrame
            Raw sales data.

        Returns
        -------
        Dict with keys: comparison_table, best_model, test_metrics,
        baseline_metrics, improvement, feature_importance
        """
        # Store last historical date
        self.last_historical_date = pd.to_datetime(df['date']).max()

        # Feature engineering
        feature_df, self.product_mapping = build_feature_matrix(df)
        self.feature_columns = get_feature_columns()

        # Chronological split
        self.train_df, self.test_df = chronological_split(feature_df, test_ratio=0.25)

        if len(self.test_df) == 0 or len(self.train_df) == 0:
            raise ValueError(
                "Not enough data for train/test split. "
                f"Feature matrix has {len(feature_df)} rows after NaN removal."
            )

        X_train = self.train_df[self.feature_columns]
        y_train = self.train_df['units_sold']
        X_test = self.test_df[self.feature_columns]
        y_test = self.test_df['units_sold']

        # --- Naive baseline ---
        naive = NaiveBaseline()
        naive.fit(self.train_df)
        naive_preds = naive.predict(self.test_df)
        self.baseline_metrics = compute_baseline_metrics(y_test.values, naive_preds)

        # --- ML Models ---
        model_configs = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=self.random_seed,
            ),
            'Gradient Boosting': HistGradientBoostingRegressor(
                max_iter=100,
                max_depth=None,
                random_state=self.random_seed,
            ),
        }

        results = []
        results.append({
            'Model': 'Naive Baseline',
            **self.baseline_metrics,
        })

        best_mae = float('inf')
        best_name = 'Naive Baseline'
        best_ml_model = None

        for name, model in model_configs.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            preds = np.maximum(preds, 0)  # Non-negative predictions
            metrics = compute_metrics(y_test.values, preds)

            results.append({'Model': name, **metrics})
            self.models[name] = model

            if metrics['MAE'] < best_mae:
                best_mae = metrics['MAE']
                best_name = name
                best_ml_model = model

        # Check if best ML model beats naive
        if best_mae < self.baseline_metrics['MAE']:
            self.best_model_name = best_name
            self.best_model = best_ml_model
        else:
            # If no ML model beats naive, still use best ML model
            # but document that baseline is competitive
            self.best_model_name = best_name
            self.best_model = best_ml_model if best_ml_model else list(self.models.values())[0]

        self.comparison_results = pd.DataFrame(results)

        # Test metrics of best model
        if self.best_model_name == 'Naive Baseline':
            self.test_metrics = self.baseline_metrics
        else:
            best_preds = self.best_model.predict(X_test)
            best_preds = np.maximum(best_preds, 0)
            self.test_metrics = compute_metrics(y_test.values, best_preds)

        # Improvement over baseline
        self.improvement = compute_improvement(self.baseline_metrics, self.test_metrics)

        # Feature importance (for tree-based models)
        self._compute_feature_importance()

        # Residual std for prediction intervals
        if self.best_model is not None:
            train_preds = self.best_model.predict(X_train)
            residuals = compute_residuals(y_train.values, train_preds)
            self.residual_std = np.std(residuals)

        self.is_fitted = True

        return {
            'comparison_table': self.comparison_results,
            'best_model': self.best_model_name,
            'test_metrics': self.test_metrics,
            'baseline_metrics': self.baseline_metrics,
            'improvement': self.improvement,
            'feature_importance': self.feature_importance,
        }

    def _compute_feature_importance(self):
        """Extract feature importance from the best tree-based model."""
        if self.best_model is None:
            self.feature_importance = pd.DataFrame()
            return

        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            self.feature_importance = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importances,
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
        else:
            self.feature_importance = pd.DataFrame()

    def forecast(
        self,
        product: str,
        weeks: int = 4,
        df: pd.DataFrame = None,
        prediction_interval_z: float = 1.645,
    ) -> pd.DataFrame:
        """
        Generate future demand forecasts for a product.

        Forecasts start from the day after the last historical date in the
        dataset (NOT from the current system date).

        Parameters
        ----------
        product : str
            Product name to forecast.
        weeks : int
            Number of weeks to forecast.
        df : pd.DataFrame, optional
            Historical data. If None, uses stored training data.
        prediction_interval_z : float
            Z-score for prediction interval. Default 1.645 (90% interval).

        Returns
        -------
        pd.DataFrame
            Forecast with columns: date, forecast, lower_bound, upper_bound
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not trained. Call train_and_evaluate first.")

        # Get product's last known data for lag features
        product_code = self.product_mapping.get(product, -1)

        if df is not None:
            product_hist = df[df['product'] == product].copy()
        else:
            # Combine train and test for full history
            full_df = pd.concat([self.train_df, self.test_df]).sort_values('date')
            product_hist = full_df[full_df['product'] == product].copy()

        if len(product_hist) == 0:
            # Unknown product: return zero forecast
            future_dates = [
                self.last_historical_date + pd.Timedelta(weeks=i)
                for i in range(1, weeks + 1)
            ]
            return pd.DataFrame({
                'date': future_dates,
                'forecast': [0] * weeks,
                'lower_bound': [0] * weeks,
                'upper_bound': [0] * weeks,
            })

        recent_values = product_hist.sort_values('date')['units_sold'].values

        forecasts = []
        # Build a running list of known values for lag computation
        running_values = list(recent_values)

        for i in range(1, weeks + 1):
            future_date = self.last_historical_date + pd.Timedelta(weeks=i)

            # Temporal features
            week_of_year = future_date.isocalendar()[1]
            month = future_date.month
            quarter = (future_date.month - 1) // 3 + 1

            # Lag features from running values
            lag_1 = running_values[-1] if len(running_values) >= 1 else 0
            lag_2 = running_values[-2] if len(running_values) >= 2 else lag_1
            lag_4 = running_values[-4] if len(running_values) >= 4 else lag_2

            # Rolling features
            if len(running_values) >= 4:
                rolling_mean_4 = np.mean(running_values[-4:])
                rolling_std_4 = np.std(running_values[-4:], ddof=1) if len(running_values[-4:]) > 1 else 0
            else:
                rolling_mean_4 = np.mean(running_values)
                rolling_std_4 = np.std(running_values, ddof=1) if len(running_values) > 1 else 0

            # Trend feature
            demand_growth = (lag_1 - lag_2) / lag_2 if lag_2 > 0 else 0

            feature_row = {
                'week_of_year': week_of_year,
                'month': month,
                'quarter': quarter,
                'product_encoded': product_code,
                'lag_1': lag_1,
                'lag_2': lag_2,
                'lag_4': lag_4,
                'rolling_mean_4': rolling_mean_4,
                'rolling_std_4': rolling_std_4,
                'demand_growth': demand_growth,
            }

            X = pd.DataFrame([feature_row])[self.feature_columns]
            pred = self.best_model.predict(X)[0]
            pred = max(pred, 0)  # Non-negative

            # Prediction interval (residual-based)
            lower = max(pred - prediction_interval_z * self.residual_std, 0)
            upper = pred + prediction_interval_z * self.residual_std

            forecasts.append({
                'date': future_date,
                'forecast': round(pred, 1),
                'lower_bound': round(lower, 1),
                'upper_bound': round(upper, 1),
            })

            # Add prediction to running values for next iteration's lags
            running_values.append(pred)

        return pd.DataFrame(forecasts)

    def get_product_forecast_with_history(
        self,
        product: str,
        df: pd.DataFrame,
        weeks: int = 4,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get both historical data and forecast for a product.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (history_df, forecast_df)
        """
        history = (
            df[df['product'] == product]
            .groupby('date')['units_sold'].sum()
            .reset_index()
            .sort_values('date')
        )
        forecast = self.forecast(product, weeks, df)
        return history, forecast
