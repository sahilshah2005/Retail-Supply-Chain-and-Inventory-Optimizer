# 📦 Retail Supply Chain and Inventory Optimizer

An AI-powered demand forecasting and inventory optimization platform for retail supply chain decision support. Transforms historical sales data into actionable inventory management recommendations through ML-driven forecasting, risk analysis, and what-if simulation.

[![Python CI](https://github.com/sahilshah2005/Retail-Supply-Chain-and-Inventory-Optimizer/actions/workflows/python.yml/badge.svg)](https://github.com/sahilshah2005/Retail-Supply-Chain-and-Inventory-Optimizer/actions)

---

## 🎯 Problem Statement

Retail businesses face constant challenges in inventory management:
- **Stockouts** lead to lost sales and customer dissatisfaction
- **Overstocking** ties up capital and increases holding costs
- **Manual forecasting** is slow and error-prone
- **Disconnected analytics** make it hard to act on data

## 💡 Solution

This platform provides an end-to-end decision-support pipeline:

```
Historical Sales Data
      ↓
Data Validation & Quality Checks
      ↓
Engineered Demand Features (temporal, lag, rolling, trend)
      ↓
Chronological Model Training & Evaluation
      ↓
Model Comparison (Naive Baseline vs Random Forest vs Gradient Boosting)
      ↓
Demand Forecast with Prediction Intervals
      ↓
Inventory Optimization (Safety Stock, Reorder Point, EOQ)
      ↓
Risk Classification & ABC Analysis
      ↓
Actionable Reorder Recommendations
      ↓
Interactive Streamlit Dashboard
```

## 🚀 Key Features

### Forecasting
- **Multi-model comparison**: Naive baseline, Random Forest, Gradient Boosting
- **Chronological validation**: Strict time-ordered train/test split (no data leakage)
- **Engineered features**: Lag, rolling, trend, temporal, and product features
- **Prediction intervals**: 90% residual-based uncertainty bands
- **Proper forecast dates**: Forecasts start after last historical date, not system clock

### Inventory Optimization
- **Safety Stock**: z × σ_demand × √(Lead Time)
- **Reorder Point**: Avg Demand × Lead Time + Safety Stock
- **EOQ**: √(2DS/H) — classic cost-minimizing formula
- **Configurable parameters**: Lead time, service level, ordering cost, holding cost

### Risk Analysis
- **Risk classification**: Stockout Risk, Reorder Soon, Monitor, Healthy, Overstock
- **ABC/Pareto analysis**: Products classified by revenue contribution (70/20/10)
- **Priority matrix**: Combines risk severity with ABC class
- **Actionable recommendations**: Order Immediately, Reorder Soon, Monitor, No Action

### Dashboard (8 Pages)
| Page | Purpose |
|------|---------|
| 📊 Executive Dashboard | KPIs, alerts, revenue trends, risk overview |
| 🔮 Demand Forecasting | Product forecast with uncertainty bands |
| 📦 Inventory Optimization | EOQ, reorder point, safety stock analysis |
| ⚠️ Inventory Risk | Risk classification, reorder recommendations, CSV export |
| 🔍 Product Intelligence | Single-SKU deep-dive with full decision view |
| 🎯 Scenario Simulator | What-if analysis for supply chain parameters |
| 📈 Sales Analytics | Trends, ABC analysis, category performance |
| 🤖 Model Evaluation | Comparison table, feature importance, methodology |

### Additional Capabilities
- **Data quality validation**: Checks for missing values, duplicates, negatives, invalid dates
- **Model explainability**: Feature importance visualization with causality disclaimer
- **CSV export**: Downloadable inventory risk reports
- **What-if simulation**: Adjust lead time, service level, inventory to see impact

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML | Scikit-Learn (RandomForest, HistGradientBoosting) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Testing | pytest (70 tests) |
| CI | GitHub Actions |

## 📁 Project Structure

```
Retail Supply Chain and Inventory Optimizer/
├── src/
│   ├── app.py                       # Streamlit dashboard (main entry point)
│   ├── forecasting/
│   │   ├── features.py              # Feature engineering pipeline
│   │   ├── models.py                # Model training, comparison, prediction
│   │   └── evaluation.py            # Chronological evaluation & metrics
│   ├── inventory/
│   │   ├── optimizer.py             # Safety Stock, ROP, EOQ
│   │   ├── risk.py                  # Risk classification & recommendations
│   │   └── abc_analysis.py          # ABC/Pareto classification
│   ├── analytics/
│   │   ├── metrics.py               # Business KPI computation
│   │   └── insights.py              # Automated alerts & insights
│   └── utils/
│       ├── config.py                # Central configuration
│       └── data_loader.py           # Data loading & validation
├── data/
│   └── retail_sales.csv             # Weekly retail sales dataset
├── tests/
│   └── test_suite.py                # 70 comprehensive tests
├── models/                          # Generated at runtime
├── reports/                         # Generated via CSV export
├── .streamlit/config.toml           # Streamlit theme
├── .github/workflows/python.yml     # CI pipeline
├── ARCHITECTURE.md                  # Technical architecture docs
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧠 Forecasting Methodology

### Feature Engineering
Features are designed to prevent data leakage — all lag/rolling features use only past values:

| Feature Type | Features | Purpose |
|-------------|----------|---------|
| Temporal | week_of_year, month, quarter | Seasonal/cyclical patterns |
| Lag | lag_1, lag_2, lag_4 | Recent demand history |
| Rolling | rolling_mean_4, rolling_std_4 | Demand trends & variability |
| Trend | demand_growth | Momentum indicator |
| Product | product_encoded | Per-SKU behavior |

### Model Comparison
Models are evaluated on chronologically held-out test data (last ~25% of dates):

| Model | Description |
|-------|-------------|
| Naive Baseline | Last known value — lower-bound benchmark |
| Random Forest | Ensemble of decision trees (bagging) |
| Gradient Boosting | Sequential error correction (HistGradientBoosting) |

Best model is selected by lowest MAE on test set. Metrics are **never** computed on training data.

### Prediction Intervals
- **Method**: Residual-based intervals from training residuals
- **Level**: 90% (configurable)
- **Formula**: forecast ± 1.645 × residual_std

These are prediction intervals (range for future observations), not confidence intervals.

## 📊 Inventory Optimization

### Formulas
| Formula | Expression | Purpose |
|---------|-----------|---------|
| Safety Stock | z × σ_d × √LT | Buffer against variability |
| Reorder Point | d̄ × LT + SS | When to trigger order |
| EOQ | √(2DS/H) | Optimal order quantity |

### ABC Classification
Products are classified by cumulative revenue contribution:
- **A**: Top ~70% of value — highest priority
- **B**: Next ~20% (70–90% cumulative)
- **C**: Remaining ~10% (90–100%)

Priority integrates ABC class with risk level: A-class + Stockout Risk = **Critical** priority.

## ⚙️ Installation & Running

### Prerequisites
- Python 3.10 or higher
- pip

### Local Setup
```bash
git clone https://github.com/sahilshah2005/Retail-Supply-Chain-and-Inventory-Optimizer.git
cd Retail-Supply-Chain-and-Inventory-Optimizer
pip install -r requirements.txt
streamlit run src/app.py
```

### Running Tests
```bash
python -m pytest tests/ -v
```

## 🚧 Limitations

These are genuine limitations of the current implementation:

- **Small dataset**: 90 records (5 products × 18 weeks). This limits model complexity and validation depth. The architecture is designed to scale to larger datasets.
- **Single store**: Data is from one store (S1). Multi-store patterns cannot be captured.
- **Limited seasonality**: 18-week window (Jan–Apr) provides limited seasonal pattern coverage.
- **No real inventory data**: Current inventory is a user-provided simulation input, not historical data.
- **Static prices**: Unit prices are constant in the dataset; price elasticity is not modeled.
- **No external factors**: Weather, promotions, holidays, and competitor actions are not included.

## 🔮 Future Enhancements

- Support for larger multi-store datasets
- Time series models (ARIMA, Prophet) for comparison
- Promotion/holiday effect modeling
- Automated data refresh pipeline
- Multi-warehouse optimization
- Demand sensing with external data sources

## 📋 Technical Highlights (for interviews)

1. **No data leakage**: Strict chronological train/test split. Lag/rolling features use `shift()` to prevent future information from leaking.
2. **Baseline comparison**: ML models are benchmarked against a naive baseline to demonstrate actual value-add.
3. **Modular architecture**: Clean separation into forecasting, inventory, analytics, and utils packages.
4. **70 automated tests**: Covers data loading, validation, feature engineering, model training, inventory calculations, risk classification, and integration flows.
5. **Explainability**: Feature importance with clear causality disclaimer.
6. **Decision support, not just prediction**: Converts forecasts into actionable reorder recommendations with risk-based prioritization.

---

**Author**: Sahil Shah  
**Built with**: Python · Scikit-Learn · Streamlit · Pandas · NumPy · Matplotlib
