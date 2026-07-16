# S&P 500 Stock Market ML Prediction

A machine learning pipeline that predicts next-day stock direction (UP / DOWN) for 10 S&P 500 stocks using technical indicators and ensemble classifiers.

## Stocks Covered

| Ticker | Company | Sector |
|--------|---------|--------|
| AAPL | Apple | Technology |
| MSFT | Microsoft | Technology |
| GOOGL | Alphabet | Communication Services |
| AMZN | Amazon | Consumer Discretionary |
| NVDA | NVIDIA | Semiconductors |
| JPM | JPMorgan Chase | Financials |
| JNJ | Johnson & Johnson | Healthcare |
| XOM | ExxonMobil | Energy |
| PG | Procter & Gamble | Consumer Staples |
| HD | Home Depot | Consumer Discretionary |

## How It Works

1. **Data Collection** -- 6 years of daily OHLCV data (Jan 2020 - Jun 2026) pulled from yfinance
2. **Feature Engineering** -- 20+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, momentum, volatility, lagged returns)
3. **Model Training** -- Logistic Regression, Random Forest, XGBoost on an 80/20 time-ordered split
4. **Evaluation** -- Accuracy, Precision, Recall, F1, ROC AUC, confusion matrices, ROC/PR curves

## Setup & Usage

```bash
pip install -r requirements.txt
```

**Step 1** -- Run `collect_data.ipynb` to download stock data
**Step 2** -- Run `stock_ml_pipeline.ipynb` to train models and generate charts

## Results

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.529 | 0.685 | 0.509 |
| Random Forest | 0.535 | 0.679 | 0.518 |
| XGBoost | 0.506 | 0.575 | 0.495 |

> ~53% accuracy is consistent with institutional stock prediction baselines. Markets are inherently noisy.

## Visualizations

### Model Comparison
All metrics side-by-side across the three classifiers.

![Model Comparison](images/model_comparison.png)

### Confusion Matrices
True vs predicted labels for each model -- shows how often each model calls UP vs DOWN correctly.

![Confusion Matrices](images/confusion_matrices.png)

### ROC Curve
Trade-off between true positive rate and false positive rate. AUC scores close to 0.5 indicate near-random performance, which is expected for stock prediction.

![ROC Curve](images/roc_curve.png)

### Feature Importance
Top 20 most influential features for Random Forest and XGBoost.

![Feature Importance](images/feature_importance.png)

### Stock Technical Charts
Price action with SMA, RSI, MACD, and Bollinger Bands for the first 3 stocks.

| AAPL (Apple) | AMZN (Amazon) | GOOGL (Alphabet) |
|---|---|---|
| ![AAPL](images/stock_chart_AAPL.png) | ![AMZN](images/stock_chart_AMZN.png) | ![GOOGL](images/stock_chart_GOOGL.png) |

### Target Class Distribution
Balance between UP and DOWN days in the test set.

![Class Distribution](images/class_distribution.png)

## Project Structure

```
SP500 ML Stock Prediction/
├── collect_data.ipynb          # Step 1: download data from yfinance
├── stock_ml_pipeline.ipynb     # Step 2: feature engineering + ML + charts
├── sp500_stocks.csv            # Raw OHLCV data (10 stocks, 6 years)
├── requirements.txt
├── images/                     # Generated charts
├── LICENSE
└── README.md
```

## License

MIT
