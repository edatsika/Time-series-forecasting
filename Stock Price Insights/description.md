# 📈 Stock Forecasting with Sentiment (LSTM + FinBERT)

This project demonstrates **stock price forecasting** enhanced with **sentiment analysis from financial news**.  
It combines:
- **Historical stock prices** (Yahoo Finance / CSV dataset)
- **Live financial news headlines** (Yahoo Finance API via `yahooquery`, if available for **U.S. stocks** or **European stocks** )
- **Sentiment analysis** using [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone)
- **LSTM neural networks** for time series forecasting

---

## Features
- Automatic fetching of stock **historical prices**
- Retrieval of **live news headlines** per ticker
- **FinBERT sentiment scoring** (positive = +1, neutral = 0, negative = -1)
- Multi-feature **LSTM training** (price + sentiment)
- Forecast visualization and CSV export of predictions
- Supports either 

---

## 📂 Scripts

### 1. `lstm_price_sentiment_demo_us.py`
- End-to-end demo for **Apple (AAPL)**.
- Fetches **6 months of historical prices** via Yahooquery.
- Pulls **live financial news** headlines.
- Computes **daily sentiment scores** using FinBERT.
- Trains an **LSTM** on price + sentiment.
- Outputs:
  - Forecast CSV: `outputs/AAPL_lstm_forecast.csv`
  - Diagnostic plots (Price vs Sentiment, Actual vs Predicted).

---

### 2. `lstm_forecast_with_finbert_live_fixed.py`
- Multi-stock forecasting for **European tickers**:  
  - `AM.PA` (LVMH – Paris)  
  - `SAP.DE` (SAP – Frankfurt)  
  - `SIE.DE` (Siemens – Frankfurt)  
- Loads historical data from:
  [european_stocks2.csv](data/european_stocks2.csv)  
- Attempts to fetch **live Yahoo Finance news** headlines per ticker.
- Computes FinBERT sentiment and merges with stock data.
- Trains an **LSTM** model per stock.
- Outputs:
  - Forecast CSVs: `outputs/{TICKER}_lstm_forecast_live.csv`
  - Forecast plots for each ticker.

