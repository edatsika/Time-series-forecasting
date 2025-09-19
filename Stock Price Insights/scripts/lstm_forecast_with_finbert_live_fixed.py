# lstm_forecast_with_finbert_live_fixed.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from yahooquery import Ticker

# ----------------------------
# Config
# ----------------------------
FEATURE = 'Close'
TICKERS = ['AM.PA', 'SAP.DE', 'SIE.DE']
SEQ_LENGTH = 20
EPOCHS = 50
HIDDEN_SIZE = 50
NUM_LAYERS = 2
LEARNING_RATE = 0.001

# ----------------------------
# Paths
# ----------------------------
SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(SCRIPT_DIR, '../data/european_stocks2.csv')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '../outputs')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ----------------------------
# Load stock CSV
# ----------------------------
df = pd.read_csv(DATA_PATH, header=[0,1,2], index_col=0)

# ----------------------------
# FinBERT pipeline
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def sentiment_to_score(label):
    if label == 'positive':
        return 1
    elif label == 'neutral':
        return 0
    else:
        return -1

# ----------------------------
# Helper: create sequences
# ----------------------------
def create_sequences_multi(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # first column = price
    return np.array(X), np.array(y)

# ----------------------------
# LSTM Model
# ----------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ----------------------------
# Process each ticker
# ----------------------------
idx = pd.IndexSlice
for ticker in TICKERS:
    try:
        stock_series = df.loc[:, idx[FEATURE, ticker]].reset_index()
        stock_series.columns = ['ds', 'price']
    except KeyError:
        print(f"{ticker} not found, skipping")
        continue

    print(f"\nProcessing {ticker} with live Yahoo Finance news...")

    # Convert stock dates to datetime
    stock_series['ds'] = pd.to_datetime(stock_series['ds'], errors='coerce')
    stock_series.dropna(subset=['ds'], inplace=True)

    # ----------------------------
    # Fetch news dynamically using yahooquery
    # ----------------------------
    ticker_yq = Ticker(ticker)
    try:
        raw_news = ticker_yq.news()  # call the method
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        raw_news = []

    # Handle different return types
    if isinstance(raw_news, dict):
        news_list = raw_news.get(ticker, [])
    elif isinstance(raw_news, list):
        news_list = raw_news
    else:
        news_list = []

    # Filter only dict entries
    news_list = [n for n in news_list if isinstance(n, dict)]

    news_data = []
    for news in news_list:
        timestamp = news.get('providerPublishTime', None)
        title = news.get('title', '')
        if timestamp and title:
            news_data.append({
                'date': pd.to_datetime(timestamp, unit='s'),
                'headline': title
            })

    news_df = pd.DataFrame(news_data)
    if news_df.empty:
        print(f"No news found for {ticker}, sentiment will be zero")
        news_df = pd.DataFrame(columns=['date','headline'])

    # ----------------------------
    # Compute FinBERT sentiment
    # ----------------------------
    def get_sentiment(text):
        try:
            result = finbert_pipeline(str(text))[0]
            return sentiment_to_score(result['label'])
        except:
            return 0

    news_df['sentiment'] = news_df['headline'].apply(get_sentiment)
    daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()

    # ----------------------------
    # Merge stock and sentiment
    # ----------------------------
    merged_df = pd.merge(stock_series, daily_sentiment, left_on='ds', right_on='date', how='left')
    merged_df['sentiment'] = merged_df['sentiment'].fillna(0)
    merged_df = merged_df[['ds', 'price', 'sentiment']]

    # ----------------------------
    # Scale
    # ----------------------------
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(merged_df[['price', 'sentiment']].values)

    # ----------------------------
    # Create sequences
    # ----------------------------
    X, y = create_sequences_multi(scaled_data, SEQ_LENGTH)

    # ----------------------------
    # Train/test split
    # ----------------------------
    train_size = int(len(X)*0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # ----------------------------
    # Model
    # ----------------------------
    model = StockLSTM(input_size=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

    # ----------------------------
    # Prediction
    # ----------------------------
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
    y_pred_rescaled = scaler.inverse_transform(np.hstack([y_pred, np.zeros_like(y_pred)]))[:,0]
    y_test_rescaled = scaler.inverse_transform(np.hstack([y_test.numpy().reshape(-1,1), np.zeros_like(y_test.numpy().reshape(-1,1))]))[:,0]

    # ----------------------------
    # Save CSV
    # ----------------------------
    forecast_file = os.path.join(OUTPUT_PATH, f"{ticker}_lstm_forecast_live.csv")
    pd.DataFrame({'Actual': y_test_rescaled, 'Predicted': y_pred_rescaled}).to_csv(forecast_file, index=False)
    print(f"Forecast saved: {forecast_file}")

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(12,6))
    plt.plot(y_test_rescaled, label='Actual')
    plt.plot(y_pred_rescaled, label='Predicted')
    plt.title(f"{ticker} LSTM Forecast with Live Yahoo Finance News & FinBERT")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
