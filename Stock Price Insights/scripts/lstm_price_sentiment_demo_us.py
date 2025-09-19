# lstm_price_sentiment_demo_us.py
"""
End-to-end demo: AAPL stock price + live Yahoo Finance news, FinBERT sentiment, multi-feature LSTM forecast.
"""

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
FEATURE = 'close'  # lowercase for yahooquery historical data
TICKER = 'AAPL'
SEQ_LENGTH = 20
EPOCHS = 50
HIDDEN_SIZE = 50
NUM_LAYERS = 2
LEARNING_RATE = 0.001

# ----------------------------
# Paths
# ----------------------------
SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '../outputs')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ----------------------------
# Fetch historical stock prices from Yahooquery
# ----------------------------
print(f"Fetching historical data for {TICKER}...")
t = Ticker(TICKER)
hist = t.history(period="6mo")  # last 6 months
if hist.empty:
    raise ValueError("No historical data fetched")
# Reset index for convenience
hist = hist.reset_index()[['date', FEATURE]]
hist.rename(columns={FEATURE:'price', 'date':'ds'}, inplace=True)
hist['ds'] = pd.to_datetime(hist['ds'])

# ----------------------------
# Load FinBERT
# ----------------------------
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def sentiment_to_score(label):
    if label == "positive":
        return 1
    elif label == "neutral":
        return 0
    else:
        return -1

# ----------------------------
# Fetch live news
# ----------------------------
print(f"Fetching live news for {TICKER}...")
news_data = []
raw_news = t.news()
news_list = raw_news.get(TICKER, []) if isinstance(raw_news, dict) else raw_news

for news in news_list:
    if not isinstance(news, dict):
        continue
    ts = news.get("providerPublishTime")
    title = news.get("title")
    if ts and title:
        news_data.append({'date': pd.to_datetime(ts, unit='s'), 'headline': title})

news_df = pd.DataFrame(news_data)
if news_df.empty:
    print("No news found, sentiment will be zero")
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
# Merge price + sentiment
# ----------------------------
merged_df = pd.merge(hist, daily_sentiment, left_on='ds', right_on='date', how='left')
merged_df['sentiment'] = merged_df['sentiment'].fillna(0)
merged_df = merged_df[['ds','price','sentiment']]

# ----------------------------
# Diagnostic Plot
# ----------------------------
fig, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(merged_df['ds'], merged_df['price'], color='blue', label='Price')
ax1.set_xlabel("Date")
ax1.set_ylabel("Price", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.bar(merged_df['ds'], merged_df['sentiment'], color='orange', alpha=0.3, label='Sentiment')
ax2.set_ylabel("Sentiment", color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
plt.title(f"{TICKER} Price vs Daily Sentiment (FinBERT)")
fig.tight_layout()
plt.show()

# ----------------------------
# Scale features
# ----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_df[['price','sentiment']].values)

def create_sequences_multi(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length,0])
    return np.array(X), np.array(y)

X, y = create_sequences_multi(scaled_data, SEQ_LENGTH)
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float().unsqueeze(1)
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float().unsqueeze(1)

# ----------------------------
# LSTM model
# ----------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,1)
    def forward(self,x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out

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

y_pred_rescaled = scaler.inverse_transform(np.hstack([y_pred,np.zeros_like(y_pred)]))[:,0]
y_test_rescaled = scaler.inverse_transform(np.hstack([y_test.numpy(),np.zeros_like(y_test.numpy())]))[:,0]

# ----------------------------
# Save CSV
# ----------------------------
output_file = os.path.join(OUTPUT_PATH, f"{TICKER}_lstm_forecast.csv")
pd.DataFrame({'Actual': y_test_rescaled, 'Predicted': y_pred_rescaled}).to_csv(output_file, index=False)
print(f"Forecast saved: {output_file}")

# ----------------------------
# Forecast Plot
# ----------------------------
plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled,label='Actual')
plt.plot(y_pred_rescaled,label='Predicted')
plt.title(f"{TICKER} LSTM Forecast with Price + FinBERT Sentiment")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()
