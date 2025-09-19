import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# === Parameters ===
CSV_FILE = "athens_data.csv"
SEQ_LENGTH = 14  # past days to use
EPOCHS = 20
BATCH_SIZE = 32
MODEL_FILE = "pm25_model.h5"
X_FILE, Y_FILE = "X.npy", "Y.npy"

# === Load and preprocess data ===
df = pd.read_csv(CSV_FILE)

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Keep only Date + PM2.5
df = df[["Date", "PM2.5"]].dropna()

# Aggregate to daily averages
df = df.resample("D", on="Date").mean().reset_index()

# Keep only last 3 years
df = df[df["Date"] >= "2020-01-01"]

print("Data loaded, shape after daily aggregation:", df.shape)

# === Scaling ===
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_pm25 = scaler.fit_transform(df[["PM2.5"]])

# === Sequence creation (cache to speed up later runs) ===
def create_sequences(data, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

if os.path.exists(X_FILE) and os.path.exists(Y_FILE):
    X = np.load(X_FILE)
    y = np.load(Y_FILE)
    print("Loaded cached sequences:", X.shape, y.shape)
else:
    X, y = create_sequences(scaled_pm25, SEQ_LENGTH)
    np.save(X_FILE, X)
    np.save(Y_FILE, y)
    print("Created and cached sequences:", X.shape, y.shape)

# Reshape for LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# === Model ===
if os.path.exists(MODEL_FILE):
    print("Loading saved model...")
    model = load_model(MODEL_FILE)
else:
    print("Building new model...")
    model = Sequential([
        LSTM(32, input_shape=(SEQ_LENGTH, 1)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    model.fit(X, y, validation_split=0.2, epochs=EPOCHS,
              batch_size=BATCH_SIZE, verbose=1)
    model.save(MODEL_FILE)

# === Forecast next 7 days ===
def forecast_next_days(model, last_seq, days=7):
    predictions = []
    current_seq = last_seq.copy()
    for _ in range(days):
        pred = model.predict(current_seq.reshape(1, SEQ_LENGTH, 1), verbose=0)
        predictions.append(pred[0,0])
        current_seq = np.append(current_seq[1:], pred)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

last_seq = scaled_pm25[-SEQ_LENGTH:]
forecast = forecast_next_days(model, last_seq, days=7)

forecast_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
                               periods=7, freq="D")

# === Plotting ===
plt.figure(figsize=(14,6))

# Subplot 1: Full history
plt.subplot(2,1,1)
plt.plot(df["Date"], df["PM2.5"], label="Historical PM2.5")
plt.scatter(forecast_dates, forecast, color="red", marker="x", label="Forecast")
plt.title("PM2.5 - Full History with Forecast")
plt.legend()

# Subplot 2: Last 30 days + forecast
plt.subplot(2,1,2)
plt.plot(df["Date"].iloc[-30:], df["PM2.5"].iloc[-30:], label="Last 30 Days")
plt.scatter(forecast_dates, forecast, color="red", marker="x", label="Forecast")
plt.title("PM2.5 - Last 30 Days + Forecast")
plt.legend()

plt.tight_layout()
plt.show()
