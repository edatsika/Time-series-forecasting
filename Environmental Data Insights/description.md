
---

## 📊 Dataset

The dataset used comes from Kaggle:  
👉 [Air Quality Monitoring in European Cities](https://www.kaggle.com/datasets/yekenot/air-quality-monitoring-in-european-cities?resource=download)

- Extract `athens_data.csv` (or another city’s file) for training.  
- Contains hourly measurements of PM2.5, PM10, NO2, O3, weather variables, etc.  
- In this project, only the **PM2.5 column** is used.

---

## ⚙️ Parameters
- `SEQ_LENGTH = 14` → number of past days used as input.
- `EPOCHS = 20` → training epochs.
- `BATCH_SIZE = 32` → training batch size.
- `MODEL_FILE = "pm25_model.h5"` → saved model path.

---

## 📊 Workflow

### 1. Data Preprocessing
- Load `athens_data.csv`.
- Convert `Date` column to datetime.
- Keep only `Date` and `PM2.5`.
- Aggregate to **daily averages** with `resample("D")`.
- Keep only records from **2020 onwards**.

### 2. Scaling
- Normalize PM2.5 values to `[0,1]` using `MinMaxScaler`.

### 3. Sequence Creation
- Transform dataset into supervised form:
  - Inputs: 14 consecutive days of PM2.5.
  - Target: PM2.5 of the next day.
- Cache results (`X.npy`, `Y.npy`).

### 4. Model
- Load existing `pm25_model.h5` if available.
- Otherwise build & train a new model:
  - `LSTM(32)` → `Dropout(0.2)` → `Dense(1)`.
  - Optimizer: Adam.
  - Loss: Mean Squared Error (MSE).

### 5. Forecasting
- Function `forecast_next_days`:
  - Takes the last 14 days.
  - Predicts the next day.
  - Slides window & repeats for 7 days.
  - Inverse transforms predictions back to real PM2.5 values.

### 6. Visualization
Generates a **dual subplot figure**:
1. Full history with forecast (7 red X’s).
2. Last 30 days + forecast.

---

## Output
- **Full historical PM2.5** curve with forecast points.  
- **Zoomed last 30 days** with forecast continuation.
- Can be extended: Integrate live data feeds (e.g., OpenAQ API), extend forecast horizon with confidence intervals and add multi-city support (other European cities)

