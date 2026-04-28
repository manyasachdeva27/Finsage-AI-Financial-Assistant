#  FinSage — Financial Time Series Analysis App

> An interactive Streamlit dashboard for personal finance forecasting, anomaly detection, and health scoring — powered by SARIMA, Prophet, Holt-Winters, XGBoost, and a weighted ensemble model.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Features

| Tab | Description |
|-----|-------------|
| 📊 **Overview** | Monthly income, expense & savings trends · correlation heatmap · top-10 spending categories |
| 🔍 **Decomposition** | Seasonal decomposition (trend / seasonality / residual) · rolling statistics · ADF stationarity test |
| 🤖 **Forecasting** | SARIMA (auto_arima) · Holt-Winters · Prophet (tuned) · XGBoost · Weighted Ensemble — with a model comparison table and 6-month forward risk report |
| ⚠️ **Anomalies** | Z-score + IQR anomaly detection with interactive chart and anomalous-month table |
| ❤️ **Health Score** | Per-month Financial Health Score (0–100) with risk classification (Stable / Moderate / High Risk) |
| 💡 **Insights** | Auto-generated narrative insights: trend direction, seasonality, savings risk, income volatility, anomaly alerts |
| 🗂️ **Clusters** | K-means category-level spending cluster profiles with adjustable k |

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/finsage.git
cd finsage
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run finsage_app.py
```

The app opens at `http://localhost:8501`. Upload your CSV from the sidebar to get started.

---

## 📁 Project Structure

```
finsage/
├── finsage_app.py              # Main Streamlit application
├── requirements.txt            # Python dependencies
├── sample_transactions.csv     # Sample 4-year transaction dataset
└── README.md                   # This file
```

---

## 📄 CSV Format

Your transactions file must contain these columns:

| Column | Type | Example |
|--------|------|---------|
| `Date` | date string | `2022-03-15` |
| `Amount` | float | `124.50` |
| `Category` | string | `Restaurants` |
| `Transaction Type` | string | `debit` / `credit` |
| `Account Name` | string | `Checking` |
| `Description` | string | `Whole Foods` |

> The sample file `sample_transactions.csv` included in this repo is a realistic 4-year synthetic dataset you can use to explore the app immediately.

---

## ⚙️ Sidebar Settings

| Setting | Description |
|---------|-------------|
| **Upload CSV** | Your transactions file |
| **Income categories** | Comma-separated category names treated as income (default: `Paycheck,Income,Transfer`) |
| **Forecast horizon** | Number of months to forecast ahead (3–12) |
| **Models to run** | Toggle individual models on/off |

---

## 🤖 Models & Methodology

### SARIMA (auto_arima)
Uses `pmdarima.auto_arima` to automatically select optimal `(p,d,q)(P,D,Q,s)` orders via AIC. The series is log-differenced before fitting for better stationarity. Falls back to `(1,0,1)(1,0,0,12)` if `pmdarima` is unavailable.

### Holt-Winters Exponential Smoothing
Additive trend + seasonal model with `seasonal_periods=12`. Automatically omits seasonality when fewer than 24 months of data are available.

### Prophet (tuned)
Facebook Prophet with tuned hyperparameters:
- `changepoint_prior_scale=0.3` — more flexible trend detection
- `seasonality_prior_scale=15` — stronger seasonal fit
- `seasonality_mode='multiplicative'` — better for spending data
- Custom monthly Fourier seasonality (`fourier_order=5`)

### XGBoost
Lag-based ML model using expense/income lags (1, 2, 3, 6 months), rolling means (3 & 6 month), and cyclical calendar features (sin/cos encoding of month & quarter). Regularised with L1/L2 and early stopping.

### Weighted Ensemble
Blends SARIMA + Holt-Winters + Prophet predictions using **inverse-MAPE weights** — models with lower test MAPE get higher influence in the final forecast.

### Anomaly Detection
Combines two methods:
- **Z-score** — flags months where `|z| > 2.5`
- **IQR** — flags months outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`

### Financial Health Score (0–100)
Composite score across three components:

| Component | Max Points | Condition |
|-----------|-----------|-----------|
| Savings rate | 40 | Linear scale to 30% savings rate |
| Expense stability | 30 | Based on coefficient of variation |
| Positive income | 30 | Binary: income > 0 |

---

## 📦 Dependencies

Key packages (see `requirements.txt` for pinned versions):

- `streamlit` — app framework
- `pandas`, `numpy` — data wrangling
- `plotly` — interactive charts
- `statsmodels` — SARIMA, Holt-Winters, decomposition
- `prophet` — Facebook Prophet forecasting
- `pmdarima` — auto_arima order selection
- `xgboost` — gradient boosted trees
- `scikit-learn` — IsolationForest, KMeans, StandardScaler, metrics

---

## 🛠️ Development Notes

- Heavy imports (`prophet`, `xgboost`, `statsmodels`) are wrapped in `@st.cache_resource` so they load only once per session.
- Data preprocessing is wrapped in `@st.cache_data` keyed on file bytes — re-uploading the same file skips reprocessing.
- All models include `try/except` with graceful fallback to mean predictions, so a single model failure won't crash the dashboard.

---

## 🗺️ Roadmap

- [ ] Multi-account / multi-currency support
- [ ] Export forecast to CSV / Excel
- [ ] Budget vs. actual comparison view
- [ ] LLM-powered natural language Q&A over your finances
- [ ] Plaid / bank API integration for live data

##  Acknowledgements

Built on top of the `financial_ts_analysis_improved.py` analytical pipeline. Streamlit UI, tab structure, caching strategy, and risk reporting layer added during conversion.
