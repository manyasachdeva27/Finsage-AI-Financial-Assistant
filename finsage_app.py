"""
FinSage — Financial Time Series Analysis App
Streamlit conversion of financial_ts_analysis_improved.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import calendar
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSage · Financial Analysis",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "income":   "#2ECC71",
    "expense":  "#E74C3C",
    "savings":  "#3498DB",
    "anomaly":  "#E67E22",
    "forecast": "#9B59B6",
}

# ── Lazy imports (heavy models loaded only when needed) ───────────────────────
@st.cache_resource(show_spinner=False)
def load_heavy_imports():
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    import xgboost as xgb
    from prophet import Prophet
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    try:
        from pmdarima import auto_arima
        has_pmdarima = True
    except ImportError:
        auto_arima = None
        has_pmdarima = False

    return dict(
        adfuller=adfuller,
        seasonal_decompose=seasonal_decompose,
        SARIMAX=SARIMAX,
        ExponentialSmoothing=ExponentialSmoothing,
        mean_absolute_error=mean_absolute_error,
        mean_squared_error=mean_squared_error,
        StandardScaler=StandardScaler,
        IsolationForest=IsolationForest,
        KMeans=KMeans,
        xgb=xgb,
        Prophet=Prophet,
        go=go,
        px=px,
        make_subplots=make_subplots,
        auto_arima=auto_arima,
        has_pmdarima=has_pmdarima,
    )

# ── Data loading & preprocessing ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(file_bytes: bytes, income_cats: tuple):
    import io
    df_raw = pd.read_csv(io.BytesIO(file_bytes))
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["YearMonth"] = df["Date"].dt.to_period("M")
    df["is_income"] = df["Category"].isin(list(income_cats))

    monthly_income = (
        df[df["is_income"]].groupby("YearMonth")["Amount"].sum().rename("Income")
    )
    monthly_expense = (
        df[~df["is_income"]].groupby("YearMonth")["Amount"].sum().rename("Expenses")
    )

    monthly = pd.DataFrame({"Income": monthly_income, "Expenses": monthly_expense})
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.sort_index()
    monthly = monthly.ffill().bfill()

    monthly["Savings"] = monthly["Income"] - monthly["Expenses"]
    monthly["Savings_Ratio"] = (monthly["Savings"] / monthly["Income"]).clip(-5, 5)
    monthly["Expense_Ratio"] = (monthly["Expenses"] / monthly["Income"]).clip(0, 5)
    monthly["Net_Cashflow"] = monthly["Income"] - monthly["Expenses"]

    def cap_outliers_mad(series, window=12, k=2.5):
        rolling_med = series.rolling(window, min_periods=3, center=True).median()
        rolling_mad = (
            (series - rolling_med)
            .abs()
            .rolling(window, min_periods=3, center=True)
            .median()
        )
        lo = rolling_med - k * rolling_mad
        hi = rolling_med + k * rolling_mad
        return series.clip(lower=lo, upper=hi)

    monthly["Expenses_capped"] = cap_outliers_mad(monthly["Expenses"])

    # Lag & rolling features
    for lag in [1, 2, 3, 6]:
        monthly[f"Expense_lag_{lag}"] = monthly["Expenses"].shift(lag)
        monthly[f"Income_lag_{lag}"] = monthly["Income"].shift(lag)

    monthly["Expense_rolling3"] = monthly["Expenses"].rolling(3).mean()
    monthly["Expense_rolling6"] = monthly["Expenses"].rolling(6).mean()
    monthly["Income_rolling3"] = monthly["Income"].rolling(3).mean()

    # Cyclical features
    monthly["month"] = monthly.index.month
    monthly["quarter"] = monthly.index.quarter
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)
    monthly["quarter_sin"] = np.sin(2 * np.pi * monthly["quarter"] / 4)
    monthly["quarter_cos"] = np.cos(2 * np.pi * monthly["quarter"] / 4)

    monthly_ml = monthly.dropna()
    return df, df_raw, monthly, monthly_ml


# ── Helper functions ──────────────────────────────────────────────────────────
def evaluate(actual, predicted, model_name, libs):
    mae_fn = libs["mean_absolute_error"]
    mse_fn = libs["mean_squared_error"]
    actual = np.array(actual)
    predicted = np.array(predicted)
    mae = mae_fn(actual, predicted)
    rmse = np.sqrt(mse_fn(actual, predicted))
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return {"Model": model_name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)}


def safe_mape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def classify_risk(score):
    if score >= 80:
        return "🟢 Stable"
    elif score >= 50:
        return "🟡 Moderate Risk"
    return "🔴 High Risk"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/money-bag.png", width=64)
    st.title("FinSage ⚙️")
    st.markdown("**Financial Time Series Analyser**")
    st.divider()

    uploaded = st.file_uploader(
        "Upload transactions CSV",
        type=["csv"],
        help="Needs columns: Date, Amount, Category, Transaction Type, Account Name",
    )

    st.divider()
    st.subheader("📂 Settings")
    income_cats_input = st.text_input(
        "Income categories (comma-separated)",
        value="Paycheck,Income,Transfer",
    )
    income_cats = tuple(c.strip() for c in income_cats_input.split(","))

    forecast_months = st.slider("Forecast horizon (months)", 3, 12, 6)
    run_models = st.multiselect(
        "Models to run",
        ["SARIMA", "Holt-Winters", "Prophet", "XGBoost", "Ensemble"],
        default=["SARIMA", "Holt-Winters", "Prophet", "Ensemble"],
    )

    st.divider()
    st.caption("Built with ❤️ using Streamlit + Plotly")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown(
        """
        <div style='text-align:center; padding: 80px 0'>
            <h1>💰 FinSage</h1>
            <h4 style='color:#888'>Financial Time Series Analysis</h4>
            <p style='color:#aaa'>Upload your transactions CSV in the sidebar to get started.</p>
            <p style='color:#aaa; font-size:0.85em'>
                Expected columns: <code>Date, Amount, Category, Transaction Type, Account Name</code>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    file_bytes = uploaded.read()
    df, df_raw, monthly, monthly_ml = load_and_preprocess(file_bytes, income_cats)

with st.spinner("Importing ML libraries (first run may take ~30 s)…"):
    libs = load_heavy_imports()

go = libs["go"]
px = libs["px"]
make_subplots = libs["make_subplots"]

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_decomp, tab_models, tab_anomaly, tab_health, tab_insights, tab_clusters = st.tabs(
    ["📊 Overview", "🔍 Decomposition", "🤖 Forecasting", "⚠️ Anomalies", "❤️ Health", "💡 Insights", "🗂️ Clusters"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Monthly Financial Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Months", len(monthly))
    c2.metric(
        "Avg Monthly Income",
        f"${monthly['Income'].mean():,.0f}",
        f"{monthly['Income'].pct_change().mean()*100:+.1f}% MoM",
    )
    c3.metric(
        "Avg Monthly Expenses",
        f"${monthly['Expenses'].mean():,.0f}",
        f"{monthly['Expenses'].pct_change().mean()*100:+.1f}% MoM",
    )
    c4.metric(
        "Avg Savings Rate",
        f"{monthly['Savings_Ratio'].mean()*100:.1f}%",
    )

    # Overview chart
    fig_ov = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Monthly Income", "Monthly Expenses", "Monthly Savings"),
        vertical_spacing=0.08,
    )
    fig_ov.add_trace(go.Scatter(
        x=monthly.index, y=monthly["Income"],
        mode="lines", name="Income",
        line=dict(color=COLORS["income"], width=2),
        fill="tozeroy", fillcolor="rgba(46,204,113,0.15)",
        hovertemplate="%{x|%b %Y}<br>Income: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig_ov.add_trace(go.Scatter(
        x=monthly.index, y=monthly["Expenses"],
        mode="lines", name="Expenses",
        line=dict(color=COLORS["expense"], width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
        hovertemplate="%{x|%b %Y}<br>Expenses: $%{y:,.0f}<extra></extra>",
    ), row=2, col=1)
    sav_colors = [COLORS["income"] if s >= 0 else COLORS["expense"] for s in monthly["Savings"]]
    fig_ov.add_trace(go.Bar(
        x=monthly.index, y=monthly["Savings"],
        name="Savings", marker_color=sav_colors, opacity=0.75,
        hovertemplate="%{x|%b %Y}<br>Savings: $%{y:,.0f}<extra></extra>",
    ), row=3, col=1)
    fig_ov.update_layout(
        hovermode="x unified", template="plotly_white", height=650, showlegend=True
    )
    fig_ov.update_yaxes(title_text="Amount ($)")
    st.plotly_chart(fig_ov, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # Correlation heatmap
        corr_cols = ["Income", "Expenses", "Savings", "Savings_Ratio", "Expense_Ratio"]
        corr = monthly[corr_cols].corr().round(2)
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdYlGn", zmid=0,
            text=corr.values.round(2), texttemplate="%{text}",
            hovertemplate="%{y} × %{x}<br>Corr: %{z:.2f}<extra></extra>",
        ))
        fig_corr.update_layout(title="Correlation Heatmap", template="plotly_white", height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_r:
        # Top 10 spending categories
        exp_cats = (
            df[~df["is_income"]]
            .groupby("Category")["Amount"].sum()
            .sort_values(ascending=False)
            .head(10)
        )
        fig_cats = go.Figure(go.Bar(
            x=exp_cats.values[::-1], y=exp_cats.index[::-1],
            orientation="h",
            marker=dict(color=exp_cats.values[::-1], colorscale="RdYlGn_r", showscale=False),
            hovertemplate="<b>%{y}</b><br>Total Spend: $%{x:,.0f}<extra></extra>",
        ))
        fig_cats.update_layout(
            title="Top 10 Spending Categories",
            xaxis_title="Total Spending ($)",
            template="plotly_white", height=400, hovermode="y unified",
        )
        st.plotly_chart(fig_cats, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
with tab_decomp:
    st.header("Time Series Decomposition & Stationarity")
    target = monthly["Expenses"].dropna()

    if len(target) >= 24:
        decomp = libs["seasonal_decompose"](target, model="additive", period=12)
        fig_dec = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=("Observed", "Trend", "Seasonality", "Residual"),
            vertical_spacing=0.07,
        )
        fig_dec.add_trace(go.Scatter(
            x=decomp.observed.index, y=decomp.observed.values,
            mode="lines", name="Observed", line=dict(color=COLORS["expense"]),
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        ), row=1, col=1)
        fig_dec.add_trace(go.Scatter(
            x=decomp.trend.index, y=decomp.trend.values,
            mode="lines", name="Trend", line=dict(color="#2C3E50"),
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        ), row=2, col=1)
        fig_dec.add_trace(go.Scatter(
            x=decomp.seasonal.index, y=decomp.seasonal.values,
            mode="lines", name="Seasonality", line=dict(color=COLORS["savings"]),
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        ), row=3, col=1)
        fig_dec.add_trace(go.Scatter(
            x=decomp.resid.index, y=decomp.resid.values,
            mode="markers", name="Residual",
            marker=dict(color=COLORS["anomaly"], size=5),
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
        ), row=4, col=1)
        fig_dec.update_layout(
            title="Seasonal Decomposition of Monthly Expenses",
            hovermode="x unified", template="plotly_white", height=750,
        )
        st.plotly_chart(fig_dec, use_container_width=True)
    else:
        st.warning(f"Only {len(target)} months available — need ≥ 24 for decomposition.")

    # Rolling stats
    win = min(6, len(target) // 3)
    fig_roll = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(f"Rolling Mean ({win}-month)", "Rolling Std Dev"),
        vertical_spacing=0.1,
    )
    fig_roll.add_trace(go.Scatter(
        x=target.index, y=target.values,
        mode="lines", name="Actual", line=dict(color=COLORS["expense"], width=1.5), opacity=0.6,
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig_roll.add_trace(go.Scatter(
        x=target.index, y=target.rolling(win).mean().values,
        mode="lines", name=f"{win}-Month Mean", line=dict(color="#2C3E50", width=2),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig_roll.add_trace(go.Scatter(
        x=target.index, y=target.rolling(win).std().values,
        mode="lines", name="Std Dev", line=dict(color=COLORS["anomaly"], width=2),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
    ), row=2, col=1)
    fig_roll.update_layout(
        title="Rolling Statistics", hovermode="x unified", template="plotly_white", height=500,
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    # ADF test
    adf = libs["adfuller"](target.dropna())
    st.subheader("ADF Stationarity Test")
    col1, col2, col3 = st.columns(3)
    col1.metric("ADF Statistic", f"{adf[0]:.4f}")
    col2.metric("p-value", f"{adf[1]:.4f}")
    col3.metric("Critical (5%)", f"{adf[4]['5%']:.4f}")
    if adf[1] < 0.05:
        st.success("✅ Series is **STATIONARY** (p < 0.05)")
    else:
        st.warning("⚠️ Series is **NON-STATIONARY** — differencing may be needed")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.header("Expense Forecasting Models")
    target = monthly["Expenses"].dropna()
    TEST_SIZE = min(6, len(target) // 5)
    train = target.iloc[:-TEST_SIZE]
    test = target.iloc[-TEST_SIZE:]

    st.info(
        f"**Train:** {len(train)} months | **Test:** {len(test)} months | "
        f"**Forecast:** {forecast_months} months ahead"
    )

    future_dates = pd.date_range(
        start=target.index[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_months, freq="MS",
    )

    results_table = []
    sarima_pred = sarima_fore = hw_pred = hw_fore_arr = None
    prophet_pred = prophet_fore_df = ensemble_pred = ensemble_fore = None

    # ── SARIMA ────────────────────────────────────────────────────────────────
    if "SARIMA" in run_models:
        with st.spinner("Fitting SARIMA…"):
            target_log = np.log1p(target)
            target_diff = target_log.diff().dropna()
            train_t = target_diff.iloc[:-TEST_SIZE]

            try:
                if libs["has_pmdarima"]:
                    auto = libs["auto_arima"](
                        train_t, seasonal=True, m=12, d=0, D=0,
                        max_p=3, max_q=3, max_P=2, max_Q=2,
                        information_criterion="aic", stepwise=True,
                        suppress_warnings=True, error_action="ignore",
                    )
                    order, s_order = auto.order, auto.seasonal_order
                else:
                    order, s_order = (1, 0, 1), (1, 0, 0, 12)

                sarima_fit = libs["SARIMAX"](
                    train_t, order=order, seasonal_order=s_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)

                sarima_pred_t = sarima_fit.forecast(TEST_SIZE)
                sarima_fore_t = sarima_fit.forecast(TEST_SIZE + forecast_months)

                last_log = target_log.iloc[len(train_t)]
                pred_log = last_log + sarima_pred_t.cumsum()
                fore_log = target_log.iloc[-1] + sarima_fore_t[-forecast_months:].cumsum()

                sarima_pred = np.expm1(pred_log)
                sarima_fore = pd.Series(np.expm1(fore_log.values), index=future_dates)
                sarima_pred.index = test.index

                results_table.append(evaluate(test, sarima_pred, "SARIMA", libs))
                st.success(f"✅ SARIMA fitted (order={order}, seasonal={s_order})")
            except Exception as e:
                st.error(f"SARIMA failed: {e}")
                sarima_pred = pd.Series([train.mean()] * TEST_SIZE, index=test.index)
                sarima_fore = pd.Series([train.mean()] * forecast_months, index=future_dates)

    # ── Holt-Winters ──────────────────────────────────────────────────────────
    if "Holt-Winters" in run_models:
        with st.spinner("Fitting Holt-Winters…"):
            try:
                sp = 12 if len(train) >= 24 else None
                hw_fit = libs["ExponentialSmoothing"](
                    train, trend="add",
                    seasonal="add" if sp else None,
                    seasonal_periods=sp,
                ).fit(optimized=True)
                hw_pred = hw_fit.forecast(TEST_SIZE)
                hw_fore_arr = np.array(hw_fit.forecast(TEST_SIZE + forecast_months))[-forecast_months:]
                results_table.append(evaluate(test, hw_pred, "Holt-Winters", libs))
                st.success("✅ Holt-Winters fitted")
            except Exception as e:
                st.error(f"Holt-Winters failed: {e}")
                hw_pred = pd.Series([train.mean()] * TEST_SIZE, index=test.index)
                hw_fore_arr = np.array([train.mean()] * forecast_months)

    # ── Prophet ───────────────────────────────────────────────────────────────
    if "Prophet" in run_models:
        with st.spinner("Fitting Prophet…"):
            try:
                prophet_df_fit = pd.DataFrame({"ds": train.index, "y": train.values})
                m = libs["Prophet"](
                    yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.3,
                    seasonality_prior_scale=15,
                    seasonality_mode="multiplicative",
                )
                m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
                m.fit(prophet_df_fit)

                future = m.make_future_dataframe(periods=TEST_SIZE + forecast_months, freq="MS")
                forecast_all = m.predict(future)

                prophet_pred = forecast_all[forecast_all["ds"].isin(test.index)]["yhat"].values
                prophet_fore_df = forecast_all.tail(forecast_months)

                if len(prophet_pred) != len(test):
                    prophet_pred = prophet_pred[: len(test)]

                results_table.append(evaluate(test, prophet_pred, "Prophet", libs))
                st.success("✅ Prophet fitted")
            except Exception as e:
                st.error(f"Prophet failed: {e}")
                prophet_pred = np.array([train.mean()] * TEST_SIZE)
                prophet_fore_df = pd.DataFrame({"ds": future_dates, "yhat": [train.mean()] * forecast_months})

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if "XGBoost" in run_models:
        with st.spinner("Fitting XGBoost…"):
            try:
                feat_cols = (
                    [c for c in monthly_ml.columns if "lag" in c or "rolling" in c]
                    + ["month_sin", "month_cos", "quarter_sin", "quarter_cos"]
                )
                feat_cols = [c for c in feat_cols if c in monthly_ml.columns]
                X = monthly_ml[feat_cols]
                y_xgb = monthly_ml["Expenses"]
                split = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y_xgb.iloc[:split], y_xgb.iloc[split:]

                xgb_model = libs["xgb"].XGBRegressor(
                    n_estimators=500, learning_rate=0.03, max_depth=3,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, verbosity=0,
                    early_stopping_rounds=30,
                )
                xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                xgb_pred = xgb_model.predict(X_test)
                results_table.append(evaluate(y_test, xgb_pred, "XGBoost", libs))
                st.success("✅ XGBoost fitted")

                fi = pd.Series(xgb_model.feature_importances_, index=feat_cols).sort_values(ascending=True)
                fig_fi = go.Figure(go.Bar(
                    x=fi.values, y=fi.index.tolist(), orientation="h",
                    marker_color=COLORS["forecast"],
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
                ))
                fig_fi.update_layout(
                    title="XGBoost Feature Importance",
                    xaxis_title="Importance Score",
                    template="plotly_white", height=400, hovermode="y unified",
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            except Exception as e:
                st.error(f"XGBoost failed: {e}")

    # ── Ensemble ─────────────────────────────────────────────────────────────
    if "Ensemble" in run_models and all(
        v is not None for v in [sarima_pred, hw_pred, prophet_pred]
    ):
        mape_s = safe_mape(test, sarima_pred.values[: len(test)])
        mape_h = safe_mape(test, hw_pred.values[: len(test)])
        mape_p = safe_mape(test, prophet_pred[: len(test)])
        inv = np.array([1 / (mape_s + 1e-9), 1 / (mape_h + 1e-9), 1 / (mape_p + 1e-9)])
        weights = inv / inv.sum()

        ensemble_pred = (
            weights[0] * sarima_pred.values[: len(test)]
            + weights[1] * hw_pred.values[: len(test)]
            + weights[2] * prophet_pred[: len(test)]
        )
        results_table.append(evaluate(test, ensemble_pred, "Ensemble", libs))

        sarima_fore_arr = np.array(sarima_fore.values)[:forecast_months]
        prophet_fore_arr = np.array(prophet_fore_df["yhat"].values)[:forecast_months]
        ensemble_fore = (
            weights[0] * sarima_fore_arr
            + weights[1] * hw_fore_arr[:forecast_months]
            + weights[2] * prophet_fore_arr
        )
        st.success(
            f"✅ Ensemble (SARIMA: {weights[0]:.2f} | HW: {weights[1]:.2f} | Prophet: {weights[2]:.2f})"
        )

    # ── Model comparison table ────────────────────────────────────────────────
    if results_table:
        st.subheader("📋 Model Comparison")
        comp_df = pd.DataFrame(results_table).set_index("Model").sort_values("MAPE (%)")
        comp_df["Rank"] = range(1, len(comp_df) + 1)
        st.dataframe(comp_df.style.highlight_min(subset=["MAPE (%)"], color="#d4edda"), use_container_width=True)
        best = comp_df.index[0]
        st.success(f"🏆 Best model by MAPE: **{best}** ({comp_df.loc[best,'MAPE (%)']:.2f}%)")

    # ── Forecast chart ────────────────────────────────────────────────────────
    st.subheader("📈 Forecast Visualisation")
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=train.index, y=train.values, mode="lines", name="Train", line=dict(color="gray")))
    fig_fc.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines+markers", name="Actual", line=dict(color="black", dash="dash")))

    if sarima_pred is not None:
        fig_fc.add_trace(go.Scatter(x=test.index, y=sarima_pred.values, mode="lines+markers", name="SARIMA"))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=sarima_fore.values, mode="lines+markers", name="SARIMA Fcast", line=dict(dash="dot")))
    if hw_pred is not None:
        fig_fc.add_trace(go.Scatter(x=test.index, y=hw_pred.values, mode="lines+markers", name="Holt-Winters"))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=hw_fore_arr, mode="lines+markers", name="HW Fcast", line=dict(dash="dot")))
    if prophet_pred is not None:
        fig_fc.add_trace(go.Scatter(x=test.index, y=prophet_pred, mode="lines+markers", name="Prophet"))
        fig_fc.add_trace(go.Scatter(x=prophet_fore_df["ds"], y=prophet_fore_df["yhat"], mode="lines+markers", name="Prophet Fcast", line=dict(dash="dot")))
    if ensemble_pred is not None:
        fig_fc.add_trace(go.Scatter(x=test.index, y=ensemble_pred, mode="lines+markers", name="Ensemble", line=dict(width=3)))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=ensemble_fore, mode="lines+markers", name="Ensemble Fcast", line=dict(width=3, dash="dot")))

    fig_fc.update_layout(
        title="Expense Forecast Comparison",
        xaxis_title="Date", yaxis_title="Expenses ($)",
        hovermode="x unified", template="plotly_white", height=500,
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── 6-month risk report ────────────────────────────────────────────────────
    if prophet_fore_df is not None:
        st.subheader("🚨 Forward Risk Report")
        avg_income = monthly["Income"].mean()
        risk_rows = []
        for _, row in prophet_fore_df.iterrows():
            exp = row["yhat"]
            sr = (avg_income - exp) / avg_income
            if exp > avg_income:
                status = "🔴 HIGH: Expenses exceed income"
            elif exp > 0.85 * avg_income:
                status = "🟡 MEDIUM: Expense ratio > 85%"
            elif sr < 0.10:
                status = "🟠 LOW SAVINGS: < 10% savings rate"
            else:
                status = "🟢 Healthy"
            risk_rows.append({"Month": pd.to_datetime(row["ds"]).strftime("%b %Y"), "Forecast Expense": f"${exp:,.0f}", "Status": status})
        st.dataframe(pd.DataFrame(risk_rows).set_index("Month"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.header("⚠️ Anomaly Detection")

    monthly2 = monthly.copy()
    monthly2["z_score"] = (monthly2["Expenses"] - monthly2["Expenses"].mean()) / monthly2["Expenses"].std()
    monthly2["anomaly_zscore"] = monthly2["z_score"].abs() > 2.5

    Q1, Q3 = monthly2["Expenses"].quantile(0.25), monthly2["Expenses"].quantile(0.75)
    IQR = Q3 - Q1
    monthly2["anomaly_iqr"] = (monthly2["Expenses"] < Q1 - 1.5 * IQR) | (monthly2["Expenses"] > Q3 + 1.5 * IQR)

    iso = libs["IsolationForest"](contamination=0.1, random_state=42)
    monthly2["anomaly_if"] = iso.fit_predict(monthly2[["Expenses"]].fillna(0)) == -1
    monthly2["anomaly"] = monthly2["anomaly_zscore"] | monthly2["anomaly_iqr"]

    anomaly_df = monthly2[monthly2["anomaly"]]
    st.metric("Anomalies Detected", len(anomaly_df))

    fig_an = go.Figure()
    fig_an.add_trace(go.Scatter(
        x=monthly2.index, y=monthly2["Expenses"],
        mode="lines+markers", name="Expenses",
        line=dict(color=COLORS["expense"]),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig_an.add_trace(go.Scatter(
        x=anomaly_df.index, y=anomaly_df["Expenses"],
        mode="markers", name="Anomaly",
        marker=dict(color=COLORS["anomaly"], size=13, line=dict(color="black", width=1)),
        hovertemplate="%{x|%b %Y}<br>⚠️ $%{y:,.0f}<extra></extra>",
    ))
    fig_an.update_layout(
        title="Expense Anomaly Detection (Z-Score + IQR)",
        xaxis_title="Date", yaxis_title="Expenses ($)",
        hovermode="x unified", template="plotly_white", height=450,
    )
    st.plotly_chart(fig_an, use_container_width=True)

    if len(anomaly_df):
        st.subheader("Anomalous Months")
        st.dataframe(
            anomaly_df[["Income", "Expenses", "Savings", "z_score"]].round(2),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HEALTH SCORE
# ══════════════════════════════════════════════════════════════════════════════
with tab_health:
    st.header("❤️ Financial Health Score")

    monthly3 = monthly.copy()

    def compute_health_score(row):
        sr = row.get("Savings_Ratio", 0)
        sr_score = np.clip(sr / 0.30, 0, 1) * 40
        cv = monthly3["Expenses"].std() / (monthly3["Expenses"].mean() + 1e-9)
        stability_score = np.clip((1 - cv), 0, 1) * 30
        income_score = 30 if row.get("Income", 0) > 0 else 0
        return round(sr_score + stability_score + income_score, 1)

    monthly3["Health_Score"] = monthly3.apply(compute_health_score, axis=1)
    monthly3["Risk_Label"] = monthly3["Health_Score"].apply(classify_risk)

    latest_score = monthly3["Health_Score"].iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Health Score", f"{latest_score} / 100")
    col2.metric("Risk Status", classify_risk(latest_score))
    col3.metric("Avg Health Score", f"{monthly3['Health_Score'].mean():.1f}")

    fig_h = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Financial Health Score Over Time", "Savings Rate (%)"),
        vertical_spacing=0.1,
    )
    hs_colors = ["#2ECC71" if s >= 80 else "#F39C12" if s >= 50 else "#E74C3C"
                 for s in monthly3["Health_Score"]]
    fig_h.add_trace(go.Bar(
        x=monthly3.index, y=monthly3["Health_Score"],
        marker_color=hs_colors, name="Health Score",
        hovertemplate="%{x|%b %Y}<br>Score: %{y:.1f}<extra></extra>",
    ), row=1, col=1)
    fig_h.add_hline(y=80, line=dict(color="#2ECC71", dash="dash"), annotation_text="Stable (80)", row=1, col=1)
    fig_h.add_hline(y=50, line=dict(color="#F39C12", dash="dash"), annotation_text="Moderate (50)", row=1, col=1)
    fig_h.add_trace(go.Scatter(
        x=monthly3.index, y=monthly3["Savings_Ratio"] * 100,
        mode="lines", name="Savings Rate (%)",
        line=dict(color=COLORS["savings"], width=2),
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra></extra>",
    ), row=2, col=1)
    fig_h.add_hline(y=0, line=dict(color="red", dash="dash", width=0.8), row=2, col=1)
    fig_h.update_yaxes(title_text="Score (0–100)", range=[0, 105], row=1, col=1)
    fig_h.update_yaxes(title_text="Savings Rate (%)", row=2, col=1)
    fig_h.update_layout(
        title="Financial Health Dashboard",
        hovermode="x unified", template="plotly_white", height=580,
    )
    st.plotly_chart(fig_h, use_container_width=True)

    st.subheader("Last 12 Months Detail")
    st.dataframe(
        monthly3[["Income", "Expenses", "Savings_Ratio", "Health_Score", "Risk_Label"]].tail(12).round(2),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.header("💡 Auto-Generated Financial Insights")

    monthly4 = monthly.copy()
    monthly4["z_score"] = (monthly4["Expenses"] - monthly4["Expenses"].mean()) / monthly4["Expenses"].std()
    monthly4["anomaly"] = monthly4["z_score"].abs() > 2.5
    anomaly_df4 = monthly4[monthly4["anomaly"]]

    def compute_health_score4(row):
        sr = row.get("Savings_Ratio", 0)
        sr_score = np.clip(sr / 0.30, 0, 1) * 40
        cv = monthly4["Expenses"].std() / (monthly4["Expenses"].mean() + 1e-9)
        stability_score = np.clip((1 - cv), 0, 1) * 30
        income_score = 30 if row.get("Income", 0) > 0 else 0
        return round(sr_score + stability_score + income_score, 1)

    monthly4["Health_Score"] = monthly4.apply(compute_health_score4, axis=1)

    insights = []
    recent_3 = monthly4["Expenses"].tail(3).mean()
    previous_3 = monthly4["Expenses"].iloc[-6:-3].mean()
    pct_change = (recent_3 - previous_3) / (previous_3 + 1e-9) * 100
    direction = "increased" if pct_change > 0 else "decreased"
    insights.append(
        ("📈 Spending Trend",
         f"Expenses have **{direction} by {abs(pct_change):.1f}%** over the last 3 months vs. the prior quarter.")
    )

    monthly4["Month_num"] = monthly4.index.month
    monthly_avg = monthly4.groupby("Month_num")["Expenses"].mean()
    peak_month = monthly_avg.idxmax()
    insights.append(
        ("📅 Seasonality",
         f"Highest average spending occurs in **{calendar.month_name[peak_month]}**, suggesting seasonal pressures (holidays, utility spikes, etc.).")
    )

    low_sav = monthly4[monthly4["Savings_Ratio"] < 0.10]
    if len(low_sav) > 0:
        pct_low = len(low_sav) / len(monthly4) * 100
        insights.append(
            ("💰 Savings Risk",
             f"**{pct_low:.1f}%** of months had a savings rate below 10%. Consider automating savings transfers.")
        )

    income_cv = monthly4["Income"].std() / (monthly4["Income"].mean() + 1e-9)
    if income_cv > 0.3:
        insights.append(
            ("⚡ Income Volatility",
             f"Income coefficient of variation = **{income_cv:.2f}**. Irregular income detected — maintain a 3-month emergency fund.")
        )

    if len(anomaly_df4) > 0:
        max_month = anomaly_df4["Expenses"].idxmax()
        insights.append(
            ("🚨 Anomaly Alert",
             f"Highest abnormal spending spike: **{max_month.strftime('%b %Y')}** "
             f"(${anomaly_df4.loc[max_month, 'Expenses']:,.0f}). Review for one-time purchases or billing errors.")
        )

    latest_score = monthly4["Health_Score"].iloc[-1]
    insights.append(
        ("❤️ Current Health",
         f"Latest Financial Health Score = **{latest_score}/100** ({classify_risk(latest_score)}). "
         + ("Consider reducing discretionary spend." if latest_score < 70 else "Keep maintaining these habits!"))
    )

    for title, body in insights:
        with st.expander(title, expanded=True):
            st.markdown(body)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_clusters:
    st.header("🗂️ Category-Level Spending Clusters")

    cat_pivot = (
        df[~df["is_income"]]
        .assign(YM=lambda x: x["Date"].dt.to_period("M"))
        .groupby(["YM", "Category"])["Amount"].sum()
        .unstack(fill_value=0)
    )

    scaler = libs["StandardScaler"]()
    cat_scaled = scaler.fit_transform(cat_pivot.fillna(0))

    best_k = st.slider("Number of clusters (k)", 2, min(6, len(cat_pivot)), 3)
    km = libs["KMeans"](n_clusters=best_k, random_state=42, n_init=10)
    cat_pivot["Cluster"] = km.fit_predict(cat_scaled)

    cluster_profiles = cat_pivot.groupby("Cluster").mean().T
    cluster_colors = px.colors.qualitative.Set2

    fig_cl = go.Figure()
    for i, col in enumerate(cluster_profiles.columns):
        fig_cl.add_trace(go.Bar(
            name=f"Cluster {col}",
            x=cluster_profiles.index.tolist(),
            y=cluster_profiles[col].values,
            marker_color=cluster_colors[i % len(cluster_colors)],
            hovertemplate="<b>%{x}</b><br>Cluster " + str(col) + ": $%{y:,.1f}<extra></extra>",
        ))
    fig_cl.update_layout(
        title="Spending Cluster Profiles",
        xaxis_title="Category", yaxis_title="Avg Monthly Spend ($)",
        barmode="group", template="plotly_white", height=500,
        hovermode="x unified", xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_cl, use_container_width=True)

    st.subheader("Cluster Profiles (Mean $ / Category)")
    st.dataframe(cluster_profiles.round(1), use_container_width=True)
