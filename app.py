from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

streamlit==1.32.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
statsmodels==0.14.1
plotly==5.20.0
joblib==1.3.2

st.set_page_config(page_title="AI Demand Forecasting", layout="wide")
st.title("AI Demand Forecasting Workbench")
st.caption("Ensemble forecasting prototype for supply chain planning and inventory optimization")

if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_dataset().to_csv(DATA_PATH, index=False)

forecaster = DemandForecaster(str(MODEL_DIR))

def ensure_trained(df: pd.DataFrame) -> None:
    if not forecaster.artifacts_path.exists():
        with st.spinner("Training forecasting models for the first time..."):
            forecaster.fit(df)


df = load_data(str(DATA_PATH))
ensure_trained(df)

with st.sidebar:
    st.header("Controls")
    sku = st.selectbox("SKU", sorted(df["sku"].unique().tolist()))
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=1)
    if st.button("Retrain model"):
        with st.spinner("Retraining..."):
            forecaster.fit(df)
        st.success("Model retrained")

result = forecaster.forecast(df, sku=sku, horizon=horizon)
forecast_df = pd.DataFrame(result["forecast"])
forecast_df["date"] = pd.to_datetime(forecast_df["date"])

sku_hist = df[df["sku"] == sku].sort_values("date").tail(180)

c1, c2, c3 = st.columns(3)
c1.metric("Approx. Forecast Accuracy", f"{result['metrics']['accuracy']}%")
c2.metric("Validation MAPE", f"{result['metrics']['mape']}%")
c3.metric("Validation RMSE", f"{result['metrics']['rmse']}")

chart = go.Figure()
chart.add_trace(go.Scatter(x=sku_hist["date"], y=sku_hist["sales"], name="Historical Sales"))
chart.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["forecast"], name="Forecast"))
chart.add_trace(
    go.Scatter(
        x=forecast_df["date"].tolist() + forecast_df["date"].tolist()[::-1],
        y=forecast_df["upper_bound"].tolist() + forecast_df["lower_bound"].tolist()[::-1],
        fill="toself",
        name="Confidence Band",
        line=dict(width=0),
        opacity=0.2,
    )
)
chart.update_layout(height=420, xaxis_title="Date", yaxis_title="Units")
st.plotly_chart(chart, use_container_width=True)

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Demand trend heatmap")
    heat = sku_hist.copy()
    heat["week"] = heat["date"].dt.isocalendar().week.astype(int)
    heat["dow"] = heat["date"].dt.day_name().str[:3]
    pivot = heat.pivot_table(index="week", columns="dow", values="sales", aggfunc="mean")
    st.plotly_chart(px.imshow(pivot, aspect="auto", labels=dict(color="Avg Sales")), use_container_width=True)

with right:
    st.subheader("Key demand drivers")
    drivers_df = pd.DataFrame(result["drivers"]).head(10)
    st.dataframe(drivers_df, use_container_width=True, hide_index=True)

b1, b2 = st.columns(2)
with b1:
    st.subheader("Anomalies / shocks")
    anomalies_df = pd.DataFrame(result["anomalies"])
    if anomalies_df.empty:
        st.info("No major anomalies detected in the recent window.")
    else:
        st.dataframe(anomalies_df, use_container_width=True, hide_index=True)

with b2:
    st.subheader("Inventory recommendations")
    rec = result["recommendations"]
    st.write(f"**Average daily forecast:** {rec['avg_daily_forecast']}")
    st.write(f"**Safety stock:** {rec['safety_stock']}")
    st.write(f"**Reorder point:** {rec['reorder_point']}")
    st.write(f"**Recommended action:** {rec['inventory_action']}")

st.subheader("Forecast output")
st.dataframe(forecast_df, use_container_width=True, hide_index=True)
