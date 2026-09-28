"""
Step 2 — run the app:  streamlit run app.py
(train first with:  python train_models.py)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "models"

FEATURE_HELP = {
    "MedInc": "Median income in block group (in $10,000s)",
    "HouseAge": "Median house age in block group (years)",
    "AveRooms": "Average number of rooms per household",
    "AveBedrms": "Average number of bedrooms per household",
    "Population": "Block group population",
    "AveOccup": "Average number of household members",
    "Latitude": "Block group latitude",
    "Longitude": "Block group longitude",
}

st.set_page_config(page_title="California House Price Predictor",
                   page_icon="🏠", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    with open(MODEL_DIR / "meta.json") as f:
        meta = json.load(f)
    models = {name: joblib.load(MODEL_DIR / fname)
              for name, fname in meta["model_files"].items()}
    return meta, models


@st.cache_data
def load_metrics():
    return pd.read_csv(MODEL_DIR / "metrics.csv") \
             .sort_values("Test R2", ascending=False).reset_index(drop=True)


@st.cache_data
def load_sample():
    return pd.read_csv(MODEL_DIR / "sample_test.csv")


if not (MODEL_DIR / "meta.json").exists():
    st.error("No trained models found. Run `python train_models.py` first.")
    st.stop()

meta, models = load_models()
metrics = load_metrics()
FEATURES = meta["features"]
INFO = meta["info"]
best_model_name = metrics.loc[0, "Model"]


def fmt_price(v):                       # target is in $100,000s
    return f"${v * 100_000:,.0f}"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")
model_names = list(models.keys())
chosen = st.sidebar.selectbox(
    "Model", model_names, index=model_names.index(best_model_name),
    help="Default is the model with the best test R².")

row = metrics[metrics["Model"] == chosen].iloc[0]
st.sidebar.metric("Test R²", f"{row['Test R2']:.3f}")
st.sidebar.metric("Test RMSE", fmt_price(row["Test RMSE"]))
st.sidebar.metric("Test MAE", fmt_price(row["Test MAE"]))
st.sidebar.caption("RMSE/MAE shown in dollars (target unit = $100k).")

st.title("🏠 California House Price Predictor")
st.caption("Nine regression models trained on the California Housing dataset (1990 census).")

tab_predict, tab_compare, tab_batch = st.tabs(
    ["🔮 Predict", "📊 Compare models", "📁 Batch prediction"])

# ---------------------------------------------------------------- tab 1: single prediction
with tab_predict:
    st.subheader("Describe the block group")
    inputs = {}
    cols = st.columns(4)
    for i, feat in enumerate(FEATURES):
        info = INFO[feat]
        # Default = median, clipped to the 1st–99th percentile to avoid extreme outliers
        lo, hi = info["p01"], info["p99"]
        step = 1.0 if feat in ("HouseAge", "Population") else 0.01
        with cols[i % 4]:
            inputs[feat] = st.number_input(
                feat, min_value=float(info["min"]), max_value=float(info["max"]),
                value=float(np.clip(info["median"], lo, hi)), step=step,
                help=FEATURE_HELP.get(feat, ""))

    X_input = pd.DataFrame([inputs])[FEATURES]
    pred = float(models[chosen].predict(X_input)[0])

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(f"Predicted median value ({chosen})", fmt_price(pred))
        if pred >= 5.0:
            st.info("Note: the dataset caps values at $500,001, so predictions "
                    "near this level may be underestimates.")
        st.map(pd.DataFrame({"lat": [inputs["Latitude"]],
                             "lon": [inputs["Longitude"]]}), zoom=5)
    with c2:
        st.markdown("**What every model predicts for these inputs**")
        all_preds = pd.DataFrame({
            "Model": model_names,
            "Prediction ($)": [float(m.predict(X_input)[0]) * 100_000
                               for m in models.values()],
        }).sort_values("Prediction ($)")
        st.bar_chart(all_preds.set_index("Model"), horizontal=True)
        spread = all_preds["Prediction ($)"]
        st.caption(f"Range across models: ${spread.min():,.0f} – ${spread.max():,.0f}")

# ---------------------------------------------------------------- tab 2: comparison
with tab_compare:
    st.subheader("Performance on the held-out test set")
    st.dataframe(
        metrics.style.format({
            "Train R2": "{:.4f}", "Test R2": "{:.4f}", "Test RMSE": "{:.4f}",
            "Test MAE": "{:.4f}", "Train time (s)": "{:.1f}"})
        .highlight_max(subset=["Test R2"], color="#c6efce")
        .highlight_min(subset=["Test RMSE", "Test MAE"], color="#c6efce"),
        hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Test R²** (higher is better)")
        st.bar_chart(metrics.set_index("Model")["Test R2"], horizontal=True)
    with c2:
        st.markdown("**Overfitting check: Train R² − Test R²**")
        gap = (metrics["Train R2"] - metrics["Test R2"]).rename("Gap")
        st.bar_chart(pd.concat([metrics["Model"], gap], axis=1).set_index("Model"),
                     horizontal=True)

    st.markdown(f"**Predicted vs actual — {chosen}** (500 test samples)")
    sample = load_sample()
    sample_pred = models[chosen].predict(sample[FEATURES])
    st.scatter_chart(pd.DataFrame({"Actual": sample["MedHouseVal"],
                                   "Predicted": sample_pred}),
                     x="Actual", y="Predicted")

    model_obj = models[chosen]
    if hasattr(model_obj, "feature_importances_"):
        st.markdown(f"**Feature importance — {chosen}**")
        st.bar_chart(pd.Series(model_obj.feature_importances_, index=FEATURES)
                     .sort_values(), horizontal=True)

# ---------------------------------------------------------------- tab 3: batch
with tab_batch:
    st.subheader("Upload a CSV to predict many rows")
    st.write("Required columns:", ", ".join(f"`{f}`" for f in FEATURES))

    template = load_sample()[FEATURES].head(5)
    st.download_button("⬇️ Download template CSV", template.to_csv(index=False),
                       "template.csv", "text/csv")

    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [f for f in FEATURES if f not in df.columns]
        X_batch = df[FEATURES].apply(pd.to_numeric, errors="coerce") if not missing else None
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        elif X_batch.isna().any().any():
            bad = X_batch.index[X_batch.isna().any(axis=1)] + 2   # +2: header row, 1-based
            st.error("Empty or non-numeric values in rows: "
                     + ", ".join(map(str, bad[:20])) + (" …" if len(bad) > 20 else ""))
        else:
            df[f"Predicted ($) — {chosen}"] = \
                models[chosen].predict(X_batch) * 100_000
            st.success(f"Predicted {len(df)} rows with {chosen}.")
            st.dataframe(df)
            st.download_button("⬇️ Download predictions", df.to_csv(index=False),
                               "predictions.csv", "text/csv")
