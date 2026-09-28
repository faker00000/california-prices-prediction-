"""
Step 1 — train all models and save them to ./models
Run once:  python train_models.py
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- data
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------- models
def poly_pipe(model):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

MODELS = {
    "Linear Regression": Pipeline([("scaler", StandardScaler()),
                                   ("model", LinearRegression())]),
    "Polynomial Regression": poly_pipe(LinearRegression()),
    "Ridge": GridSearchCV(poly_pipe(Ridge()),
                          {"model__alpha": [0.1, 1, 10, 100]},
                          cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1),
    "Lasso": GridSearchCV(poly_pipe(Lasso(max_iter=20000)),
                          {"model__alpha": [0.0005, 0.001, 0.01, 0.1]},
                          cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1),
    "Decision Tree": GridSearchCV(DecisionTreeRegressor(random_state=RANDOM_STATE),
                                  {"max_depth": [6, 8, 10, 12],
                                   "min_samples_leaf": [5, 10, 20]},
                                  cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                           max_features=0.5, n_jobs=-1,
                                           random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                                   max_depth=5, subsample=0.8,
                                                   random_state=RANDOM_STATE),
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            n_jobs=-1, random_state=RANDOM_STATE),
    "SVR": Pipeline([("scaler", StandardScaler()),
                     ("model", SVR(kernel="rbf", C=10, epsilon=0.1))]),
}

def slug(name):
    return name.lower().replace(" ", "_")

# ---------------------------------------------------------------- train + save
metrics = []
for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    if isinstance(model, GridSearchCV):          # keep only the best pipeline
        model = model.best_estimator_

    pred_tr, pred_te = model.predict(X_train), model.predict(X_test)
    row = {
        "Model": name,
        "Train R2": r2_score(y_train, pred_tr),
        "Test R2": r2_score(y_test, pred_te),
        "Test RMSE": float(np.sqrt(mean_squared_error(y_test, pred_te))),
        "Test MAE": mean_absolute_error(y_test, pred_te),
        "Train time (s)": elapsed,
    }
    metrics.append(row)
    joblib.dump(model, MODEL_DIR / f"{slug(name)}.joblib", compress=3)
    print(f"{name:<22} Test R2={row['Test R2']:.4f}  RMSE={row['Test RMSE']:.4f}  "
          f"({elapsed:.1f}s) -> saved")

pd.DataFrame(metrics).to_csv(MODEL_DIR / "metrics.csv", index=False)

# Feature info for the app's input widgets
feature_info = {
    col: {"min": float(X[col].min()), "max": float(X[col].max()),
          "median": float(X[col].median()),
          "p01": float(X[col].quantile(0.01)), "p99": float(X[col].quantile(0.99))}
    for col in X.columns
}
json.dump({"features": list(X.columns), "info": feature_info,
           "model_files": {n: f"{slug(n)}.joblib" for n in MODELS}},
          open(MODEL_DIR / "meta.json", "w"), indent=2)

# A small sample of test data for the app's map / batch demo
X_test.assign(MedHouseVal=y_test).sample(500, random_state=RANDOM_STATE) \
      .to_csv(MODEL_DIR / "sample_test.csv", index=False)

print("\nAll models saved in", MODEL_DIR.resolve())
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
    meta = json.load(open(MODEL_DIR / "meta.json"))
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
        # Use 1st–99th percentile as slider range to avoid extreme outliers
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
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            df[f"Predicted ($) — {chosen}"] = \
                models[chosen].predict(df[FEATURES]) * 100_000
            st.success(f"Predicted {len(df)} rows with {chosen}.")
            st.dataframe(df)
            st.download_button("⬇️ Download predictions", df.to_csv(index=False),
                               "predictions.csv", "text/csv")