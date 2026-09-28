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
        elapsed = model.refit_time_              # time of the final fit, not the whole search
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
with open(MODEL_DIR / "meta.json", "w") as f:
    json.dump({"features": list(X.columns), "info": feature_info,
               "model_files": {n: f"{slug(n)}.joblib" for n in MODELS}},
              f, indent=2)

# A small sample of test data for the app's map / batch demo
X_test.assign(MedHouseVal=y_test).sample(500, random_state=RANDOM_STATE) \
      .to_csv(MODEL_DIR / "sample_test.csv", index=False)

print("\nAll models saved in", MODEL_DIR.resolve())
