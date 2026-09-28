# California House Price Prediction

Trains nine regression models on the scikit-learn California Housing dataset
(1990 census) and serves them in a Streamlit app.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Train the models (downloads the dataset on first run, takes a few minutes):

   ```bash
   python train_models.py
   ```

   This writes the models, `metrics.csv`, `meta.json` and `sample_test.csv` to `models/`.

2. Start the app:

   ```bash
   streamlit run app.py
   ```

## Docker

Training runs inside the image build, so the first build takes a few minutes
and needs internet access to download the dataset.

```bash
docker compose up --build
# or without compose:
docker build -t california-prices-prediction .
docker run -p 8501:8501 california-prices-prediction
```

Then open http://localhost:8501.

## Models

Linear, Polynomial (degree 2), Ridge, Lasso, Decision Tree, Random Forest,
Gradient Boosting, XGBoost and SVR. Ridge, Lasso and Decision Tree are tuned
with 5-fold grid search. All models are scored on a 20% held-out test set.

## App

- **Predict**: enter the features of a block group and get a price from each model.
- **Compare models**: test metrics, overfitting gap, predicted vs actual, feature importance.
- **Batch prediction**: upload a CSV and download predictions.

Prices are median house values; the dataset caps them at $500,001.
