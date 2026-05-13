<!-- car-price-predictor/README.md -->

# car-price-predictor

End-to-end vehicle listing price estimation: ingestion with synthetic fallback, leakage-aware target encoding, multi-model training with `GridSearchCV` for tree ensembles, evaluation plots, and a small Flask service with HTML and JSON interfaces.

## Overview

- **Data**: Supports three layouts automatically:
  - **Cardekho-style** (`car_data.csv`): columns such as `Car Name`, `Year`, `Selling Price`, `Kms Driven`, `Fuel_Type`, `Transmission`, `Owner`.
  - **Auto Scout export** (`auto_scout_car.csv`): `make_model`, `price`, `km`, `Fuel`, `Gearing_Type`, `Previous_Owners`, `age` (registration **year** is derived as `reference_year ? age` from `config/config.yaml`, defaulting to the current calendar year when `reference_year` is null).
  - **Kleinanzeigen-style** (`autos.csv`): `brand`, `yearOfRegistration`, `price`, `kilometer`, `fuelType`, `gearbox` (owner is unknown; rows get a neutral **Second Owner** label).
  Place a CSV under `data/raw/` and set `paths.raw_csv` in `config/config.yaml`, **or** point to any absolute path with **`CAR_PRICE_RAW_CSV`** (for example your iCloud copies). If no readable file is found, `src/data_loader.py` generates **10,000** deterministic synthetic rows.
- **Currency**: The API reports `"currency": "EUR"` for consistent labeling. **Align your training CSV** (or synthetic generator) to the currency you care about so predictions are interpretable.
- **Artifacts**: Training writes `models/feature_engineering.joblib`, `models/preprocessor.joblib`, and `models/model.joblib` (best model bundle). Figures land in `reports/figures/`.

## Requirements

- Python **3.11+** recommended (matches `Dockerfile`).
- See `requirements.txt`: `flask`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `pyyaml`, `gunicorn`.

### Kaggle dataset setup

1. Download `car_data.csv` from the [Cardekho dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).
2. Copy it to `data/raw/car_data.csv`.
3. Column mapping is handled automatically: `Car Name` / `name` ? derived **brand** (first token), `Year`, `Selling Price`, `Kms Driven`, `Fuel_Type`, `Transmission`, `Owner`, etc.

### Auto Scout or Kleinanzeigen CSV (local paths)

Example (quotes matter because of spaces in `Mobile Documents`):

```bash
export CAR_PRICE_RAW_CSV="/Users/you/Library/Mobile Documents/com~apple~CloudDocs/auto_scout_car.csv"
python -m src.train
```

Use the same variable with `autos.csv` when training on the Kleinanzeigen-style file. Alternatively copy either file into `data/raw/` and set `paths.raw_csv` in `config/config.yaml` to `data/raw/auto_scout_car.csv` or `data/raw/autos.csv`.

## Local installation

```bash
cd car-price-predictor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train models and write artifacts:

```bash
python -m src.train
python -m src.evaluate
```

Run the web app (`FLASK_DEBUG=1` enables debug/reloader):

```bash
export FLASK_DEBUG=0
python run.py
```

Open `http://127.0.0.1:5000/` for the form, or call the JSON API (see below).

Optional config path:

```bash
export CAR_PREDICTOR_CONFIG=/absolute/path/to/config/config.yaml
python run.py
```

## Docker

Build and run (ensure `models/*.joblib` exist from a prior training run on the host, or train inside the container after copying source):

```bash
docker build -t car-price-predictor .
docker run --rm -p 5000:5000 car-price-predictor
```

Use `curl` against `http://localhost:5000` once the container is up.

### Production-style serve

`gunicorn` is listed for production deployments:

```bash
gunicorn -w 2 -b 0.0.0.0:5000 "run:app"
```

## API documentation

### `POST /api/predict`

**Request body (JSON)** ù all fields required:

| Field           | Type    | Example        |
|----------------|---------|----------------|
| `brand`        | string  | `Maruti`       |
| `year`         | integer | `2016`         |
| `km_driven`    | number  | `45000`        |
| `fuel_type`    | string  | `Petrol`       |
| `transmission` | string  | `Manual`       |
| `owner_type`   | string  | `First Owner`  |

**Response (JSON)**:

```json
{
  "predicted_price": 12.3456,
  "confidence_low": 11.9000,
  "confidence_high": 12.7900,
  "currency": "EUR"
}
```

**Example (`curl`)**:

```bash
curl -s -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Maruti",
    "year": 2016,
    "km_driven": 45000,
    "fuel_type": "Petrol",
    "transmission": "Manual",
    "owner_type": "First Owner"
  }'
```

### HTML routes

- `GET /` ù input form (`app/templates/index.html`).
- `POST /predict` ù form submission; renders `result.html` with prediction and interval.

## Model performance

Metrics below are **held-out test** scores from `python -m src.train` (same split logic as training). After you train on your machine or on real Kaggle data, refresh numbers from `reports/training_metrics.csv`.

| Model             | RMSE (lower is better) | MAE   | Rù    |
|-------------------|-------------------------|-------|-------|
| LinearRegression  | 2.663                   | 2.138 | 0.962 |
| Ridge             | 2.663                   | 2.138 | 0.962 |
| RandomForest      | 2.164                   | 1.528 | 0.975 |
| GradientBoosting  | **2.111**               | 1.558 | **0.976** |

*Example run used synthetic data because no CSV was present; Kaggle metrics will differ.*

The **best** model by test RMSE is saved as `models/model.joblib`. Tree-based intervals use dispersion across boosted stages or forest trees; linear models fall back to training residual spread.

## Project layout

See repository tree in the brief ù configuration lives in `config/config.yaml` (paths, encoders, grids, Flask defaults).

## Notebook

`notebooks/01_eda.ipynb` explores price distribution, brand vs price, mileage scatter, correlation heatmap, and year vs price trend.

## Git ignore notes

`.gitignore` excludes `data/raw/*.csv` and `models/*.joblib`. Commit code and config; **retrain after clone** to regenerate artifacts.
