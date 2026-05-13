# Used Car Price Prediction 🚗💡

A production-grade Machine Learning application that predicts fair market vehicle prices. This project integrates a high-accuracy `GradientBoostingRegressor` with a premium **Glassmorphism web interface** featuring dynamic **3D X-Ray car blueprints**.

---

## 📁 Repository Structure

| Path                        | Description                                                                    |
| --------------------------- | ------------------------------------------------------------------------------ |
| **`app/`**                  | Flask application core (routes, HTML templates, result logic)                  |
| **`static/`**               | Web assets: CSS (Glassmorphism), `car_data.json` (UI logic), and 3D images     |
| **`src/`**                  | Data Science pipeline: `data_loader`, `preprocessing`, `train`, and `evaluate` |
| **`models/`**               | Serialized ML artifacts: `model.joblib`, `preprocessor.joblib`                 |
| **`reports/`**              | Model performance charts and 3D mechanical blueprints                          |
| **`data/`**                 | Training data storage for raw CSV files                                        |
| **`generate_menu_data.py`** | Script to pre-process data for the smart frontend dropdowns                    |
| **`run.py`**                | Main entry point to launch the Flask web server                                |

---

## ✨ Features

- **Premium Glassmorphism UI:** A modern, translucent interface with high-end CSS blur effects and backdrop filters.
- **3D X-Ray Visualization:** The background intelligently swaps between SUV, Sedan, Hatchback, and Station Wagon blueprints based on the vehicle selected.
- **Smart UI Logic:** Frontend dropdowns are driven by a pre-calculated dependency tree, ensuring valid Make/Model/HP combinations and auto-detecting body types.
- **Robust ML Pipeline:** Multi-model training using `GridSearchCV` to find the best performing estimator with leakage-aware target encoding.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/NikhilTanwar48/used-car-price-prediction.git
cd used-car-price-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


### 2. Setup Data & UI

Generate the metadata for the smart frontend dropdowns:Bashpython generate_menu_data.py

### 3. Run the Application

Start the Flask server:
python run.py
Visit http://127.0.0.1:8080 in your browser to experience the dashboard.

### 📊 Model Evaluation & Insights

The system evaluates several models, with the Gradient Boosting regressor currently providing the highest fidelity. Performance charts are automatically updated in the reports/figures/ directory after training.

### Feature Importance

The model confirms that brand prestige (brand_avg_price) and usage (km_driven) are the primary drivers of market value.

### Model Accuracy

The alignment between predicted and actual prices shows a high $R^2$ score (~0.97), indicating reliable performance across various price segments.

### Residual Analysis

Residuals are randomly scattered, indicating that the model has captured the underlying data patterns without significant bias.

### 🛠️ Configuration

Central settings live in config/config.yaml. This allows for seamless modification of:
File paths and directory structures.
Hyperparameter grids for GridSearchCV.
Dataset column mapping for different CSV formats (Cardekho, AutoScout24, etc.).

### 📝 Development Notes

Git Ignore: Large CSV files and .joblib models are excluded via .gitignore to keep the repository lightweight.
Inference: Predictions include a confidence interval (High/Low) to represent market volatility and model uncertainty.
Logging: The project utilizes a structured logging system instead of standard print statements for better production monitoring.
```
