# Fraud Detection MLOps Pipeline

An end-to-end **production-style Fraud & Anomaly Detection System** featuring:

* ✅ Isolation Forest baseline model
* ✅ Deep Autoencoder anomaly detector (PyTorch)
* ✅ MLflow experiment tracking + metrics + artifacts
* ✅ Airflow DAG for daily scoring & weekly retraining
* ✅ Evidently data & prediction drift monitoring
* ✅ FastAPI real-time scoring endpoint
* ✅ Dockerfile for deployment
* ✅ Clean, modular project architecture

This project mirrors how real fintech / risk engineering teams build fraud pipelines.

---

## 🔥 Project Architecture

```
                 ┌───────────────────────────┐
                 │       Raw Transactions     │
                 └──────────────┬────────────┘
                                │
                                ▼
                    (1) Synthetic / Real Data

                                ▼
                 ┌───────────────────────────┐
                 │  Feature Engineering +     │
                 │  Scaling (sklearn)         │
                 └──────────────┬────────────┘
                                │
                                ▼
                 ┌───────────────────────────┐
                 │ Train Models               │
                 │ - Isolation Forest         │
                 │ - Autoencoder (PyTorch)    │
                 └──────────────┬────────────┘
                                │ artifacts/
                                ▼
                      Saved Models + Thresholds

                                ▼
                 ┌───────────────────────────┐
                 │ FastAPI Scoring Service   │
                 │  /score endpoint          │
                 └──────────────┬────────────┘
                                │
                                ▼
                        Real-time Predictions

                                ▼
                 ┌───────────────────────────┐
                 │ Drift Monitoring (Evid.)  │
                 │ - Data Drift              │
                 │ - Feature Drift           │
                 └──────────────┬────────────┘
                                ▼
                      HTML drift reports

                                ▼
                 ┌───────────────────────────┐
                 │ Airflow DAG               │
                 │ - Daily scoring           │
                 │ - Weekly retraining       │
                 └───────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python src/utils/generate_data.py --rows 50000 --out data/transactions.csv
```

### 3. Train Models (logs to MLflow)

```bash
python -m src.pipelines.train --data data/transactions.csv --outdir artifacts
```

Artifacts generated:

* `artifacts/ae.pt` — trained autoencoder
* `artifacts/threshold.json` — tuned thresholds

### 4. Launch API

```bash
uvicorn api.main:app --reload
```

Test:

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"amount": 220, "hour": 3, "distance": 12, "device_score": 0.92, "country_mismatch": 1}'
```

---

## 📊 MLflow Tracking

Run MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5001
```

Open:

👉 [http://127.0.0.1:5001](http://127.0.0.1:5001)

You will see:

* model runs
* metrics (ROC-AUC, AP@threshold)
* parameters
* artifacts (thresholds + models)

---

## 🧠 Models Included

### ✅ Isolation Forest (Sklearn)

* Contamination = 6%
* Good baseline anomaly detector

### ✅ Autoencoder (PyTorch)

Layers:

```
Input → 16 → 8 → 16 → Output
```

Reconstruction error used as anomaly score.

Lower reconstruction = normal
Higher reconstruction = suspicious

Thresholds are tuned via validation AP.

---

## 📈 Drift Monitoring (Evidently)

Generate drift report:

```bash
python monitoring/evidently_report.py --ref data/transactions.csv --cur data/transactions.csv
```

Output:

```
monitoring/drift_report.html
```

Open in browser to view:

* Data quality
* Feature drift
* Statistical tests

---

## ⏱ Airflow Pipeline

Located at:

```
airflow/dags/fraud_pipeline.py
```

Includes:

* Daily scoring task
* Weekly retraining task
* Synthetic data refresh

Drop this DAG into Airflow to activate automated retraining.

---

## 🐳 Docker Support

Build image:

```bash
docker build -t fraud-api .
```

Run container:

```bash
docker run -p 8000:8000 fraud-api
```

---

## 📁 Project Structure

```
fraud-detection-mlops/
  data/
  notebooks/
  src/
    utils/
    models/
    pipelines/
  api/
  monitoring/
  airflow/
  artifacts/
  docker/
  mlruns/
  requirements.txt
  README.md
```

---



