
# fraud-detection-mlops

End-to-end fraud and anomaly detection pipeline with **Isolation Forest** and a **PyTorch Autoencoder**, plus **MLflow tracking**, an **Airflow DAG** for retraining, **Evidently** drift monitoring, and a **FastAPI** scoring service.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/utils/generate_data.py --rows 50000 --out data/transactions.csv
python src/pipelines/train.py --data data/transactions.csv --outdir artifacts

uvicorn api.main:app --reload
curl -X POST "http://127.0.0.1:8000/score" -H "Content-Type: application/json" -d '{"amount": 129.9, "hour": 13, "distance": 2.1, "device_score": 0.82, "country_mismatch": 0}'
```
