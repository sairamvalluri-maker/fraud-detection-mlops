
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.pipelines.predict import Predictor

app = FastAPI(title="Fraud Detection Scoring API")
predictor = Predictor(artifacts_dir="artifacts")

class Txn(BaseModel):
    amount: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)
    distance: float = Field(..., ge=0)
    device_score: float = Field(..., ge=0, le=1)
    country_mismatch: int = Field(..., ge=0, le=1)

@app.post("/score")
def score(txn: Txn):
    res = predictor.score(txn.model_dump())
    return {"ok": True, "result": res}
