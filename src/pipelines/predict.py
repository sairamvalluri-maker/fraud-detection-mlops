
import json, torch, numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.models.autoencoder import AE
from src.pipelines.preprocess import FEATURES

class Predictor:
    def __init__(self, artifacts_dir: str | Path = "artifacts"):
        self.dir = Path(artifacts_dir)
        self.scaler = StandardScaler()
        self.iso = IsolationForest(n_estimators=200, random_state=42, contamination=0.06)
        self.ae = AE(in_dim=len(FEATURES))
        ae_path = self.dir / "ae.pt"
        if ae_path.exists():
            self.ae.load_state_dict(torch.load(ae_path, map_location="cpu"))
            self.ae.eval()
        thr_path = self.dir / "threshold.json"
        if thr_path.exists():
            with open(thr_path,"r") as f:
                t = json.load(f)
        else:
            t = {}
        self.iso_thr = t.get("iso_thr", None)
        self.ae_thr = t.get("ae_thr", None)

    def score(self, row: dict) -> dict:
        x = np.array([[row.get(k, 0.0) for k in FEATURES]], dtype=np.float32)
        xs = x
        try:
            self.iso.fit(xs)
            iso_score = float(-self.iso.decision_function(xs)[0])
        except Exception:
            iso_score = float(0.0)
        with torch.no_grad():
            xr = self.ae(torch.tensor(xs, dtype=torch.float32))
            ae_score = float(((xr.numpy() - xs)**2).mean())
        iso_flag = int(self.iso_thr is not None and iso_score >= self.iso_thr)
        ae_flag = int(self.ae_thr is not None and ae_score >= self.ae_thr)
        return {"iso_score": iso_score, "ae_score": ae_score, "iso_flag": iso_flag, "ae_flag": ae_flag}
