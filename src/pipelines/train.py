
import argparse, os, json
import numpy as np, pandas as pd, mlflow
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from src.utils.io import read_csv, ensure_dir
from src.pipelines.preprocess import split_and_scale, FEATURES, TARGET
from src.models.autoencoder import train_autoencoder
import torch

def tune_threshold(scores, y_true, pos_label=1):
    best_thr, best_ap = None, -1
    for thr in np.quantile(scores, np.linspace(0.5, 0.99, 30)):
        y_pred = (scores >= thr).astype(int)
        ap = average_precision_score(y_true, y_pred)
        if ap > best_ap:
            best_ap = ap
            best_thr = thr
    return float(best_thr), float(best_ap)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="CSV with features + label")
    ap.add_argument("--outdir", type=str, default="artifacts")
    args = ap.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("fraud-detection")

    df = read_csv(args.data)
    (Xtr, Xva, ytr, yva, scaler) = split_and_scale(df)

    outdir = ensure_dir(args.outdir)

    with mlflow.start_run(run_name="isoforest"):
        iso = IsolationForest(n_estimators=200, random_state=42, contamination=0.06)
        iso.fit(Xtr)
        scores = -iso.decision_function(Xva)
        thr, ap_like = tune_threshold(scores, yva)
        roc = roc_auc_score(yva, scores)
        mlflow.log_params({"model":"IsolationForest","n_estimators":200,"contamination":0.06})
        mlflow.log_metrics({"val_roc_auc": float(roc), "val_ap_thr": float(ap_like)})
        with open(Path(outdir)/"threshold.json","w") as f:
            json.dump({"iso_thr": thr}, f)
        print(f"[ISO] val ROC-AUC={roc:.4f} thr={thr:.4f}")

    with mlflow.start_run(run_name="autoencoder"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ae = train_autoencoder(Xtr, Xva, epochs=8, lr=1e-3, device=device)
        with torch.no_grad():
            import numpy as np
            Xva_t = torch.tensor(Xva, dtype=torch.float32).to(device)
            rec = ae(Xva_t).cpu().numpy()
        mse = ((rec - Xva)**2).mean(axis=1)
        thr, ap_like = tune_threshold(mse, yva)
        roc = roc_auc_score(yva, mse)
        mlflow.log_params({"model":"Autoencoder","epochs":8,"lr":1e-3,"device":device})
        mlflow.log_metrics({"val_roc_auc": float(roc), "val_ap_thr": float(ap_like)})
        torch.save(ae.state_dict(), Path(outdir)/"ae.pt")
        with open(Path(outdir)/"threshold.json","r+") as f:
            d = json.load(f)
        d["ae_thr"] = thr
        with open(Path(outdir)/"threshold.json","w") as f:
            json.dump(d, f, indent=2)
        print(f"[AE]  val ROC-AUC={roc:.4f} thr={thr:.4f}")

    print(f"[train] artifacts saved in {outdir}")

if __name__ == "__main__":
    main()
