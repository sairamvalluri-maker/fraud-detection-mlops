
import argparse, pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=str, required=True, help="reference dataset CSV")
    ap.add_argument("--cur", type=str, required=True, help="current dataset CSV")
    ap.add_argument("--out", type=str, default="monitoring/drift_report.html")
    args = ap.parse_args()

    ref = pd.read_csv(args.ref)
    cur = pd.read_csv(args.cur)

    report = Report(metrics=[DataQualityPreset(), DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(args.out)
    print(f"[evidently] saved report -> {args.out}")

if __name__ == "__main__":
    main()
