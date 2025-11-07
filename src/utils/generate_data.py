
import argparse, numpy as np, pandas as pd
from pathlib import Path

def make_synthetic(rows: int = 50000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = rng.gamma(shape=2.0, scale=40.0, size=rows)
    hour = rng.integers(0, 24, size=rows)
    distance = rng.exponential(scale=3.0, size=rows)
    device_score = rng.uniform(0, 1, size=rows)
    country_mismatch = rng.binomial(1, 0.08, size=rows)

    z = (amount/120) + (distance/5) + (device_score*1.5) + (country_mismatch*2.0) + (hour%3==0)*0.3
    prob_fraud = 1/(1+np.exp(-(z-2.2)))
    fraud = rng.binomial(1, prob_fraud)

    df = pd.DataFrame({
        "amount": amount,
        "hour": hour,
        "distance": distance,
        "device_score": device_score,
        "country_mismatch": country_mismatch,
        "label": fraud
    })
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50000)
    ap.add_argument("--out", type=str, default="data/transactions.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = make_synthetic(rows=args.rows, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[data] wrote {len(df):,} rows to {out}")

if __name__ == "__main__":
    main()
