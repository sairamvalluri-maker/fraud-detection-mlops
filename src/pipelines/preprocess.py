
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["amount","hour","distance","device_score","country_mismatch"]
TARGET = "label"

def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df[FEATURES].copy()
    y = df[TARGET].copy() if TARGET in df.columns else None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    return (X_train_s, X_val_s, y_train.values, y_val.values, scaler)
