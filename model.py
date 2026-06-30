#!/usr/bin/env python3
"""
Trains XGBoost model on historical NBA data.
Train: 2021-22 through 2023-24 seasons.
Test:  2024-25 season (out-of-sample).
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
import xgboost as xgb

from features import build_features

MODEL_PATH = Path(__file__).parent / "model.pkl"
TRAIN_CUTOFF = "2024-10-01"


def train(save: bool = True):
    df, feature_cols = build_features()

    train_df = df[df["game_date"] < TRAIN_CUTOFF].copy()
    test_df  = df[df["game_date"] >= TRAIN_CUTOFF].copy()

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test,  y_test  = test_df[feature_cols],  test_df["target"]

    print(f"Train: {len(train_df)} games | Test (OOS): {len(test_df)} games")

    base = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    print(f"Accuracy:    {accuracy_score(y_test, preds):.3f}  (break-even at odds 1.91: 52.4%)")
    print(f"Brier score: {brier_score_loss(y_test, probs):.4f}  (baseline: 0.2500)")
    print(f"AUC-ROC:     {roc_auc_score(y_test, probs):.3f}  (random: 0.500)")

    # Home win rate baseline
    baseline = y_test.mean()
    print(f"Home win rate in test set: {baseline:.3f}")

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "features": feature_cols}, f)
        print(f"Model saved → {MODEL_PATH}")

    return model, feature_cols, test_df.reset_index(drop=True), probs


def load():
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["features"]


if __name__ == "__main__":
    train()
