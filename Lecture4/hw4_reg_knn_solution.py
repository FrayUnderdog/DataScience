#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------- I/O helpers ----------------------------

def find_datasets() -> Tuple[Path, Path, bool]:
    """
    Return train_path, test_path, and a boolean flag 'fallback_split'.
    If 'hw4_test.csv' does not exist, 'fallback_split' will be True and
    a temporary split will be performed from the train file.
    """
    base = Path(".")
    train_path = (base / "hw4_train.csv").resolve()
    test_path = (base / "hw4_test.csv").resolve()

    if not train_path.exists():
        print("ERROR: 'hw4_train.csv' not found in current directory.", file=sys.stderr)
        sys.exit(1)

    if not test_path.exists():
        # No explicit test set; we will create one by splitting the train set.
        return train_path, test_path, True

    return train_path, test_path, False


def save_artifacts(test_df: pd.DataFrame, k_acc_df: pd.DataFrame, best_k: int, best_acc: float) -> None:
    """
    Save generated outputs to disk.
    """
    test_out = Path("hw4_test_with_bp.csv")
    acc_out = Path("knn_k_vs_accuracy.csv")
    report_out = Path("hw4_report.txt")

    # Save the test dataset with predicted BloodPressure
    test_df.to_csv(test_out, index=False)

    # Save K-vs-Accuracy table
    k_acc_df.to_csv(acc_out, index=False)

    # Save a short text report
    report_lines = [
        "HW4 Regression + KNN Report",
        "---------------------------------------",
        f"Best k: {best_k}",
        f"Best accuracy: {best_acc:.4f}",
        "",
        "Files generated:",
        f"- {test_out.name}: TEST dataset with predicted/final 'BloodPressure' values.",
        f"- {acc_out.name}: Accuracy for k=1..19.",
    ]
    Path(report_out).write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[INFO] Saved: {test_out.resolve()}")
    print(f"[INFO] Saved: {acc_out.resolve()}")
    print(f"[INFO] Saved: {report_out.resolve()}")


# --------------------------- Core logic ----------------------------

def fit_regression_and_fill_bp(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, Pipeline]:
    """
    Fit a multiple linear regression on TRAIN with target 'BloodPressure' and
    features = all columns except ['Outcome', 'BloodPressure'].
    Predict 'BloodPressure' for TEST and SET the column values accordingly.
    Returns the modified test_df and the fitted regression pipeline.
    """
    if "BloodPressure" not in train_df.columns:
        raise ValueError("TRAIN dataset must contain 'BloodPressure'.")

    # Identify columns
    excluded_cols = {"Outcome", "BloodPressure"}
    feature_cols = [c for c in train_df.columns if c not in excluded_cols]

    # Separate X, y for regression
    X_train_reg = train_df[feature_cols]
    y_train_reg = train_df["BloodPressure"]

    # Build pipeline: impute (median) -> linear regression
    reg_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", LinearRegression())
        ]
    )

    reg_pipeline.fit(X_train_reg, y_train_reg)
    print("[INFO] Regression model trained.")

    # Prepare TEST features
    # If 'BloodPressure' column is missing from TEST, create it after prediction.
    X_test_reg = test_df.reindex(columns=feature_cols)
    y_test_pred = reg_pipeline.predict(X_test_reg)

    # Set/overwrite 'BloodPressure' in TEST
    test_df = test_df.copy()
    test_df["BloodPressure"] = y_test_pred
    print("[INFO] 'BloodPressure' values in TEST have been set using regression predictions.")

    return test_df, reg_pipeline


def evaluate_knn(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    """
    Train 19 KNN models (k=1..19) on TRAIN with label 'Outcome' and
    features = all columns except 'Outcome'. Evaluate accuracy on TEST.
    Returns a DataFrame of (k, accuracy), the best_k, and best_acc.
    """
    if "Outcome" not in train_df.columns or "Outcome" not in test_df.columns:
        raise ValueError("Both TRAIN and TEST datasets must contain 'Outcome' for accuracy computation.")

    # Features are all columns except the target label
    feature_cols = [c for c in train_df.columns if c != "Outcome"]

    X_train = train_df[feature_cols]
    y_train = train_df["Outcome"]
    X_test = test_df[feature_cols]
    y_test = test_df["Outcome"]

    # Numeric-only selection (KNN requires numeric features)
    # If there are any non-numeric columns, they will be dropped.
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train = X_train[num_cols]
    X_test = X_test[num_cols]

    # Build a pipeline for KNN: impute (median) -> scale -> KNN
    results = []
    best_k, best_acc = None, -1.0

    for k in range(1, 20):
        knn_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k))
            ]
        )
        knn_pipeline.fit(X_train, y_train)
        y_pred = knn_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({"k": k, "accuracy": acc})

        if acc > best_acc or (acc == best_acc and (best_k is None or k < best_k)):
            best_acc = acc
            best_k = k

    k_acc_df = pd.DataFrame(results)
    print("[INFO] KNN evaluation complete.")
    print(k_acc_df)

    print(f"[RESULT] Best k = {best_k}, Accuracy = {best_acc:.4f}")
    return k_acc_df, best_k, best_acc


def main():
    train_path, test_path, fallback_split = find_datasets()
    print(f"[INFO] Using train file: {train_path.name}")
    if fallback_split:
        print("[WARN] 'hw4_test.csv' not found. Falling back to a random split from the train file "
              "so the full pipeline can be executed. To strictly follow the assignment, add 'hw4_test.csv'.")

    # Load TRAIN
    train_df_full = pd.read_csv(train_path)

    # If TEST missing, create a split (75% train, 25% test) while preserving 'Outcome' distribution when possible
    if fallback_split:
        # We will split only for classification stage; regression uses full remaining after split
        train_df, test_df = train_test_split(train_df_full, test_size=0.25, stratify=train_df_full["Outcome"] if "Outcome" in train_df_full.columns else None, random_state=42)
    else:
        test_df = pd.read_csv(test_path)
        train_df = train_df_full

    # 1) Fit regression on TRAIN and set 'BloodPressure' in TEST
    test_df_with_bp, reg_model = fit_regression_and_fill_bp(train_df, test_df)

    # 2) Train 19 KNN models (k=1..19) and evaluate on TEST
    k_acc_df, best_k, best_acc = evaluate_knn(train_df, test_df_with_bp)

    # 3) Save artifacts
    save_artifacts(test_df_with_bp, k_acc_df, best_k, best_acc)


if __name__ == "__main__":
    main()
