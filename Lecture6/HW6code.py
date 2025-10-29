# HW6code.py
# -*- coding: utf-8 -*-
"""
End-to-end feature selection / extraction and KNN comparison on world_ds.csv
- Forward wrapper (3 features) using sklearn SequentialFeatureSelector
- PCA (3 components)
- LDA (2 components)
"""

from pathlib import Path
import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.exceptions import NotFittedError


def main():
    # ------------------------------------------------------------
    # 0) Paths: use the SAME directory as this script (__file__)
    # ------------------------------------------------------------
    script_dir = Path(__file__).parent.resolve()
    data_path = script_dir / "world_ds.csv"  # dataset in the same folder
    out_forward = script_dir / "forward_selected_features.csv"
    out_loadings = script_dir / "pca_loadings_top3.csv"
    out_cmp = script_dir / "knn_accuracy_comparison.csv"

    # ------------------------------------------------------------
    # 1) Load dataset with clear error message if missing
    # ------------------------------------------------------------
    if not data_path.exists():
        print(
            f"[ERROR] CSV not found: {data_path}\n"
            f"Please make sure 'world_ds.csv' is placed in the SAME folder as this script:\n{script_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(data_path)

    # ------------------------------------------------------------
    # 2) Prepare features and target
    # ------------------------------------------------------------
    label_col = "development_status"
    id_cols = ["Country", label_col]
    if not all(col in df.columns for col in id_cols):
        print(
            "[ERROR] Expected columns 'Country' and 'development_status' not found in the CSV.",
            file=sys.stderr,
        )
        sys.exit(1)

    X_df = df.drop(columns=id_cols)
    y = df[label_col].astype(int)

    # Sanity check for numeric dtypes
    if not all(pd.api.types.is_numeric_dtype(X_df[c]) for c in X_df.columns):
        print(
            "[ERROR] Non-numeric feature found. Please ensure all features (except 'Country' and label) are numeric.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------
    # 3) Shared config
    # ------------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn9 = KNeighborsClassifier(n_neighbors=9)

    # ------------------------------------------------------------
    # 4) Forward wrapper selection (select best 3 features)
    #    - Fit SFS once on standardized features to get subset
    #    - Then evaluate KNN with 5-fold CV on those features
    # ------------------------------------------------------------
    scaler_global = StandardScaler().fit(X_df)
    X_scaled = scaler_global.transform(X_df)

    sfs = SequentialFeatureSelector(
        knn9,
        n_features_to_select=3,
        direction="forward",
        cv=5,      # internal CV used by SFS
        n_jobs=1,  # robust for limited environments
    )
    sfs.fit(X_scaled, y)
    support_mask = sfs.get_support()
    selected_features = X_df.columns[support_mask].tolist()

    pipe_knn_sel = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", knn9),
    ])
    acc_forward = cross_val_score(
        pipe_knn_sel, X_df[selected_features], y, cv=cv, scoring="accuracy"
    )
    forward_mean, forward_std = acc_forward.mean(), acc_forward.std()

    # Save selected features
    pd.DataFrame({"selected_feature": selected_features}).to_csv(out_forward, index=False)

    # ------------------------------------------------------------
    # 5) PCA (3 components) + KNN
    # ------------------------------------------------------------
    pipe_pca = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=3, random_state=42)),
        ("knn", knn9),
    ])
    acc_pca = cross_val_score(pipe_pca, X_df, y, cv=cv, scoring="accuracy")
    pca_mean, pca_std = acc_pca.mean(), acc_pca.std()

    # Fit PCA on globally standardized data for interpretation (loadings)
    pca_model = PCA(n_components=3, random_state=42).fit(X_scaled)
    loadings = pd.DataFrame(
        pca_model.components_.T,
        index=X_df.columns,
        columns=[f"PC{i+1}" for i in range(3)]
    )
    loadings.to_csv(out_loadings, index=True)

    # ------------------------------------------------------------
    # 6) LDA (2 components) + KNN
    #    Note: with 3 classes, LDA yields up to 2 components.
    # ------------------------------------------------------------
    pipe_lda = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LDA(n_components=2)),
        ("knn", knn9),
    ])
    acc_lda = cross_val_score(pipe_lda, X_df, y, cv=cv, scoring="accuracy")
    lda_mean, lda_std = acc_lda.mean(), acc_lda.std()

    # ------------------------------------------------------------
    # 7) Comparison table
    # ------------------------------------------------------------
    cmp_df = pd.DataFrame({
        "Method": ["Forward (3 features)", "PCA (3 components)", "LDA (2 components)"],
        "KNN (K=9) Accuracy (mean 5-fold CV)": [forward_mean, pca_mean, lda_mean],
        "Std (5-fold CV)": [forward_std, pca_std, lda_std],
    }).sort_values(
        by="KNN (K=9) Accuracy (mean 5-fold CV)",
        ascending=False
    ).reset_index(drop=True)

    cmp_df.to_csv(out_cmp, index=False)

    # ------------------------------------------------------------
    # 8) Console summary
    # ------------------------------------------------------------
    print("====================================================")
    print("Selected features (Forward):", selected_features)
    print("Results saved to:")
    print(f" - Forward features:        {out_forward}")
    print(f" - PCA loadings (3 PCs):    {out_loadings}")
    print(f" - KNN accuracy comparison: {out_cmp}")
    print("----------------------------------------------------")
    print(cmp_df)
    print("====================================================")


if __name__ == "__main__":
    try:
        main()
    except NotFittedError as e:
        print(f"[ERROR] Model fitting issue: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
