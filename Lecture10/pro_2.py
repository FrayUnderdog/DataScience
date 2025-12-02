import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42


def remove_outliers_iqr(df, numeric_cols, k=1.5):
    """
    Remove outliers from numeric columns using the IQR rule.
    """
    mask = pd.Series(True, index=df.index)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask &= df[col].between(lower, upper)
    return df.loc[mask].copy()


def preprocess_airline(df_raw):
    """
    Step 1 for Airline dataset:
    - Strip column names.
    - Separate label column ('satisfied').
    - Identify categorical and numeric feature columns.
    - Remove outliers on numeric features.
    - Impute missing values.
    - Build a preprocessing object: OneHotEncoder for categorical features,
      StandardScaler for numeric features, then PCA (3 components can be used later).
    """
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Label column
    label_col = "satisfied"
    y = df[label_col].values
    X = df.drop(columns=[label_col])

    # Column types
    categorical_cols = ["Gender", "CustomerType", "TravelType", "Class"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Remove outliers only from numeric columns (on the full dataset before split)
    df_full = X.copy()
    df_full["__label__"] = y

    df_full = remove_outliers_iqr(df_full, numeric_cols)
    y_clean = df_full["__label__"].values
    X_clean = df_full.drop(columns=["__label__"])

    # Update X, y after outlier removal
    X = X_clean
    y = y_clean

    # Re-identify numeric and categorical just in case
    categorical_cols = ["Gender", "CustomerType", "TravelType", "Class"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Preprocess: imputation and scaling for numeric; imputation and one-hot for categorical
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # PCA on top of preprocessed features
    pca = PCA(n_components=3, random_state=RANDOM_STATE)

    return X, y, preprocessor, pca


def build_super_learner():
    """
    Build a stacking classifier (super learner) for Airline dataset.
    """
    nb = GaussianNB()
    knn = KNeighborsClassifier()
    mlp = MLPClassifier(
        max_iter=200,
        random_state=RANDOM_STATE
    )

    estimators = [
        ("nb", nb),
        ("knn", knn),
        ("mlp", mlp),
    ]

    meta_learner = DecisionTreeClassifier(random_state=RANDOM_STATE)

    super_learner = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        passthrough=False,
        n_jobs=-1
    )

    return super_learner


def tune_and_train_airline_model(X_train, y_train, preprocessor, pca):
    """
    Step 3 + 4 for Airline dataset:
    - Build a full pipeline: preprocessing -> PCA -> StackingClassifier.
    - Run a small GridSearchCV on a sampled subset of data (for speed).
    - Refit the best model on the full training data.
    """
    base_model = build_super_learner()

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("pca", pca),
            ("model", base_model),
        ]
    )

    param_grid = {
        "model__knn__n_neighbors": [3, 5],
        "model__knn__weights": ["uniform", "distance"],
        "model__mlp__hidden_layer_sizes": [(50,), (100,)],
        "model__mlp__alpha": [0.0001, 0.001],
        "model__final_estimator__max_depth": [3, 5],
        "model__final_estimator__min_samples_split": [2, 5],
    }

    # If the dataset is large, use a subset for hyperparameter search
    X_search = X_train
    y_search = y_train
    if X_train.shape[0] > 20000:
        X_search, _, y_search, _ = train_test_split(
            X_train,
            y_train,
            train_size=20000,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_search, y_search)
    print("Best CV accuracy (Airline):", grid_search.best_score_)
    print("Best hyperparameters (Airline):", grid_search.best_params_)

    # Refit best model on full training data
    best_pipeline = grid_search.best_estimator_
    best_pipeline.fit(X_train, y_train)

    return best_pipeline


def main():
    # ===== Step 0: Load data =====
    data_path = "Airline_Satisfaction.csv"
    df_raw = pd.read_csv(data_path)
    print("Original Airline data shape:", df_raw.shape)

    # ===== Step 1: Preprocessing (with outlier removal) =====
    X, y, preprocessor, pca = preprocess_airline(df_raw)
    print("After preprocessing (before train/test split) shape:", X.shape)

    # ===== Step 3: Train/test split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ===== Step 4: Super Learner with PCA + GridSearch =====
    print("\n=== Airline dataset: Super Learner with GridSearch ===")
    best_pipeline = tune_and_train_airline_model(X_train, y_train, preprocessor, pca)

    # Evaluate on test data
    y_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("Test accuracy on Airline dataset:", test_acc)


if __name__ == "__main__":
    main()
