import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42


def remove_outliers_iqr(df, numeric_cols, k=1.5):
    """
    Remove outliers using IQR rule for each numeric column.
    Rows with any numeric feature outside [Q1 - k*IQR, Q3 + k*IQR] will be removed.
    """
    mask = pd.Series(True, index=df.index)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask &= df[col].between(lower, upper)
    cleaned_df = df.loc[mask].copy()
    return cleaned_df


def preprocess_and_label_diabetes(df_raw):
    """
    Step 1 + Step 2:
    - Remove outliers
    - Impute missing values
    - K-means on Glucose, BMI, Age -> create pseudo labels (Outcome)
    - Normalize all feature columns (excluding Outcome)
    """
    df = df_raw.copy()

    # numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove outliers
    df = remove_outliers_iqr(df, numeric_cols)

    # Impute missing values (median)
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # K-means clustering on 3 required columns
    kmeans_features = ["Glucose", "BMI", "Age"]
    for col in kmeans_features:
        if col not in df.columns:
            raise ValueError(f"Required feature '{col}' is missing.")

    # Scale before kmeans
    scaler_kmeans = StandardScaler()
    X_kmeans = scaler_kmeans.fit_transform(df[kmeans_features])

    # KMeans
    kmeans = KMeans(
        n_clusters=2,
        random_state=RANDOM_STATE,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(X_kmeans)
    df["cluster"] = cluster_labels

    # Determine which cluster is "Diabetes" (higher glucose mean)
    cluster_means = df.groupby("cluster")["Glucose"].mean()
    diabetes_cluster = cluster_means.idxmax()

    df["Outcome"] = (df["cluster"] == diabetes_cluster).astype(int)
    df.drop(columns=["cluster"], inplace=True)

    # Normalize all feature columns except Outcome
    feature_cols = [c for c in df.columns if c != "Outcome"]
    scaler_features = StandardScaler()
    X_scaled = scaler_features.fit_transform(df[feature_cols])

    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    scaled_df["Outcome"] = df["Outcome"]

    return scaled_df, feature_cols


def train_super_learner_with_pca(scaled_df, label_col="Outcome"):
    """
    Step 3 + Step 4:
    - Train/test split
    - PCA (3 components)
    - Super learner (NB + KNN + MLP) with DecisionTree meta learner
    - 5-fold CV, GridSearchCV
    - n_jobs = 1 for full stability on Windows
    """
    X = scaled_df.drop(columns=[label_col]).values
    y = scaled_df[label_col].values

    # Split data (20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # PCA to 3 components
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Base models
    base_estimators = [
        ("nb", GaussianNB()),
        ("knn", KNeighborsClassifier()),
        ("mlp", MLPClassifier(
            max_iter=500,
            random_state=RANDOM_STATE,
            early_stopping=True
        )),
    ]

    # Meta learner
    meta_learner = DecisionTreeClassifier(random_state=RANDOM_STATE)

    # Stacking classifier
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=1   # IMPORTANT: set to 1 for stability
    )

    # Hyperparameter grid
    param_grid = {
        "knn__n_neighbors": [3, 5, 7],
        "knn__weights": ["uniform", "distance"],
        "mlp__hidden_layer_sizes": [(50,), (100,)],
        "mlp__alpha": [0.0001, 0.001],
        "final_estimator__max_depth": [3, 5, None],
        "final_estimator__min_samples_leaf": [1, 3],
    }

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=stacking_clf,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=1,   # IMPORTANT: set to 1 for stability
        verbose=0   # IMPORTANT: suppress output to avoid decode errors
    )

    print("Training super learner with PCA...")

    grid_search.fit(X_train_pca, y_train)

    print("Best CV accuracy:", grid_search.best_score_)
    print("Best hyperparameters:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Evaluate on test data
    y_pred = best_model.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, y_pred)

    print("Test accuracy on Diabetes dataset:", test_accuracy)

    return best_model, pca, test_accuracy


def preprocess_labeled_dataset_for_step5(df_raw, label_col):
    """
    Step 5:
    Preprocess a labeled dataset:
    - Remove outliers (numeric)
    - Impute numeric features
    - Normalize numeric features
    NOTE: If categorical variables exist, encode them before scaling.
    """
    df = df_raw.copy()

    # Drop missing labels
    df = df.dropna(subset=[label_col])

    feature_cols = [c for c in df.columns if c != label_col]

    # Only numeric columns for outlier removal
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Remove outliers based on numeric columns
    df_numeric = df[numeric_cols + [label_col]]
    cleaned_numeric = remove_outliers_iqr(df_numeric, numeric_cols)
    df = df.loc[cleaned_numeric.index]

    # Re-select features and label
    X = df[feature_cols]
    y = df[label_col]

    # Impute numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy="median")
    X_numeric_imputed = imputer.fit_transform(X[numeric_cols])

    # Normalize
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric_imputed)

    scaled_df = pd.DataFrame(X_numeric_scaled, columns=numeric_cols, index=X.index)
    scaled_df[label_col] = y.values

    return scaled_df


def main():
    diabetes_path = "diabetes_project.csv"

    df_diabetes = pd.read_csv(diabetes_path)
    print("Original Diabetes data shape:", df_diabetes.shape)

    scaled_diabetes_df, feature_cols = preprocess_and_label_diabetes(df_diabetes)
    print("After preprocessing and labeling shape:", scaled_diabetes_df.shape)

    best_model, pca, test_acc = train_super_learner_with_pca(scaled_diabetes_df, label_col="Outcome")


if __name__ == "__main__":
    main()
