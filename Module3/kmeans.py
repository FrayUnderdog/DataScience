# kmeans_marketing_english.py
# --------------------------------------------------------
# 1) Find the optimal K for K-Means on marketing dataset
# 2) Fit K-Means with the optimal K
# 3) Plot Income–Spending and Income–Age scatter plots
# 4) Assign descriptive names to clusters based on z-scores
# --------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ------------------------
# Load and prepare data
# ------------------------
CSV_PATH = "market_ds.csv"
FEATURES = ["Age", "Income", "Spending"]
RANDOM_STATE = 42

df = pd.read_csv(CSV_PATH)
X = df[FEATURES].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------
# Determine optimal number of clusters
# ------------------------
inertias = []
sil_scores = []
K_RANGE = range(2, 11)

for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)

optimal_k = K_RANGE[np.argmax(sil_scores)]
print(f"Optimal number of clusters (by silhouette): {optimal_k}")

# ------------------------
# Fit final K-Means model
# ------------------------
kmeans = KMeans(n_clusters=optimal_k, n_init="auto", random_state=RANDOM_STATE)
labels = kmeans.fit_predict(X_scaled)
df["cluster"] = labels

centroids_scaled = kmeans.cluster_centers_
centroids = pd.DataFrame(
    scaler.inverse_transform(centroids_scaled),
    columns=FEATURES
)
centroids["cluster"] = range(optimal_k)

# ------------------------
# Cluster summary and naming
# ------------------------
summary = df.groupby("cluster")[FEATURES].mean().round(2)
summary["count"] = df.groupby("cluster").size()

global_mean = X.mean()
global_std = X.std(ddof=0)
z = (summary[FEATURES] - global_mean) / global_std
z = z.rename(columns={c: f"{c}_z" for c in z.columns})
summary_z = pd.concat([summary, z], axis=1)

def name_cluster(row):
    inc_z = row["Income_z"]
    spend_z = row["Spending_z"]
    age_z = row["Age_z"]

    if inc_z > 0.7 and spend_z > 0.7:
        return "Affluent Big Spenders"
    if inc_z > 0.7 and spend_z < -0.3:
        return "Frugal Affluents"
    if inc_z < -0.5 and spend_z > 0.5:
        return "Aspirational Spenders"
    if spend_z < -0.6:
        return "Budget-Minded Savers"
    if age_z < -0.5 and spend_z > 0.2:
        return "Young Spenders"
    if age_z > 0.7 and spend_z < 0:
        return "Senior Savers"
    return "Balanced Regulars"

summary_z["cluster_name"] = summary_z.apply(name_cluster, axis=1)

print("\n=== Cluster Summary ===")
print(summary_z[[*FEATURES, "count", "Age_z", "Income_z", "Spending_z", "cluster_name"]]
      .sort_index().to_string())

# ------------------------
# Plots
# ------------------------
plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), inertias, marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(list(K_RANGE), sil_scores, marker="o")
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

# Income vs Spending
plt.figure(figsize=(6, 5))
plt.scatter(df["Income"], df["Spending"], c=df["cluster"], alpha=0.7)
plt.scatter(centroids["Income"], centroids["Spending"], marker="X", s=200, edgecolor="k")
plt.title(f"Income vs Spending (k={optimal_k})")
plt.xlabel("Income")
plt.ylabel("Spending")
plt.tight_layout()
plt.show()

# Income vs Age
plt.figure(figsize=(6, 5))
plt.scatter(df["Income"], df["Age"], c=df["cluster"], alpha=0.7)
plt.scatter(centroids["Income"], centroids["Age"], marker="X", s=200, edgecolor="k")
plt.title(f"Income vs Age (k={optimal_k})")
plt.xlabel("Income")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# ------------------------
# Save results
# ------------------------
df_out = df.copy()
name_map = summary_z["cluster_name"].to_dict()
df_out["cluster_name"] = df_out["cluster"].map(name_map)

df_out.to_csv("marketing_clustered.csv", index=False)
summary_z.to_csv("cluster_summary.csv")

print("\nFiles saved:")
print(" - marketing_clustered.csv (dataset with cluster labels)")
print(" - cluster_summary.csv (cluster statistics)")
