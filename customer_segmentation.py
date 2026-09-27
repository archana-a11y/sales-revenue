# ============================================================
# CUSTOMER SEGMENTATION PROJECT
# Thiranex - Skill Development & Future Tech
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -----------------------------
# 1. LOAD DATA
# -----------------------------
DATA_FILE = "customer_data.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Keep customer_data.csv in the same folder."
    )

df = pd.read_csv(DATA_FILE)

print("\n========== CUSTOMER SEGMENTATION ==========\n")
print("Dataset loaded successfully!")
print(f"Total customers: {len(df)}")

# -----------------------------
# 2. BASIC DATA CHECK
# -----------------------------
print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATA TYPES ==========\n")
print(df.dtypes)

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# -----------------------------
# 3. CREATE OUTPUT FOLDER
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 4. BASIC VISUALIZATION
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/01_age_distribution.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["AnnualIncome"], bins=30, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/02_income_distribution.png", dpi=150)
plt.show()

# -----------------------------
# 5. FEATURES FOR CLUSTERING
# -----------------------------
features = [
    "Age",
    "AnnualIncome",
    "PurchaseFrequency",
    "AveragePurchaseValue",
    "TotalPurchaseAmount",
    "WebsiteVisits",
    "DiscountUsage"
]

X = df[features]

# -----------------------------
# 6. STANDARDIZE FEATURES
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures standardized successfully.")

# -----------------------------
# 7. ELBOW METHOD
# -----------------------------
inertia = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/03_elbow_method.png", dpi=150)
plt.show()

# -----------------------------
# 8. K-MEANS CLUSTERING
# -----------------------------
N_CLUSTERS = 4

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# Silhouette score
silhouette = silhouette_score(X_scaled, df["Cluster"])
print(f"\nSilhouette Score: {silhouette:.3f}")

# -----------------------------
# 9. AUTOMATIC SEGMENT LABELS
# -----------------------------
# Label clusters according to their average spending/frequency.
cluster_summary = df.groupby("Cluster")[[
    "AnnualIncome",
    "PurchaseFrequency",
    "AveragePurchaseValue",
    "TotalPurchaseAmount",
    "WebsiteVisits"
]].mean()

score = (
    cluster_summary["PurchaseFrequency"].rank(pct=True)
    + cluster_summary["AveragePurchaseValue"].rank(pct=True)
    + cluster_summary["TotalPurchaseAmount"].rank(pct=True)
)

ordered_clusters = score.sort_values().index.tolist()

segment_names = [
    "Occasional Customers",
    "Budget Customers",
    "Frequent Customers",
    "High Value Customers"
]

cluster_to_segment = {
    cluster: segment_names[i]
    for i, cluster in enumerate(ordered_clusters)
}

df["Segment"] = df["Cluster"].map(cluster_to_segment)

# -----------------------------
# 10. SEGMENT COUNT
# -----------------------------
segment_count = df["Segment"].value_counts()

print("\n========== CUSTOMER SEGMENTS ==========\n")
print(segment_count)

plt.figure(figsize=(9, 5))
sns.countplot(
    data=df,
    x="Segment",
    order=segment_count.index
)
plt.title("Number of Customers in Each Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/04_segment_count.png", dpi=150)
plt.show()

# -----------------------------
# 11. SEGMENT ANALYSIS
# -----------------------------
segment_analysis = df.groupby("Segment")[features].mean().round(2)

print("\n========== SEGMENT CHARACTERISTICS ==========\n")
print(segment_analysis)

segment_analysis.to_csv("outputs/segment_analysis.csv")

# -----------------------------
# 12. INCOME VS PURCHASE
# -----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="AnnualIncome",
    y="TotalPurchaseAmount",
    hue="Segment",
    s=70
)
plt.title("Income vs Total Purchase Amount")
plt.xlabel("Annual Income")
plt.ylabel("Total Purchase Amount")
plt.legend(title="Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("outputs/05_income_vs_purchase.png", dpi=150)
plt.show()

# -----------------------------
# 13. PURCHASE FREQUENCY VS VALUE
# -----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="PurchaseFrequency",
    y="AveragePurchaseValue",
    hue="Segment",
    s=70
)
plt.title("Purchase Frequency vs Average Purchase Value")
plt.xlabel("Purchase Frequency")
plt.ylabel("Average Purchase Value")
plt.legend(title="Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("outputs/06_frequency_vs_value.png", dpi=150)
plt.show()

# -----------------------------
# 14. PCA VISUALIZATION
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Segment": df["Segment"]
})

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Segment",
    s=70
)
plt.title("Customer Segments using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("outputs/07_pca_segments.png", dpi=150)
plt.show()

# -----------------------------
# 15. HEATMAP
# -----------------------------
plt.figure(figsize=(12, 7))
sns.heatmap(
    segment_analysis,
    annot=True,
    fmt=".1f",
    cmap="Blues"
)
plt.title("Customer Segment Characteristics")
plt.xlabel("Features")
plt.ylabel("Customer Segment")
plt.tight_layout()
plt.savefig("outputs/08_segment_heatmap.png", dpi=150)
plt.show()

# -----------------------------
# 16. SAVE FINAL DATA
# -----------------------------
df.to_csv(
    "outputs/customer_segmentation_results.csv",
    index=False
)

# -----------------------------
# 17. FINAL REPORT
# -----------------------------
print("\n========== PROJECT COMPLETED ==========\n")
print(f"Total Customers: {len(df)}")
print(f"Number of Segments: {df['Segment'].nunique()}")
print(f"Silhouette Score: {silhouette:.3f}")

print("\nSegment Summary:")
for segment, count in segment_count.items():
    print(f"- {segment}: {count} customers")

print("\nFiles saved inside the outputs folder.")
