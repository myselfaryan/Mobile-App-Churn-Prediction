import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("mobile_app_data.csv")

# --- Preprocessing for Segmentation ---

# Convert last_activity_date and last_purchase_date to datetime objects
df["last_activity_date"] = pd.to_datetime(df["last_activity_date"])
df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])

# Recency: Days since last activity
df["recency"] = (pd.to_datetime("today") - df["last_activity_date"]).dt.days

# Select features for clustering (excluding user_id and churned)
features_for_clustering = df.drop(columns=["user_id", "churned", "last_activity_date", "last_purchase_date"])

# Handle missing values
# Numerical columns: Impute with median
numerical_cols = features_for_clustering.select_dtypes(include=np.number).columns.tolist()
imputer_numerical = SimpleImputer(strategy="median")
features_for_clustering[numerical_cols] = imputer_numerical.fit_transform(features_for_clustering[numerical_cols])

# Categorical columns: Impute with most frequent
categorical_cols = features_for_clustering.select_dtypes(include='object').columns.tolist()
imputer_categorical = SimpleImputer(strategy="most_frequent")
features_for_clustering[categorical_cols] = imputer_categorical.fit_transform(features_for_clustering[categorical_cols])

# One-hot encode categorical features
features_for_clustering = pd.get_dummies(features_for_clustering, columns=categorical_cols, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_clustering)

# --- Clustering (K-Means) ---

# Determine optimal number of clusters (e.g., using Elbow Method - for demonstration, we'll use 5 as requested)
# For a real project, you'd typically run a loop and plot inertia

# Apply KMeans with 5 clusters
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init to suppress warning
df["cluster"] = kmeans.fit_predict(features_scaled)

# --- Analyze Segments ---

# Calculate churn rate per segment
churn_rate_by_segment = df.groupby("cluster")["churned"].mean().reset_index()
churn_rate_by_segment["churn_rate"] = churn_rate_by_segment["churned"].apply(lambda x: f"{x:.2%}")
print("\nChurn Rate by Segment:")
print(churn_rate_by_segment[["cluster", "churn_rate"]])

# Describe each segment
segment_analysis = df.groupby("cluster").mean(numeric_only=True)
print("\nSegment Analysis (Mean values for numerical features):")
print(segment_analysis)

# Visualize key features by segment
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="cluster", y=col, data=df, palette="viridis")
    plt.title(f"Distribution of {col} by Cluster")
    plt.savefig(f"cluster_{col}_boxplot.png")
    plt.close()

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(x="cluster", hue=col, data=df, palette="viridis")
    plt.title(f"Distribution of {col} by Cluster")
    plt.savefig(f"cluster_{col}_countplot.png")
    plt.close()

print("User segmentation complete. Visualizations saved as PNG files.")

