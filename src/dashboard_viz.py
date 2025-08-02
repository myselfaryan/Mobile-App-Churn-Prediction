import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("mobile_app_data.csv")
ab_results = pd.read_csv("ab_testing_results.csv")
business_metrics = pd.read_csv("business_metrics.csv")

# --- Executive Summary with Key Metrics ---
# This would typically be a summary table or text, but for visualization, we can show key trends.

# Overall Churn Rate
plt.figure(figsize=(6, 4))
churn_rate = df["churned"].mean()
sns.barplot(x=["Churn Rate"], y=[churn_rate], palette="viridis")
plt.title("Overall Churn Rate")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.text(0, churn_rate + 0.02, f"{churn_rate:.2%}", ha="center")
plt.savefig("dashboard_overall_churn_rate.png")
plt.close()

# --- Churn Prediction Model Performance Metrics ---
# Assuming we have the model results from ml_model.py, we can visualize accuracy.
# For simplicity, let's just use the accuracies from the previous run.

model_accuracies = {
    "Logistic Regression": 0.79,
    "Random Forest": 0.82,
    "XGBoost": 0.82
}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()), palette="magma")
plt.title("Churn Prediction Model Accuracies")
plt.ylabel("Accuracy")
plt.ylim(0.7, 0.9)
for index, value in enumerate(model_accuracies.values()):
    plt.text(index, value + 0.005, f"{value:.2f}", ha="center")
plt.savefig("dashboard_model_accuracies.png")
plt.close()

# --- User Segmentation Visualizations ---
# Re-run segmentation to get clusters in the main dataframe for visualization
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Convert date columns to datetime
df["last_activity_date"] = pd.to_datetime(df["last_activity_date"])
df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])

# Select features for clustering (excluding user_id and churned)
features_for_clustering = df.drop(columns=["user_id", "churned", "last_activity_date", "last_purchase_date"])

# Handle missing values for clustering features
numerical_cols_clustering = features_for_clustering.select_dtypes(include=np.number).columns.tolist()
imputer_numerical_clustering = SimpleImputer(strategy="median")
features_for_clustering[numerical_cols_clustering] = imputer_numerical_clustering.fit_transform(features_for_clustering[numerical_cols_clustering])

categorical_cols_clustering = features_for_clustering.select_dtypes(include='object').columns.tolist()
imputer_categorical_clustering = SimpleImputer(strategy="most_frequent")
features_for_clustering[categorical_cols_clustering] = imputer_categorical_clustering.fit_transform(features_for_clustering[categorical_cols_clustering])

# One-hot encode categorical features
features_for_clustering = pd.get_dummies(features_for_clustering, columns=categorical_cols_clustering, drop_first=True)

# Scale numerical features
scaler_clustering = StandardScaler()
features_scaled_clustering = scaler_clustering.fit_transform(features_for_clustering)

k = 5
kmeans_clustering = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans_clustering.fit_predict(features_scaled_clustering)

# Churn Rate by Segment
churn_rate_by_segment = df.groupby("cluster")["churned"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x="cluster", y="churned", data=churn_rate_by_segment, palette="plasma")
plt.title("Churn Rate by User Segment")
plt.xlabel("Segment")
plt.ylabel("Churn Rate")
plt.ylim(0, 0.4)
for index, row in churn_rate_by_segment.iterrows():
    plt.text(row.name, row.churned + 0.01, f"{row.churned:.2%}", ha="center")
plt.savefig("dashboard_churn_by_segment.png")
plt.close()

# --- A/B Testing Results ---
plt.figure(figsize=(12, 7))
sns.barplot(x="intervention", y="churn_reduction", data=ab_results, palette="coolwarm")
plt.title("Churn Reduction by Intervention (A/B Test)")
plt.xlabel("Intervention")
plt.ylabel("Churn Reduction (Control - Test)")
plt.axhline(0, color="grey", linestyle="--")
for index, row in ab_results.iterrows():
    plt.text(index, row.churn_reduction + 0.005 if row.churn_reduction > 0 else row.churn_reduction - 0.005, f"{row.churn_reduction:.2%}", ha="center")
plt.savefig("dashboard_ab_test_churn_reduction.png")
plt.close()

# --- Business Impact Charts ---
# Revenue Impact of Churn Reduction
revenue_impact = business_metrics[business_metrics["Metric"] == "Revenue Impact (Best Intervention)"]["Value"].iloc[0]
revenue_impact_val = float(revenue_impact.replace("$", "").replace(",", ""))

plt.figure(figsize=(6, 4))
sns.barplot(x=["Revenue Impact"], y=[revenue_impact_val], palette="rocket")
plt.title("Estimated Revenue Impact of Best Retention Strategy")
plt.ylabel("Revenue ($)")
plt.text(0, revenue_impact_val + 5000, f"${revenue_impact_val:,.2f}", ha="center")
plt.savefig("dashboard_revenue_impact.png")
plt.close()

# ROI for Best Intervention
roi_value = business_metrics[business_metrics["Metric"] == "ROI (Best Intervention)"]["Value"].iloc[0]
roi_val = float(roi_value.replace("%", ""))

plt.figure(figsize=(6, 4))
sns.barplot(x=["ROI"], y=[roi_val], palette="rocket")
plt.title("ROI for Best Retention Strategy")
plt.ylabel("ROI (%)")
plt.text(0, roi_val + 1, f"{roi_val:.2f}%", ha="center")
plt.savefig("dashboard_roi.png")
plt.close()

print("Dashboard visualizations generated and saved as PNG files.")

