
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("mobile_app_data.csv")

# Display basic information
print("Dataset Info:")
df.info()

print("\nDataset Head:")
print(df.head())

print("\nDataset Description:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Churn Rate
churn_rate = df["churned"].mean()
print(f"\nOverall Churn Rate: {churn_rate:.2%}")

# Churn rates across different segments
# Gender
plt.figure(figsize=(8, 6))
sns.barplot(x="gender", y="churned", data=df, palette="viridis")
plt.title("Churn Rate by Gender")
plt.ylabel("Churn Rate")
plt.savefig("churn_by_gender.png")
plt.close()

# Device Type
plt.figure(figsize=(8, 6))
sns.barplot(x="device_type", y="churned", data=df, palette="viridis")
plt.title("Churn Rate by Device Type")
plt.ylabel("Churn Rate")
plt.savefig("churn_by_device_type.png")
plt.close()

# Subscription Type
plt.figure(figsize=(8, 6))
sns.barplot(x="subscription_type", y="churned", data=df, palette="viridis")
plt.title("Churn Rate by Subscription Type")
plt.ylabel("Churn Rate")
plt.savefig("churn_by_subscription_type.png")
plt.close()

# Age Distribution and Churn
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="age", hue="churned", multiple="stack", bins=20, kde=True)
plt.title("Age Distribution and Churn")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("age_distribution_churn.png")
plt.close()

# Sessions per week vs Churn
plt.figure(figsize=(10, 6))
sns.boxplot(x="churned", y="sessions_per_week", data=df, palette="viridis")
plt.title("Sessions per Week vs Churn")
plt.xlabel("Churned (0=No, 1=Yes)")
plt.ylabel("Sessions per Week")
plt.savefig("sessions_per_week_churn.png")
plt.close()

# Total time spent vs Churn
plt.figure(figsize=(10, 6))
sns.boxplot(x="churned", y="total_time_spent", data=df, palette="viridis")
plt.title("Total Time Spent vs Churn")
plt.xlabel("Churned (0=No, 1=Yes)")
plt.ylabel("Total Time Spent (minutes)")
plt.savefig("total_time_spent_churn.png")
plt.close()

# Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.savefig("correlation_matrix.png")
plt.close()

print("EDA complete. Visualizations saved as PNG files.")


