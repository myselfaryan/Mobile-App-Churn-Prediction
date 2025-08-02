
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("mobile_app_data.csv")

# --- Feature Engineering ---

# Convert last_activity_date and last_purchase_date to datetime objects
df["last_activity_date"] = pd.to_datetime(df["last_activity_date"])
df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])

# Recency: Days since last activity
df["recency"] = (pd.to_datetime("today") - df["last_activity_date"]).dt.days

# Frequency: Sessions per week (already available, but can be combined with other features)
# Monetary: Total spent (already available)

# Handle missing values
# Numerical columns: Impute with median
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
# Exclude 'user_id' and 'churned' from imputation if they are numerical
if 'user_id' in numerical_cols: numerical_cols.remove('user_id')
if 'churned' in numerical_cols: numerical_cols.remove('churned')

imputer_numerical = SimpleImputer(strategy="median")
df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])

# Categorical columns: Impute with most frequent
categorical_cols = df.select_dtypes(include='object').columns.tolist()
imputer_categorical = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

# One-hot encode categorical features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop original date columns and user_id
df = df.drop(columns=["last_activity_date", "last_purchase_date", "user_id"])

# --- Model Building ---

X = df.drop("churned", axis=1)
y = df["churned"]

# Scale numerical features
scaler = StandardScaler()
X[X.select_dtypes(include=np.number).columns] = scaler.fit_transform(X[X.select_dtypes(include=np.number).columns])

# Train/Validation/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train) # 0.25 * 0.8 = 0.2 of original data

print(f"Train set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm
    }
    
    print(f"{name} Accuracy: {accuracy:.2f}")
    print(f"{name} Classification Report:\n{report}")
    print(f"{name} Confusion Matrix:\n{cm}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{name.replace(" ", "_")}_confusion_matrix.png')
    plt.close()

# Feature Importance for Random Forest (if available)
if "Random Forest" in models:
    rf_model = models["Random Forest"]
    if hasattr(rf_model, "feature_importances_"):
        feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importances.values, y=feature_importances.index)
        plt.title("Random Forest Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig("random_forest_feature_importance.png")
        plt.close()
        print("\nRandom Forest Feature Importance saved.")

print("\nMachine learning model training and evaluation complete.")


