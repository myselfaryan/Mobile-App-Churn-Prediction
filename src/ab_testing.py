
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("mobile_app_data.csv")

# --- Preprocessing for A/B Testing ---
# Impute missing values in sessions_per_week before simulation
imputer_sessions = SimpleImputer(strategy="median")
df["sessions_per_week"] = imputer_sessions.fit_transform(df[["sessions_per_week"]])

# --- A/B Testing Simulation ---

# 1. Create control and test groups
# For simplicity, let\"s assume we randomly split the existing user base.
# In a real scenario, this would be done before the intervention.

df["group"] = np.random.choice(["control", "test"], size=len(df), p=[0.5, 0.5])

# Define retention interventions and their simulated effects
# Let\"s assume the \"churned\" column represents churn in the last 30 days.
# A retention strategy aims to reduce churn, so we\"ll simulate a reduction in churn probability.

interventions = {
    "personalized_notifications": {
        "description": "Personalized notifications based on user behavior.",
        "churn_reduction_factor": 0.10, # 10% reduction in churn probability
        "engagement_improvement": 0.05 # 5% increase in engagement metrics (e.g., sessions_per_week)
    },
    "improved_onboarding": {
        "description": "Enhanced onboarding experience for new users.",
        "churn_reduction_factor": 0.15, # 15% reduction
        "engagement_improvement": 0.08
    },
    "feature_recommendations": {
        "description": "AI-powered feature recommendations within the app.",
        "churn_reduction_factor": 0.08, # 8% reduction
        "engagement_improvement": 0.03
    }
}

results = []

for intervention_name, intervention_data in interventions.items():
    print(f"\nSimulating A/B test for: {intervention_name}")
    
    # Create a copy to avoid modifying the original DataFrame in the loop
    df_sim = df.copy()
    
    # Apply intervention effect to the test group
    test_group_mask = (df_sim["group"] == "test")
    
    # Simulate churn reduction: reduce the probability of churn for test group
    # For users in the test group who churned, randomly select some to be retained
    churned_in_test_group_mask = (df_sim["group"] == "test") & (df_sim["churned"] == 1)
    num_churned_in_test_group = churned_in_test_group_mask.sum()
    
    num_to_retain = int(num_churned_in_test_group * intervention_data["churn_reduction_factor"])
    
    # Get indices of churned users in the test group
    churned_test_indices = df_sim[churned_in_test_group_mask].index
    
    # Randomly select indices to change from churned (1) to retained (0)
    indices_to_retain = np.random.choice(churned_test_indices, num_to_retain, replace=False)
    df_sim.loc[indices_to_retain, "churned"] = 0

    # Simulate engagement improvement (e.g., increase sessions_per_week for test group)
    df_sim.loc[test_group_mask, "sessions_per_week"] = df_sim.loc[test_group_mask, "sessions_per_week"] * (1 + intervention_data["engagement_improvement"])
    
    # Ensure sessions_per_week remains integer or appropriate type after multiplication
    df_sim["sessions_per_week"] = df_sim["sessions_per_week"].round().astype(int)

    # Calculate churn rates for control and test groups
    control_churn_rate = df_sim[df_sim["group"] == "control"]["churned"].mean()
    test_churn_rate = df_sim[df_sim["group"] == "test"]["churned"].mean()

    # Calculate engagement metrics (e.g., average sessions per week)
    control_avg_sessions = df_sim[df_sim["group"] == "control"]["sessions_per_week"].mean()
    test_avg_sessions = df_sim[df_sim["group"] == "test"]["sessions_per_week"].mean()

    # Statistical significance (for churn rate using chi-squared test)
    # Create a contingency table for churned vs. group
    control_churned_count = df_sim[df_sim["group"] == "control"]["churned"].sum()
    control_retained_count = len(df_sim[df_sim["group"] == "control"]) - control_churned_count
    
    test_churned_count = df_sim[df_sim["group"] == "test"]["churned"].sum()
    test_retained_count = len(df_sim[df_sim["group"] == "test"]) - test_churned_count

    contingency_table = np.array([
        [control_retained_count, control_churned_count],
        [test_retained_count, test_churned_count]
    ])
    
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

    # Business impact (simplified: revenue saved by reduced churn)
    # Assuming each churned user costs a certain amount (e.g., average revenue per user)
    # This will be refined in the next phase.
    avg_revenue_per_user = df["total_spent"].mean() # Using total_spent as a proxy for ARPU
    churn_reduction_count = (control_churn_rate - test_churn_rate) * len(df_sim[df_sim["group"] == "test"])
    revenue_saved = churn_reduction_count * avg_revenue_per_user

    results.append({
        "intervention": intervention_name,
        "control_churn_rate": control_churn_rate,
        "test_churn_rate": test_churn_rate,
        "churn_reduction": control_churn_rate - test_churn_rate,
        "control_avg_sessions": control_avg_sessions,
        "test_avg_sessions": test_avg_sessions,
        "engagement_improvement": test_avg_sessions - control_avg_sessions,
        "p_value": p_value,
        "statistical_significance": "Yes" if p_value < 0.05 else "No",
        "revenue_saved": revenue_saved
    })

results_df = pd.DataFrame(results)
print("\nA/B Testing Simulation Results:")
print(results_df.to_string())

results_df.to_csv("ab_testing_results.csv", index=False)
print("\nA/B testing simulation complete. Results saved to ab_testing_results.csv")


