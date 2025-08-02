import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_records=50000):
    np.random.seed(42)

    # User demographics
    age = np.random.randint(18, 70, num_records).astype(float)
    gender = np.random.choice([np.nan, 'Male', 'Female', 'Other'], num_records, p=[0.01, 0.48, 0.48, 0.03])
    location = np.random.choice([np.nan, 'North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania'], num_records, p=[0.01, 0.29, 0.24, 0.19, 0.1, 0.1, 0.07])
    device_type = np.random.choice(['iOS', 'Android'], num_records, p=[0.55, 0.45])

    # App usage metrics
    sessions_per_week = (np.random.poisson(5, num_records) + 1).astype(float)
    avg_session_duration = np.random.normal(15, 5, num_records)
    avg_session_duration[avg_session_duration < 1] = 1
    days_since_install = np.random.randint(1, 365, num_records).astype(float)
    features_used_count = np.random.randint(1, 10, num_records).astype(float)

    # Engagement data
    push_notifications_enabled = np.random.choice([True, False], num_records, p=[0.7, 0.3])
    last_activity_date = [datetime.now() - timedelta(days=int(d)) for d in np.random.exponential(30, num_records)]
    total_time_spent = np.random.normal(1000, 500, num_records)
    total_time_spent[total_time_spent < 0] = 0

    # Behavioral patterns
    weekend_usage_ratio = np.random.beta(2, 5, num_records)
    peak_usage_hour = np.random.randint(8, 23, num_records).astype(float)
    support_tickets_count = np.random.poisson(0.5, num_records).astype(float)

    # Revenue data
    subscription_type = np.random.choice(['Free', 'Premium', 'VIP'], num_records, p=[0.6, 0.3, 0.1])
    total_spent = np.random.exponential(50, num_records)
    total_spent[subscription_type == 'Free'] = 0
    total_spent[subscription_type == 'Premium'] = total_spent[subscription_type == 'Premium'] * 2
    total_spent[subscription_type == 'VIP'] = total_spent[subscription_type == 'VIP'] * 5
    
    # Initialize last_purchase_date with NaT for all, then fill for non-free users
    last_purchase_date_series = pd.Series([pd.NaT] * num_records)
    non_free_indices = np.where(subscription_type != 'Free')[0]
    last_purchase_date_series.iloc[non_free_indices] = [datetime.now() - timedelta(days=int(d)) for d in np.random.exponential(60, len(non_free_indices))]

    # Target variable: churned
    # Introduce correlations: lower engagement, higher support tickets, older last activity -> higher churn
    churn_probability = 1 / (1 + np.exp(
        - (0.05 * (30 - sessions_per_week)) 
        + (0.02 * (100 - avg_session_duration)) 
        + (0.01 * (days_since_install / 365)) 
        + (0.1 * (10 - features_used_count)) 
        + (0.3 * (1 - push_notifications_enabled)) 
        + (0.05 * ((pd.to_datetime(last_activity_date) - datetime.now()).days / 30)) 
        + (0.001 * (1000 - total_time_spent)) 
        + (0.5 * weekend_usage_ratio) 
        + (0.1 * support_tickets_count) 
        + (0.2 * (subscription_type == 'Free')) 
        - (0.01 * total_spent)
    ))
    churned = (np.random.rand(num_records) < churn_probability).astype(int)

    # Introduce missing values and edge cases
    for col_name, col_data in {
        'age': age,
        'sessions_per_week': sessions_per_week,
        'avg_session_duration': avg_session_duration,
        'features_used_count': features_used_count,
        'total_time_spent': total_time_spent,
        'weekend_usage_ratio': weekend_usage_ratio,
        'peak_usage_hour': peak_usage_hour,
        'support_tickets_count': support_tickets_count,
        'total_spent': total_spent
    }.items():
        num_missing = np.random.randint(0, int(num_records * 0.01))
        missing_indices = np.random.choice(num_records, num_missing, replace=False)
        col_data[missing_indices] = np.nan

    # For boolean column, introduce missing values differently
    push_notifications_enabled_series = pd.Series(push_notifications_enabled)
    num_missing = np.random.randint(0, int(num_records * 0.01))
    missing_indices = np.random.choice(num_records, num_missing, replace=False)
    push_notifications_enabled_series.iloc[missing_indices] = np.nan

    data = pd.DataFrame({
        'user_id': range(num_records),
        'age': age,
        'gender': gender,
        'location': location,
        'device_type': device_type,
        'sessions_per_week': sessions_per_week,
        'avg_session_duration': avg_session_duration,
        'days_since_install': days_since_install,
        'features_used_count': features_used_count,
        'push_notifications_enabled': push_notifications_enabled_series,
        'last_activity_date': last_activity_date,
        'total_time_spent': total_time_spent,
        'weekend_usage_ratio': weekend_usage_ratio,
        'peak_usage_hour': peak_usage_hour,
        'support_tickets_count': support_tickets_count,
        'subscription_type': subscription_type,
        'total_spent': total_spent,
        'last_purchase_date': last_purchase_date_series,
        'churned': churned
    })

    # Ensure 'churned' is 1 if last_activity_date is very old (e.g., > 90 days)
    data.loc[(datetime.now() - data['last_activity_date']).dt.days > 90, 'churned'] = 1

    # Ensure 'churned' is 0 if last_activity_date is very recent (e.g., < 7 days)
    data.loc[(datetime.now() - data['last_activity_date']).dt.days < 7, 'churned'] = 0

    return data

if __name__ == '__main__':
    df = generate_synthetic_data(num_records=50000)
    df.to_csv('mobile_app_data.csv', index=False)
    print('Synthetic data generated and saved to mobile_app_data.csv')
    print(f'Dataset shape: {df.shape}')
    print(f'Churn rate: {df["churned"].mean():.2%}')
    print(f'Missing values per column:')
    print(df.isnull().sum())

