# Mobile App Churn Prediction & Retention Strategy

A comprehensive data science project demonstrating end-to-end machine learning pipeline for predicting user churn in mobile applications and developing data-driven retention strategies.

## Project Overview

This project addresses one of the most critical challenges in mobile app business: user retention. With mobile apps typically losing 70-80% of their users within the first 90 days, predicting and preventing churn is essential for sustainable growth and profitability.

### Key Features

- **Synthetic Dataset Generation**: Created a realistic dataset of 50,000 user records with logical correlations
- **Exploratory Data Analysis**: Comprehensive analysis of user behavior patterns and churn factors
- **Machine Learning Models**: Implemented and compared Logistic Regression, Random Forest, and XGBoost
- **User Segmentation**: K-means clustering to identify distinct user personas
- **A/B Testing Simulation**: Tested three retention strategies with statistical significance analysis
- **Business Impact Analysis**: Calculated ROI, CLV, ARPU, and revenue projections
- **Interactive Dashboard**: Professional visualizations for stakeholder presentations

### Business Impact

- **Churn Prediction Accuracy**: Achieved 82% accuracy with Random Forest model
- **Revenue Impact**: Projected $101,538 revenue savings through improved retention
- **User Segmentation**: Identified 5 distinct user segments with varying churn rates (27.8% - 29.4%)
- **A/B Testing Results**: Improved onboarding showed 3.73% churn reduction

## Project Structure

```
mobile-app-churn-prediction/
├── data/
│   ├── mobile_app_data.csv          # Synthetic dataset
│   ├── ab_testing_results.csv       # A/B testing simulation results
│   └── business_metrics.csv         # Business impact calculations
├── notebooks/
│   ├── 01_data_generation.ipynb     # Data creation process
│   ├── 02_exploratory_analysis.ipynb # EDA and visualizations
│   ├── 03_machine_learning.ipynb    # Model training and evaluation
│   ├── 04_user_segmentation.ipynb   # Clustering analysis
│   └── 05_business_analysis.ipynb   # ROI and impact calculations
├── src/
│   ├── generate_data.py             # Synthetic data generation
│   ├── eda.py                       # Exploratory data analysis
│   ├── ml_model.py                  # Machine learning pipeline
│   ├── user_segmentation.py         # Clustering implementation
│   ├── ab_testing.py                # A/B testing simulation
│   ├── business_impact.py           # Business metrics calculation
│   └── dashboard_viz.py             # Dashboard visualizations
├── visualizations/
│   ├── eda_charts/                  # Exploratory analysis charts
│   ├── model_performance/           # ML model evaluation plots
│   ├── segmentation/                # User segment visualizations
│   └── dashboard/                   # Executive dashboard charts
├── presentation/
│   └── Mobile_App_Churn_Strategy.html # Business presentation
├── docs/
│   ├── methodology.md               # Technical methodology
│   ├── data_dictionary.md           # Feature descriptions
│   └── results_summary.md           # Key findings summary
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/mobile-app-churn-prediction.git
   cd mobile-app-churn-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate synthetic data**
   ```bash
   python src/generate_data.py
   ```

5. **Run the analysis pipeline**
   ```bash
   python src/eda.py
   python src/ml_model.py
   python src/user_segmentation.py
   python src/ab_testing.py
   python src/business_impact.py
   python src/dashboard_viz.py
   ```

## Usage

### Running Individual Components

**Data Generation**
```bash
python src/generate_data.py
```
Generates a synthetic dataset with 50,000 user records including demographics, usage patterns, and churn labels.

**Exploratory Data Analysis**
```bash
python src/eda.py
```
Creates visualizations showing churn rates across different segments and identifies key patterns.

**Machine Learning Pipeline**
```bash
python src/ml_model.py
```
Trains and evaluates three classification models, generates confusion matrices and feature importance plots.

**User Segmentation**
```bash
python src/user_segmentation.py
```
Performs K-means clustering to identify user segments and analyzes their characteristics.

**A/B Testing Simulation**
```bash
python src/ab_testing.py
```
Simulates A/B tests for three retention strategies and calculates statistical significance.

**Business Impact Analysis**
```bash
python src/business_impact.py
```
Calculates key business metrics including ARPU, CLV, ROI, and revenue projections.

### Jupyter Notebooks

For interactive analysis, use the provided Jupyter notebooks:

```bash
jupyter notebook notebooks/
```

Each notebook corresponds to a specific phase of the analysis and includes detailed explanations and visualizations.

## Key Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 79% | 0.73 | 0.45 | 0.56 |
| Random Forest | 82% | 0.81 | 0.49 | 0.61 |
| XGBoost | 82% | 0.76 | 0.53 | 0.62 |

### User Segments

| Segment | Size | Churn Rate | Key Characteristics |
|---------|------|------------|-------------------|
| Segment 0 | 20% | 29.12% | Mixed subscription types, average engagement |
| Segment 1 | 20% | 28.99% | Higher premium users, good retention |
| Segment 2 | 20% | 29.01% | Younger users, mobile-first behavior |
| Segment 3 | 20% | 27.83% | **Best retention**, high engagement |
| Segment 4 | 20% | 29.44% | Highest churn risk, low engagement |

### A/B Testing Results

| Intervention | Churn Reduction | Engagement Improvement | Statistical Significance |
|--------------|----------------|----------------------|------------------------|
| Personalized Notifications | 2.28% | +0.006 sessions/week | p < 0.001 |
| **Improved Onboarding** | **3.73%** | **+0.35 sessions/week** | **p < 0.001** |
| Feature Recommendations | 1.69% | -0.026 sessions/week | p < 0.001 |

### Business Metrics

- **Overall ARPU**: $54.43
- **Customer Lifetime Value**: $189.25
- **Monthly Revenue Projection**: $1.43M
- **Annual Revenue Projection**: $17.15M
- **Revenue Impact (Best Strategy)**: $101,538

## Technical Methodology

### Data Generation

The synthetic dataset was created using Python's NumPy library with realistic correlations between features and churn behavior. Key considerations included:

- **Demographic Features**: Age, gender, location, device type
- **Usage Metrics**: Sessions per week, session duration, features used
- **Engagement Data**: Push notifications, activity recency, total time spent
- **Behavioral Patterns**: Weekend usage, peak hours, support tickets
- **Revenue Data**: Subscription type, total spent, purchase history

### Feature Engineering

- **Recency**: Days since last activity (strong churn predictor)
- **Frequency**: Sessions per week and usage patterns
- **Monetary**: Total spending and subscription value
- **Missing Value Imputation**: Median for numerical, mode for categorical
- **Scaling**: StandardScaler for numerical features
- **Encoding**: One-hot encoding for categorical variables

### Model Selection

Three classification algorithms were evaluated:

1. **Logistic Regression**: Baseline linear model for interpretability
2. **Random Forest**: Ensemble method with feature importance
3. **XGBoost**: Gradient boosting for optimal performance

Models were evaluated using 80/20 train-test split with stratification to maintain class balance.

### Clustering Methodology

K-means clustering with k=5 was selected based on:
- Elbow method analysis
- Business interpretability
- Segment size balance
- Actionable insights potential

## Portfolio Highlights

This project demonstrates proficiency in:

### Technical Skills
- **Python Programming**: Pandas, NumPy, Scikit-learn, XGBoost
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Classification, clustering, feature engineering
- **Statistical Analysis**: Hypothesis testing, A/B testing, significance testing
- **Business Analytics**: ROI calculation, CLV modeling, revenue forecasting

### Business Acumen
- **Problem Definition**: Identifying key business challenges
- **Stakeholder Communication**: Executive-level presentations
- **Strategic Recommendations**: Actionable retention strategies
- **Impact Quantification**: Revenue and ROI calculations
- **Data-Driven Decision Making**: Evidence-based recommendations

### Project Management
- **End-to-End Pipeline**: From data generation to business recommendations
- **Documentation**: Comprehensive technical and business documentation
- **Reproducibility**: Clear setup instructions and modular code
- **Presentation Skills**: Professional slide deck for stakeholders

## Future Enhancements

### Technical Improvements
- **Real-Time Prediction**: Deploy model as REST API
- **Feature Store**: Implement feature pipeline for production
- **Model Monitoring**: Track model drift and performance degradation
- **Advanced Models**: Deep learning approaches (LSTM, Neural Networks)
- **Ensemble Methods**: Combine multiple models for better performance

### Business Applications
- **Personalization Engine**: Real-time content recommendations
- **Dynamic Pricing**: Subscription optimization based on churn risk
- **Customer Success**: Proactive outreach to at-risk users
- **Product Development**: Feature prioritization based on retention impact
- **Marketing Optimization**: Targeted campaigns for different segments

