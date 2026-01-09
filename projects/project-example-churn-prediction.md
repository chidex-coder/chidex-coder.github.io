---
layout: default
title: Customer Churn Prediction
permalink: /projects/churn-prediction/
---

# Customer Churn Prediction

**Problem:** High customer churn rate was costing the company $500K annually.

**Solution:** Built a machine learning model to identify at-risk customers before they churn.

## Approach

1. **Data Collection:** Extracted 2 years of customer data (50K+ records)
2. **EDA:** Analyzed patterns in customer behavior, identified key features
3. **Modeling:** Tested Random Forest, XGBoost, and Logistic Regression
4. **Evaluation:** Achieved 89% precision on test set

## Results

- Model correctly identifies 89% of customers who will churn
- Reduced false positives by 40% compared to previous heuristic
- Enabled targeted retention campaigns

## Code Highlights

```python
# Feature engineering example
def create_features(df):
    df['days_since_active'] = (pd.Timestamp.now() - df['last_activity']).dt.days
    df['usage_trend'] = df['recent_usage'] / df['historical_usage']
    return df
