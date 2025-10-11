# Netflix-Customer-Churn
An exploratory data analysis to understand customer churn pattern using R

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis(EDA)](#eda)
- [Hypothesis Testing](#hypothesis testing)
- [Results](#results)
- [Authors](#authors)

# Overview

This project aims to analyze customer churn using monthly subscription data. It includes EDA, data visualization, hypothesis testing and logistic regression modeling in R.

# Dataset

The dataset contains:
•	customer_id: Unique identifier for each customer.
•	gender: Gender of the customer.
•	age: Age in years.
•	monthly_fee: Subscription fee paid by the customer.
•	watch_time: Average monthly watch time (in hours).
•	plan_type: Type of subscription plan.
•	signup_date: Date of registration.
•	last_active_date: Most recent date of activity.
•	churned: Binary variable indicating whether the customer has churned (1 = Yes, 0 = No).

*(Dataset is fictional and used for demonstration purposes.)*

## Exploratory Data Analysis(EDA)

•Encoded categorical variables using:
•	Label Encoding (for binary/ordinal variables)
•	One-Hot Encoding (for nominal variables)
•	Analyzed distributions of numerical variables using histograms and boxplots.
•	Explored relationships between variables using:
•	Bar plots for categorical variables (e.g., churn rate by subscription type)
•	Boxplots (e.g., last login days vs churn status)
•	Correlation heatmaps (for numeric variables)
•	Identified outliers and trends in user behavior (e.g., higher churn with low login activity)
•	Detected potential features for modeling based on visual insights

## Hypothesis Testing

- **T-test**: Compared `tenure` between churned and non-churned users. Found statistically significant difference (p < 0.05).
- **Chi-squared Test**: Examined relationship between `plan_type` and churn. Result suggests dependency (p < 0.01).
- **Correlation Test**: Pearson correlation between `monthly_fee` and churn was weak and not statistically significant.


## Results

•	Categorical variables were encoded using label encoding or one-hot encoding where required.
•	Outliers were detected using boxplots and treated accordingly.
•	Churn Rate: Approximately 50.3% of customers were identified as churned.
•	Gender Distribution: Balanced representation across genders.
•	Plan Type vs Churn: Basic plans had a higher churn rate compared to premium plans.
•	Watch Time and Churn: Customers with lower average watch times were more likely to churn.
•	Monthly Fee: Churned customers generally subscribed to lower-priced plans.

