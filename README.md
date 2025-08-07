# Netflix-Customer-Churn
An exploratory data analysis to understand customer churn pattern using R

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Authors](#authors)

# Overview

This project aims to analyze customer churn using monthly subscription data. It includes EDA, data visualization, and logistic regression modeling in R.

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

## Installation

Make sure you have R and RStudio installed.

Install required R packages:
```r
install.packages(c("ggplot2", "dplyr", "caTools", car))
```

## Project Structure

