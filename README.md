
# Marathon Finish Time Prediction

A machine learning analysis exploring whether marathon finish times can be predicted from runner training data, physiological metrics, and race-day conditions.

## Overview

This project applies supervised regression across three model families — K-Nearest Neighbours, Decision Trees, and Random Forest — to a dataset of 20,000 marathon runners. The full workflow covers data cleaning, exploratory visualisation, feature exploration, hyperparameter tuning, and statistical model comparison.

## Dataset

The dataset contains 39 features per runner across four broad categories:

| Category | Features |
|---|---|
| Training load | Weekly mileage, runs per week, long run distance, speed work sessions, rest days |
| Physiology | VO₂ max, BMI, resting heart rate, recovery score, sleep hours |
| Behaviour & habits | Nutrition score, hydration consistency, warmup/stretching adherence, training streak |
| Race context | Course difficulty, marathon weather, mental preparation score |

The target variable is **Actual Finish Time (minutes)**. `target_finish_time_minutes` was excluded as a feature — a runner's stated goal time is not information a model should use to predict their actual finish time.

## Notebook Structure


1. **Data Cleaning** — handling missing values, outlier detection, checking categorical values, dropping redundant columns
2. **Exploratory Data Analysis** — visualisations of key feature distributions and their relationship to finish time, broken down by gender
3. **Feature Exploration** — Sequential Feature Selection using Linear Regression to identify the most informative features across varying feature counts
4. **Ordinal/Label Encoding** — categorical variables encoded numerically to prepare for model fitting
5. **Model Fitting & Tuning** — GridSearchCV with 5-fold cross-validation for KNN, Decision Tree, and Random Forest; commented param grids show the search ranges explored
6. **Feature Importance** — Random Forest feature importances extracted to identify the strongest predictors
7. **Residual Plots** — predicted vs. residual plots for all three models on held-out test data
8. **Model Comparison** — paired t-tests on cross-validated R² scores across all three models


## Results

All three pairwise model comparisons were statistically significant (p < 0.001), with Random Forest outperforming both KNN and Decision Tree:

| Comparison | T-statistic | p-value |
|---|---|---|
| KNN vs. Decision Tree | −15.78 | 9.4 × 10⁻⁵ |
| KNN vs. Random Forest | −27.40 | 1.1 × 10⁻⁵ |
| Decision Tree vs. Random Forest | −23.23 | 2.0 × 10⁻⁵ |

Negative T-statistics indicate the second model in each pair achieved higher R² scores.

## Discussion

Random Forest was the clear winner across all comparisons. As an ensemble method, it reduces the variance that causes a single Decision Tree to overfit, and handles the high feature count better than KNN, which degrades as dimensionality grows. The feature importance output from the Random Forest gives an additional layer of interpretability — identifying which training and physiological factors most strongly predict race day performance.

The statistical significance of the comparisons is encouraging, but the results should be interpreted with some caution. The models were trained on a sample of 10,000 runners drawn from the full dataset; generalisation to runners outside the training distribution (e.g. elites, first-timers) would require further validation.

## Requirements

```
scikit-learn
pandas
numpy
matplotlib
seaborn
scipy
```