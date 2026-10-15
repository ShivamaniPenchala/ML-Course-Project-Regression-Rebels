# ML Course Project - Regression Rebels

Our Machine Learning course project (M.Tech CSE). There were two Kaggle competitions, one classification and one regression.

## Folders

```
classification/
    machine_condition_classification.ipynb   -> EDA + models + submission
regression/
    flight_price_eda.ipynb                    -> EDA
    flight_price_models.ipynb                 -> features, models, tuning, submission
requirements.txt
```

## Part A - Machine Condition Classification

Predict if a machine is Normal (0), has a Thermal Failure (1) or a Mechanical Failure (2) from its sensor readings. Metric is accuracy.

What we did:
- filled missing values with the mean, mapped unit_type S/M/H to 0/1/2
- EDA (distributions, outliers, features vs class, correlation, ANOVA)
- tried Logistic Regression, Naive Bayes, Decision Tree, Random Forest, XGBoost
- added 2 features: `wear_torque` and `stress_per_flow`
- tuned XGBoost with GridSearchCV

Final model: XGBoost (300 trees, depth 3, lr 0.1)

| | score |
|---|---|
| CV accuracy | 0.8877 |
| Kaggle public leaderboard | 0.89000 |

## Part B - Flight Price Prediction

Predict the ticket price of a flight. Metric is R2.

What we did:
- log of price as target (price is very skewed)
- removed the 25 zero price rows from training
- label / frequency encoding + target encoding (done out-of-fold)
- found that `id` follows the order the data was scraped in, and prices are sorted inside blocks, so we made features from neighbouring prices
- tried Ridge, KNN, Decision Tree, Random Forest, XGBoost
- tuned XGBoost with optuna and random forest with a small grid
- mixture model: a classifier decides if the ticket is expensive (> 1000), then XGBoost for normal prices and Random Forest for expensive ones
- also tried stacking and blending

Final model: MIX_T1000 (the mixture model)

| | score |
|---|---|
| CV R2 | 0.8109 |
| Kaggle public leaderboard R2 | 0.86440 |

## How to run

Run the notebooks on Kaggle with the competition data added (paths are the Kaggle input paths).
Locally, install the libraries and change the `path` variable at the top of each notebook.

```
pip install -r requirements.txt
```

Random seed is 42 everywhere. The regression models notebook takes a while to run (random forests + optuna).
