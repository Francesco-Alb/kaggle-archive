# Kaggle Playground Series - Season 6, Episode 9: Electric Vehicle Purchases Prediction

The objective of this challenge is to predict whether a customer will buy an electric vehicle (`Will_Buy_EV`).

---

## Challenge Overview

- **Competition:** Kaggle Playground Series S6E09 (`playground-series-s6e9`)
- **Task:** Binary Classification
- **Target Variable:** `Will_Buy_EV` (Values: probabilities between 0 and 1)
- **Evaluation Metric:** [Area Under the ROC Curve (ROC AUC)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

### Submission Format
For each `id` in the test set, you must predict a probability for the `Will_Buy_EV` variable. The submission file should look like this:
```csv
id,Will_Buy_EV
668665,0.2
668666,0.3
668667,0.2
etc.
```

### Timeline
- **Start Date:** September 1, 2026
- **Entry Deadline:** September 30, 2026
- **Team Merger Deadline:** September 30, 2026
- **Final Submission Deadline:** September 30, 2026  
*(All deadlines are at 11:59 PM UTC)*

---

## Repository Structure

```tree
S6E09-electric-vehicle-purchases/
├── gbdt/                       # Gradient Boosted Decision Trees track (LightGBM, XGBoost, CatBoost)
│   ├── evals/                  # Best parameters, Optuna results
│   ├── notebooks/              # Exploratory data analysis & pipeline notebooks
│   ├── predictions/            # OOF and test set predictions
│   ├── results/                # Visualizations and score summaries
│   ├── src/                    # Source code modules
│   │   ├── adversarial_validation/ # Adversarial validation scripts
│   │   ├── data_processing/    # Feature engineering and preprocessing
│   │   ├── eda/                # Exploratory data analysis scripts
│   │   ├── modelling/          # Training, tuning, and evaluation scripts
│   │   └── utils/              # Data setup and helper utilities
│   └── requirements.txt        # Track dependencies
```
