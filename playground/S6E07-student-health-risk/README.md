# Kaggle Playground Series - Season 6, Episode 7: Student Health Risk Prediction

The objective of this challenge is to predict student health risk categories based on various lifestyle, demographic, and health-related features.

---

## Challenge Overview

- **Competition:** Kaggle Playground Series S6E07 (`playground-series-s6e7`)
- **Task:** Multiclass Classification
- **Target Variable:** `health_condition` (Classes: `at-risk`, `unhealthy`, `fit`)
- **Evaluation Metric:** [Balanced Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)

### Submission Format
For each `id` in the test set, you must predict the appropriate `health_condition` label. The submission file should look like this:
```csv
id,health_condition
690088,at-risk
690089,at-risk
690090,at-risk
etc.
```

### Timeline
- **Start Date:** July 1, 2026
- **Entry Deadline:** July 31, 2026
- **Team Merger Deadline:** July 31, 2026
- **Final Submission Deadline:** July 31, 2026  
*(All deadlines are at 11:59 PM UTC)*

---

## Repository Structure

```tree
S6E07-student-health-risk/
├── gbdt/                       # Gradient Boosted Decision Trees track (LightGBM, XGBoost, CatBoost)
│   ├── evals/                  # Best parameters, Optuna results
│   ├── notebooks/              # Exploratory data analysis & pipeline notebooks
│   ├── predictions*/            # OOF and test set predictions
│   ├── results/                # Visualizations and score summaries
│   ├── src/                    # Source code modules
│   │   ├── adversarial_validation/ # Adversarial validation scripts
│   │   ├── data_processing/    # Feature engineering and preprocessing
│   │   ├── eda/                # Exploratory data analysis scripts
│   │   ├── modelling/          # Training, tuning, and evaluation scripts
│   │   └── utils/              # Data setup and helper utilities
│   ├── requirements.txt        # Track dependencies
│   └── snapshot.txt        # Track dependencies (full snapshot)
└── linear/                     # Linear models track / alternative approaches

* .gitignore
```