# The Kaggle Archive

This repo is my personal archive for past and current challenges, where I clean up my old code, build proper pipelines, and write down what actually works (and what doesn't).

---

## Repository Structure

```tree
.
├── playground/                 # Playground series and Kaggle competitions
│   ├── S6E02-heart-disease/    # Heart Disease Prediction challenge
│   ├── S6E03-customer-churn/   # Customer Churn Prediction challenge
│   ├── S6E04-irrigation-need/  # Irrigation Need prediction challenge
│   └── S6E07-student-health-risk/ # Student Health Risk (modular GBDT & linear pipeline)
├── research/                   # Scientific and scholarly challenges & deep dives
├── templates/                  # Reusable project boilerplates and templates
│   ├── gbdt-template/          # GBDT setup (LightGBM, XGBoost, CatBoost, Optuna)
│   ├── linear-template/        # Baseline linear models and pipelines
│   ├── nlp-template/           # NLP boilerplates
│   └── nn-template/            # Neural networks / PyTorch setup
```

---

## Tracked Challenges

| Competition / Series | Focus / Domain | Approach / Models | Status |
| :--- | :--- | :--- | :--- |
| **[S6E02 - Heart Disease](./playground/S6E02-heart-disease/)** | Tabular / Healthcare | EDA, Baseline Classifiers | Completed (Legacy Code) |
| **[S6E03 - Customer Churn](./playground/S6E03-customer-churn/)** | Tabular / Classification | Feature Engineering, Ensembles | Completed (Legacy Code) |
| **[S6E04 - Irrigation Need](./playground/S6E04-irrigation-need/)** | Tabular / Environmental | Preprocessing & Modeling | Completed (Legacy Code) |
| **[S6E07 - Student Health Risk](./playground/S6E07-student-health-risk/)** | Tabular / Risk Assessment | GBDT (Optuna tuning, CV, Modular `src/`) | Completed |
| **[S6E09 - Electric Vehicle Purchases](./playground/S6E09-electric-vehicle-purchases/)** | Tabular / Classification | GBDT (Optuna tuning, CV, Modular `src/`) | Completed |