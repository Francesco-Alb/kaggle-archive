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
│   ├── S6E07-student-health-risk/ # Student Health Risk (modular GBDT & linear pipeline)
│   ├── S6E09-electric-vehicle-purchases/ # Electric Vehicle Purchases challenge
│   └── S6E10-airline-satisfaction/ # Airline Satisfaction challenge
├── featured/                   # Featured Kaggle competitions, grouped by year
│   └── 2025/
│       ├── 2025-06-cmi-detect-behavior-with-sensor-data/
│       ├── 2025-06-neurips-open-polymer-prediction/
│       ├── 2025-07-map-charting-student-math-misunderstandings/
│       └── 2017/
│           └── 2017-12-jigsaw-toxic-comment-classification-challenge/
├── research/                   # Scientific and scholarly challenges & deep dives
├── templates/                  # Reusable project boilerplates and templates
│   ├── gbdt-template/          # GBDT setup (LightGBM, XGBoost, CatBoost, Optuna)
│   ├── linear-template/        # Baseline linear models and pipelines
│   ├── nlp-template/           # NLP boilerplates
│   └── nn-template/            # Neural networks / PyTorch setup
```

---

## Playground Challenges

| Competition / Series | Focus / Domain | Approach / Models | Status |
| :--- | :--- | :--- | :--- |
| **[S6E02 - Heart Disease](./playground/S6E02-heart-disease/)** | Tabular / Healthcare | EDA, Baseline Classifiers | Archived (2026) |
| **[S6E03 - Customer Churn](./playground/S6E03-customer-churn/)** | Tabular / Classification | Feature Engineering, Ensembles | Archived (2026) |
| **[S6E04 - Irrigation Need](./playground/S6E04-irrigation-need/)** | Tabular / Environmental | Preprocessing & Modeling | Archived (2026) |
| **[S6E07 - Student Health Risk](./playground/S6E07-student-health-risk/)** | Tabular / Risk Assessment | GBDT (Optuna tuning, CV, Modular `src/`) | Archived (2026) |
| **[S6E09 - Electric Vehicle Purchases](./playground/S6E09-electric-vehicle-purchases/)** | Tabular / Classification | GBDT (Optuna tuning, CV, Modular `src/`) | Archived (2026) |
| **[S6E10 - Airline Satisfaction](./playground/S6E10-airline-satisfaction/)** | Tabular / Classification | GBDT (Optuna tuning, CV, Modular `src/`) | In Progress |

## Featured Competitions

| Competition | Focus / Domain | Approach / Models | Status |
| :--- | :--- | :--- | :--- |
| **[Jigsaw - Toxic Comment Classification](./featured/2025/2017/2017-12-jigsaw-toxic-comment-classification-challenge/)** | NLP / Multi-label text classification | TF-IDF features; multi-label classifiers | Archived (2024) |
| **[CMI - Detect Behavior with Sensor Data](./featured/2025/2025-06-cmi-detect-behavior-with-sensor-data/)** | Wearables / Time Series | Sensor-based gesture classification | Archived (2025) |
| **[NeurIPS - Open Polymer Prediction](./featured/2025/2025-06-neurips-open-polymer-prediction/)** | Chemistry / Materials Science | Polymer property regression | Archived (2025) |
| **[MAP - Charting Student Math Misunderstandings](./featured/2025/2025-07-map-charting-student-math-misunderstandings/)** | Education / NLP | DeBERTa fine-tuning; curriculum learning | Archived (2025) |