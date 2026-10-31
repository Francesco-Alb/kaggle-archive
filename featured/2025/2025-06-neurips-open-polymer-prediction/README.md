# NeurIPS - Open Polymer Prediction 2025

The goal of this competition is to predict real-world polymer properties from their chemical structures, represented as SMILES strings.

> **Archive status:** Competition concluded in 2025. The notebook and this README are retained as a legacy archive; the competition page remains the source for its rules and data.

---

## Challenge Overview

- **Competition:** [NeurIPS - Open Polymer Prediction 2025](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025)
- **Task:** Multi-target regression
- **Input:** Polymer structure as a SMILES string
- **Target Variables:** `Tg` (glass transition temperature), `FFV` (fractional free volume), `Tc` (thermal conductivity), `Density`, and `Rg` (radius of gyration)
- **Evaluation Metric:** Weighted Mean Absolute Error (wMAE) across the five target properties

The weighted error is computed across polymers and the properties available for each polymer:

$$
\mathrm{wMAE} = \frac{1}{|\mathcal{X}|}\sum_{x \in \mathcal{X}}\sum_{i \in I(x)} w_i\,\left|\hat{y}_i(x)-y_i(x)\right|
$$

Here, $w_i$ is the competition-defined weight for property $i$, and $\hat{y}_i(x)$ and $y_i(x)$ are its predicted and true values.

### Submission Format

The submission is a CSV with one row per `id` and a predicted value for each property:

```csv
id,Tg,FFV,Tc,Density,Rg
2112371,0.0,0.0,0.0,0.0,0.0
```

### Timeline

- **Start Date:** June 16, 2025
- **Entry and Team Merger Deadline:** September 8, 2025
- **Final Submission Deadline:** September 15, 2025

---

## Repository Structure

```text
2025-06-neurips-open-polymer-prediction/
├── neurips-polypred.ipynb  # Competition notebook
└── README.md
```