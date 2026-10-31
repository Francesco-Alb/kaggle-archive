# Jigsaw Toxic Comment Classification Challenge

This challenge focused on identifying different types of toxicity in Wikipedia comments, including insults, threats, and identity-based hate. The archived notebook explores text-based multi-label classification.

> **Archive status:** This challenge attempt was closed and archived in 2024. The Kaggle competition ran from 2017 to 2018; its page remains the source for the original data and rules.

---

## Challenge Overview

- **Competition:** [Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
- **Task:** Multi-label text classification
- **Input:** Wikipedia talk-page comment text (`comment_text`)
- **Target Variables:** `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`
- **Evaluation Metric:** Mean column-wise ROC AUC, averaged over the six target labels

The score is the mean of the six per-label ROC AUC values:

$$
\mathrm{Score} = \frac{1}{6}\sum_{j=1}^{6}\mathrm{AUC}(y_j,\hat{p}_j)
$$

Here, $y_j$ contains the true binary labels for toxicity type $j$, and $\hat{p}_j$ contains the corresponding predicted probabilities.

### Submission Format

For each `id`, submit a probability for each target label:

```csv
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
```

### Timeline

- **Competition Start Date:** December 19, 2017
- **Entry and Team Merger Deadline:** March 13, 2018
- **Final Submission Deadline:** March 20, 2018

---

## Repository Structure

```text
2017-12-jigsaw-toxic-comment-classification-challenge/
├── ml-competitions-toxic-comment-classification.ipynb  # EDA and multi-label modeling experiments
└── README.md
```