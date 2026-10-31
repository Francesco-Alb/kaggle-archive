# MAP - Charting Student Math Misunderstandings

This competition focuses on identifying potential mathematical misconceptions from students' open-ended explanations, helping educators interpret incorrect reasoning.

> **Archive status:** Competition concluded in 2025. The notebooks and this README are retained as a legacy archive; the competition page remains the source for its rules and data.

---

## Challenge Overview

- **Competition:** [MAP - Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)
- **Task:** NLP classification and ranked label prediction
- **Input:** Student responses to mathematics questions
- **Target:** `Category:Misconception` labels for each `row_id`
- **Evaluation Metric:** Mean Average Precision at 3 (MAP@3)

For $N$ rows, the metric is the mean of the per-row average precision over the top three predictions:

$$
\mathrm{MAP@3} = \frac{1}{N}\sum_{u=1}^{N}\sum_{k=1}^{\min(3,n_u)} P_u(k)\,\mathrm{rel}_u(k)
$$

Here, $P_u(k)$ is precision at rank $k$, and $\mathrm{rel}_u(k)$ indicates whether the prediction at that rank is correct. A correct label is counted only once per row.

### Submission Format

For each `row_id`, submit up to three space-separated `Category:Misconception` predictions in the `Category:Misconception` column:

```csv
row_id,Category:Misconception
36696,True_Correct:NA False_Neither:NA False_Misconception:Incomplete
```

### Archived Notebooks

- `math-misconceptions-deberta-lora-w-weighted-loss.ipynb`: DeBERTa fine-tuning with LoRA and weighted loss
- `curriculum-style-fine-tuning-process.ipynb`: Curriculum-style fine-tuning workflow

### Timeline

- **Start Date:** July 10, 2025
- **Entry and Team Merger Deadline:** October 8, 2025
- **Final Submission Deadline:** October 15, 2025

---

## Repository Structure

```text
2025-07-map-charting-student-math-misunderstandings/
├── curriculum-style-fine-tuning-process.ipynb
├── math-misconceptions-deberta-lora-w-weighted-loss.ipynb
└── README.md
```