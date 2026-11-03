# Name That LLM

Identify which of 18 large language models generated each short text passage. The competition explores authorship attribution: distinguishing models by their vocabulary, phrasing, and style.

> **Competition status:** In progress as of November 3, 2026.

---

## Challenge Overview

- **Competition:** [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)
- **Task:** Multiclass text classification
- **Input:** Short text passages written in response to varied prompts
- **Target Variable:** `model` — the authoring language model
- **Number of Classes:** 18
- **Evaluation Metric:** Balanced accuracy, the average recall across the 18 model classes

The score is calculated as:

$$
\mathrm{BalancedAccuracy}
= \frac{1}{K}\sum_{k=1}^{K}
\frac{\mathrm{TP}_k}{\mathrm{TP}_k+\mathrm{FN}_k},
\qquad K=18
$$

where $\mathrm{TP}_k$ and $\mathrm{FN}_k$ are the true positives and false negatives for class $k$. This gives each model class equal weight.

### Submission Format

Submissions must be made through a Kaggle Notebook that writes `submission.parquet`. Include one prediction for every test `id`; the required columns are `id` and `model`.

```python
submission = pd.DataFrame({"id": test["id"], "model": predictions})
submission.to_parquet("submission.parquet", index=False)