# CMI - Detect Behavior with Sensor Data

This competition asks whether wrist-worn motion, temperature, and proximity sensors can identify body-focused repetitive behavior (BFRB)-like gestures among everyday movements.

> **Archive status:** Competition concluded in 2025. This folder is a legacy archive; the competition page remains the source for its rules and data.

---

## Challenge Overview

- **Competition:** [CMI - Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data)
- **Task:** Multiclass gesture classification, including BFRB-like and non-BFRB-like gestures
- **Target Variable:** `gesture`
- **Input:** Sequential sensor readings from a wrist-worn device (IMU, thermopiles, and time-of-flight sensors)
- **Evaluation Metric:** Equal-weight average of binary F1 for target vs. non-target gestures and macro F1 across gesture classes, with non-target gestures collapsed into one class for the latter

The competition score is:

$$
\mathrm{Score} = \frac{\mathrm{F1}_{\mathrm{binary}} + \mathrm{F1}_{\mathrm{macro}}}{2}
$$

### Submission

Predictions are made for each `sequence_id` through Kaggle's evaluation API. The competition evaluates sequences with different sensor subsets, including IMU-only data.

### Timeline

- **Start Date:** May 29, 2025
- **Entry and Team Merger Deadline:** August 26, 2025
- **Final Submission Deadline:** September 2, 2025

---

## Repository Structure

```text
2025-06-cmi-detect-behavior-with-sensor-data/
└── README.md  # Competition notes and link
```