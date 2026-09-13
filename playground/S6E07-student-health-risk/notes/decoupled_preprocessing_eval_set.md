# Decoupled Preprocessing and eval_set in Cross-Validation Loops

## The Problem

When using gradient-boosted models (XGBoost, LightGBM, CatBoost) with validation-aware training (`eval_set` for early stopping) inside a cross-validation loop, wrapping the preprocessor and model together in a sklearn `Pipeline` causes issues:

- The pipeline applies preprocessing to training data
- But `eval_set` receives raw, untransformed validation data
- This creates a mismatch: the model trains on transformed features but validates on raw features
- Result: shape mismatches, feature mismatches, or cryptic errors

## The Solution: Decouple Preprocessor from Model

Instead of using a single Pipeline, fit the preprocessor and model as separate steps:

```python
# Per fold in your cross-validation loop
preprocessor = clone(preprocessing_pipeline)
model = clone(base_model)

# Step 1: Fit preprocessor on training data ONLY
X_train_transformed = preprocessor.fit_transform(X_train)
X_valid_transformed = preprocessor.transform(X_valid)
X_test_transformed = preprocessor.transform(X_test)

# Step 2: Fit model on transformed data with proper eval_set
model.fit(
    X_train_transformed, 
    y_train, 
    eval_set=[(X_valid_transformed, y_valid)],
    early_stopping_rounds=stopping_rounds,
    verbose=False
)

# Step 3: Predict using transformed data
predictions = model.predict(X_valid_transformed)
```

## Why This Works

1. **No data leakage**: The preprocessor is fitted only on training data. Validation and test sets use the learned parameters.
2. **Consistent feature space**: Both train and validation are transformed identically before reaching the model.
3. **Native eval_set support**: The model receives validation data in the correct, transformed format.
4. **Simpler API**: You pass arguments directly to the model instead of forcing them through a Pipeline wrapper.
5. **Per-fold preprocessing**: Each fold gets its own preprocessor fitted state, maintaining fold independence.

## Key Takeaway

**Separate concerns**: Use a Pipeline only when you need to bundle preprocessing + model for deployment or simple single-fit scenarios. In competition machine learning with cross-validation and early stopping, keep preprocessing separate and explicit. This gives you full control over data flow and prevents subtle bugs.

## When This Matters Most

- Tabular competitions with cross-validation folds
- Models with `eval_set` or `validation_data` parameters (XGBoost, LightGBM, CatBoost, Keras)
- Any scenario where you need train-only fitting of preprocessing steps
