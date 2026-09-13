# Handling Missing Categorical Values in CatBoost, LightGBM, and XGBoost

## Lesson Learned
- **CatBoost constraint**: CatBoost requires categorical features to be strings or integers, and **disallows real number or `NaN` values** in columns specified as `cat_features`. It raises a `CatBoostError` if `NaN`s are encountered.
- **Imputation Strategy**: Rather than using mode/most-frequent imputation (which can distort distributions and erase signal), filling missing categorical values with an explicit placeholder string like `"Missing"` or `"Unknown"` before casting them to pandas `"category"` dtype is superior.
- **Model Compatibility**: This approach works seamlessly across CatBoost, LightGBM, and XGBoost without negatively interfering with how they handle missing categorical values natively.