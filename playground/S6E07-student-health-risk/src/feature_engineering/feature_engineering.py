import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def consensus_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Fits LightGBM, XGBoost, and CatBoost models on the training set to calculate
    a normalized consensus feature importance.

    This function trains each of the three gradient boosting frameworks, extracts
    their feature importances, normalizes them individually so they sum to 1.0, and 
    then averages them to produce a stable, robust consensus ranking of features.

    Args:
        X: Training features.
        y: Target labels.
        device: Hardware device to use for training (e.g., "cpu", "cuda", "gpu").
            Defaults to "cpu".
        seed: Random seed for reproducibility across model fits. Defaults to 42.
        verbose: If True, prints progress updates and model performance metrics.
            Defaults to True.
        plot: If True, renders a bar plot displaying the top ranked consensus 
            features. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame sorted in descending order of consensus 
            importance, containing columns:
            - 'feature': Name of each feature.
            - 'lgb_importance': Normalized LightGBM feature importance.
            - 'xgb_importance': Normalized XGBoost feature importance.
            - 'cat_importance': Normalized CatBoost feature importance.
            - 'consensus_importance': Mean of the three normalized importances.
    """

    X_temp = X.copy()

    # ------------------------------------------------------------------ Prepare Categorical Columns
    categorical_cols = X_temp.select_dtypes(["category", "object"]).columns.tolist()

    for col in categorical_cols:
        X_temp[col] = X_temp[col].astype('category')

    # ------------------------------------------------------------------ Prepare "light" models for speed
    importance_models = {
        "XGB": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            device=device,
            random_state=seed,
            enable_categorical=True
        ),
        "LGBM": LGBMClassifier(
            n_estimators=500,
            num_leaves=30,
            device=device,
            random_state=seed
        ),
        "CAT": CatBoostClassifier(
            iterations=500,
            depth=6,
            verbose=0,
            task_type=device.upper(),
            random_state=seed
        )
    }

    # ------------------------------------------------------------------ Get feature importance
    importance_results = pd.DataFrame(index=X_temp.columns)

    if verbose:
        print("=== Running Consensus Importance Pass ===")

    for name, model in importance_models.items():
        if verbose:
            print(f"Fitting {name}...")

        if name == "CAT":
            model.fit(X_temp, y, cat_features=categorical_cols)
            importance_results[name] = model.get_feature_importance()
        elif name == "LGBM":
            model.fit(X_temp, y, categorical_feature=categorical_cols)
            importance_results[name] = model.feature_importances_
        else: # XGB
            model.fit(X_temp, y)
            importance_results[name] = model.feature_importances_

        # Normalize scores to 0-100 to make them comparable
        importance_results[name] = (importance_results[name] / importance_results[name].sum()) * 100

    # Calculate Global Score and Perform Selection
    importance_results['mean_importance'] = importance_results.mean(axis=1)
    importance_results = importance_results.sort_values('mean_importance', ascending=False)

    # ------------------------------------------------------------------ Plot
    if plot:
        plt.figure(figsize=(15, 8))
        sns.barplot(
            data=importance_results,
            y=importance_results.index,
            x='mean_importance'
        )
        plt.title('Consensus Feature Importance', fontsize=16, weight='bold')
        plt.xlabel('Mean Normalized Importance (%)', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.tight_layout()
        plt.show()

    return importance_results


def interact(df, numeric_features):
    temp_df = df.copy()

    other_cols = [col for col in temp_df.columns if col not in numeric_features]

    interactions = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interactions = interactions.fit_transform(temp_df[numeric_features])

    interaction_names = interactions.get_feature_names_out(numeric_features)
    interaction_names = [name.replace(" ", "_X_") for name in interaction_names]

    interactions_df = pd.DataFrame(X_interactions, columns=interaction_names, index=temp_df.index)

    return pd.concat([temp_df[other_cols], interactions_df], axis=1)


def preprocessing(df, numeric_features):
    temp_df = df.copy()
    temp_df = interact(temp_df, numeric_features)
    return temp_df
