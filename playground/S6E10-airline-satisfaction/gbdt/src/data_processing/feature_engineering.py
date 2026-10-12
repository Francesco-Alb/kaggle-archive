import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Union

from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin

# TODO: push to a different module (like feature_selection) 
# or drop entirely and rely on sklearn feature_selection in a Pipeline

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

# STATEFUL
class IQRCapper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    # NOTE: needed for to avoid errors when using .set_output(transform = 'pandas')
    def set_output(self, transform=None):
        return self

    def fit(self, X, y=None):
        self.columns_ = (
            self.columns
            if self.columns is not None
            else X.select_dtypes(include=np.number).columns.tolist()
        )

        self.bounds_ = {}

        for col in self.columns_:
            col_data = X[col].dropna()
            if len(col_data) == 0:
                continue
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1

            self.bounds_[col] = (
                q1 - 1.5 * iqr,
                q3 + 1.5 * iqr
            )

        return self

    def transform(self, X):
        X = X.copy()

        for col, (lower, upper) in self.bounds_.items():
            X[col] = X[col].clip(lower, upper)

        return X


# STATEFUL
class GroupAggregator(BaseEstimator, TransformerMixin):
    """
    Stateful transformer that computes group-level aggregations (mean, size, etc.) 
    of target numeric columns grouped by a key (anchor) column.
    Learns lookup statistics during .fit() on training data and applies them during .transform().
    """
    def __init__(self, anchor_col: str, target_cols: List[str], agg_funcs: List[str] = ['mean']):
        self.anchor_col = anchor_col
        self.target_cols = target_cols
        self.agg_funcs = agg_funcs
        self.mappings_ = {}
        self.global_means_ = {}

    def set_output(self, transform=None):
        return self

    def fit(self, X, y=None):
        X = pd.DataFrame(X.copy())
        self.mappings_ = {}
        self.global_means_ = {}

        for target in self.target_cols:
            if target not in X.columns or self.anchor_col not in X.columns:
                continue
            
            # Compute global fallback mean
            self.global_means_[target] = float(X[target].mean())
            
            # Group by anchor and compute requested aggregations
            grouped = X.groupby(self.anchor_col)[target]
            for agg in self.agg_funcs:
                if agg == 'mean':
                    mapping = grouped.mean().to_dict()
                elif agg == 'size':
                    mapping = grouped.size().to_dict()
                elif agg == 'std':
                    mapping = grouped.std().fillna(0).to_dict()
                else:
                    raise ValueError(f"Unsupported aggregation: {agg}")

                self.mappings_[f"{target}-{self.anchor_col}-{agg}"] = mapping

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for feat_name, mapping in self.mappings_.items():
            # Extract target name from feat_name (e.g., 'seat_comfort_route_mean' -> 'seat_comfort')
            target = feat_name.split('_route_')[0]
            fallback = self.global_means_.get(target, 0.0)
            
            # Map values onto anchor column, filling unseen keys with global fallback
            X[feat_name] = X[self.anchor_col].map(mapping).fillna(fallback).astype(np.float32)

        return X


# STATELESS
def interact(
        df: pd.DataFrame,
        degree: int = 2,
        interaction_features: Optional[Union[List[str], List[Tuple[str, str]]]] = None,
        interaction_only: bool = True,
        include_bias: bool = False,
        name_separator: str = "_X_",
    ) -> pd.DataFrame:
    """
    Computes interaction features.
    - If interaction_features is None, computes polynomial/interaction features on all numeric columns.
    - If interaction_features is a list of strings, computes interactions on that subset of numeric columns.
    - If interaction_features is a list of tuples (e.g. [('A', 'B')]), computes explicitly interactions between specified pairs.
    """
    temp_df = df.copy()

    if len(interaction_features) > 0 and isinstance(interaction_features[0], tuple):
        interaction_data = {}
        
        # Explicit pairs: (col_a, col_b)
        for pair in interaction_features:
            if len(pair) == 2:
                col_a, col_b = pair
                interaction_data[f"{col_a}{name_separator}{col_b}"] = temp_df[col_a] * temp_df[col_b]
        interactions_df = pd.DataFrame(interaction_data, index=temp_df.index)
        return pd.concat([temp_df, interactions_df], axis=1)

    else:
        interaction_features = interaction_features or temp_df.select_dtypes(include=np.number).columns.tolist()

        interactions = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        X_interactions = interactions.fit_transform(temp_df[interaction_features])
        
        interaction_names = interactions.get_feature_names_out(interaction_features)
        interaction_names = [name.replace(" ", name_separator) for name in interaction_names]
        
        interactions_df = pd.DataFrame(X_interactions, columns=interaction_names, index=temp_df.index)
        return pd.concat([temp_df, interactions_df], axis=1)


# STATELESS
def ratios(
        df: pd.DataFrame,
        ratio_features: Optional[Union[List[str], List[Tuple[str, str]]]] = None,
        name_separator: str = "_/_",
        epsilon: float = 1e-5,
    ) -> pd.DataFrame:
    """
    Computes ratio features. 
    - If ratio_features is a list of strings, computes all pairwise ratios (A/B and B/A).
    - If ratio_features is a list of tuples (e.g. [('A', 'B')]), computes explicitly A/B.
    """
    temp_df = df.copy()
    remaining_features = [col for col in temp_df.columns] # or filter out what's used
    ratio_data = {}

    # None: all combinations
    if ratio_features is None:
        cols = temp_df.select_dtypes(include=np.number).columns.tolist()
        n_cols = len(cols)
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                col_a, col_b = cols[i], cols[j]
                ratio_data[f"{col_a}{name_separator}{col_b}"] = temp_df[col_a] / (temp_df[col_b].abs() + epsilon)
                ratio_data[f"{col_b}{name_separator}{col_a}"] = temp_df[col_b] / (temp_df[col_a].abs() + epsilon)

    # List[Tuple[str, str]]: explicit pairs (numerator, denominator)
    elif len(ratio_features) > 0 and isinstance(ratio_features[0], tuple):
        for col_a, col_b in ratio_features:
            ratio_data[f"{col_a}{name_separator}{col_b}"] = temp_df[col_a] / (temp_df[col_b].abs() + epsilon)

    # List[str]: Pairwise subset from list of strings 
    else:
        cols = ratio_features
        n_cols = len(cols)
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                col_a, col_b = cols[i], cols[j]
                ratio_data[f"{col_a}{name_separator}{col_b}"] = temp_df[col_a] / (temp_df[col_b].abs() + epsilon)
                ratio_data[f"{col_b}{name_separator}{col_a}"] = temp_df[col_b] / (temp_df[col_a].abs() + epsilon)

    ratios_df = pd.DataFrame(ratio_data, index=temp_df.index)
    return pd.concat([temp_df, ratios_df], axis=1)
    # return pd.concat([temp_df[remaining_features], ratios_df], axis=1)

# STATELESS
# TODO: add logic for List[str] where all combinations between features included will be turned into a combined feature 
def combine_features(
        df: pd.DataFrame,
        combine_features: Optional[List[Tuple[str, str]]] = None,
        name_separator: str = "_&_",
    ) -> pd.DataFrame:
    """
    Computes combined features by concatenating string representations of specified columns.
    - If combine_features is a list of tuples (e.g. [('A', 'B')]), computes explicitly A&B.
    """
    temp_df = df.copy()
    combined_data = {}

    if combine_features is not None:
        for col_a, col_b in combine_features:
            combined_data[f"{col_a}{name_separator}{col_b}"] = (
                temp_df[col_a].astype(str) + name_separator + temp_df[col_b].astype(str)
            )

    combined_df = pd.DataFrame(combined_data, index=temp_df.index)
    return pd.concat([temp_df, combined_df], axis=1)