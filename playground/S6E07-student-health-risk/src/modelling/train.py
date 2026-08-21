from functools import partial
import time
from attrs import field
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Optional, Any, Literal

from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import TargetEncoder
from sklearn.base import clone

import lightgbm as lgb

from src.utils.helpers import iqr_outlier_capping, get_cv_splits


ModelName = Literal["xgboost", "lightgbm", "catboost"]


def fit_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_features: list[str] | None = None,
    stopping_rounds: int = 150,
    verbose: bool = False,
):
    """Fit a model using model-specific training settings."""

    if model_name == "xgboost":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=200 if verbose else False,
        )

    elif model_name == "lightgbm":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=stopping_rounds,
                    verbose=200 if verbose else False,
                )
            ],
        )

    elif model_name == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            cat_features=categorical_features,
            early_stopping_rounds=stopping_rounds,
            verbose=200 if verbose else False,
        )

    else:
        model.fit(X_train, y_train)

    return model

# TODO: add support for automated weighting (i.e., scale_pos_weight)
def build_classifier(
    model_name: ModelName,
    params: dict[str, Any] = field(default_factory=dict),
    seed: int = 42,
    device: str = "cpu",
    n_classes: int = 2,
) -> Any:
    """Build a classifier instance for the requested model family."""
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            num_class=None if n_classes == 2 else n_classes,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            enable_categorical=True,
            early_stopping_rounds=100,
            random_state=seed,
            device=device,
        )

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping

        model = LGBMClassifier(
            **params,
            objective="binary" if n_classes == 2 else "multiclass",
            n_classes=n_classes if n_classes > 2 else None,
            verbose=-1,
            random_state=seed,
            device=device,
        )
        model._early_stopping_cb = early_stopping(100, verbose=False)  # small helper hook
        return model

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **params,
            loss_function="Logloss" if n_classes == 2 else "MultiClass",
            random_state=seed,
            verbose=0,
            task_type=device.upper(),
            allow_writing_files=False,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def train_cv_models(
    estimators: List[Tuple[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    task_type: str = "classification",          # "classification" or "regression"
    metric_fn: Callable = None,                 # e.g., balanced_accuracy_score
    metric_needs_proba: bool = False,          # True if metric expects probabilities (e.g., roc_auc_score)
    return_proba: bool = True,                 # True for soft probabilities, False for hard classes
    te_cols: Optional[List[str]] = None,        # Target encoding columns
    categorical_features: Optional[List[str]] = None,
    stopping_rounds: int = 150,
    seed: int = 42,
    split_config: dict = {
        "n_splits": 5,
        "binned_stratify_continuous": False,
        "n_bins": 10,
    },
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[float]]]:
    """
    Runs K-Fold cross-validation across multiple estimators, handling outlier capping,
    optional target encoding, dynamic metric evaluation, and soft/hard predictions.
    """

    # ------------------------------------------------------------------ Setup Cross-Validation Splitter
    folds = get_cv_splits(
        X, y,
        task_type=task_type,
        random_state=seed,
        n_splits=split_config["n_splits"],
        binned_stratify_continuous=split_config["binned_stratify_continuous"],
        n_bins=split_config["n_bins"]
    )

    # ------------------------------------------------------------------ Output storage
    OOF_PREDS: Dict[str, np.ndarray] = {}
    TEST_PREDS: Dict[str, np.ndarray] = {}
    FOLD_SCORES: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------ Start training loop

    n_samples = len(X)
    n_test = len(X_test)
    n_classes = len(y.unique()) if task_type == "classification" else 1

    te_cols = te_cols or []
    categorical_features = categorical_features or []

    for model_name, base_model in estimators:

        if verbose:
            print(f"\n{'='*20} Fitting {model_name} {'='*20}")

        # Determine output shapes based on soft (proba) vs hard predictions
        if task_type == "classification" and return_proba:
            oof_preds = np.zeros((n_samples, n_classes))
            test_preds = np.zeros((n_test, n_classes))
        else:
            oof_preds = np.zeros(n_samples)
            test_preds = np.zeros(n_test)

        fold_scores = []

        X_temp = X.copy()
        X_test_temp = X_test.copy()

        for fold, (train_idx, valid_idx) in enumerate(folds, start=1):

            if verbose:
                print(f"\n{'#'*10} Fold {fold}/{split_config['n_splits']} {'#'*10}")

            # Splitting and capping numerical outliers
            num_cols = X_temp.select_dtypes(include=np.number).columns.difference([y.name])
            X_train, X_valid, _ = iqr_outlier_capping(
                X_temp.iloc[train_idx], 
                X_temp.iloc[valid_idx], 
                None, 
                columns=num_cols
            )
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            X_test_loop = X_test_temp.copy()

            # Target Encoding Operationalisation
            if te_cols:
                valid_te_cols = [c for c in te_cols if c in X_train.columns]
                if valid_te_cols:
                    if verbose and fold == 0:
                        print(f"ℹ️ Target Encoding applied to: {valid_te_cols}")
                    
                    target_enc = TargetEncoder(
                        target_type="auto",
                        smooth="auto",
                        cv=4,
                        shuffle=True,
                        random_state=seed,
                    )
                    
                    X_train[valid_te_cols] = target_enc.fit_transform(X_train[valid_te_cols], y_train)
                    X_valid[valid_te_cols] = target_enc.transform(X_valid[valid_te_cols])
                    X_test_loop[valid_te_cols] = target_enc.transform(X_test_loop[valid_te_cols])

                    # Target encoded features are now numeric; remove them from categorical list
                    categorical_features = [c for c in categorical_features if c not in valid_te_cols]

            start = time.time()

            # Model Fitting
            model = clone(base_model)
            model = fit_model(
                model_name,
                model,
                X_train,
                y_train,
                X_valid,
                y_valid,
                categorical_features=categorical_features,
                stopping_rounds=stopping_rounds,
                verbose=verbose
            )

            # Predictions & Probability Handling
            if task_type == "classification":
                val_proba = model.predict_proba(X_valid)
                test_proba = model.predict_proba(X_test_loop)
                val_hard = np.argmax(val_proba, axis=1)
                test_hard = np.argmax(test_proba, axis=1)

                if return_proba:
                    oof_preds[valid_idx] = val_proba
                    test_preds += test_proba
                else:
                    oof_preds[valid_idx] = val_hard
                    test_preds += test_hard

                eval_preds = val_proba if metric_needs_proba else val_hard
            else:
                val_pred = model.predict(X_valid)
                test_pred = model.predict(X_test_loop)

                oof_preds[valid_idx] = val_pred
                test_preds += test_pred
                eval_preds = val_pred

            # Dynamic Metric Calculation
            if metric_fn is not None:
                fold_score = metric_fn(y_valid, eval_preds)
                fold_scores.append(fold_score)
                if verbose:
                    print(f" Fold {fold} {metric_fn.__name__}: {fold_score:.5f}")

            end = time.time()
            if verbose:
                print(f"Fold {fold} finished in {end - start:.2f}s")

        if fold_scores and verbose:
            print(f"Mean Validation {metric_fn.__name__ if metric_fn else 'Score'}: {np.mean(fold_scores):.5f}")

        # Average test predictions across folds
        test_preds /= split_config['n_splits']

        OOF_PREDS[model_name] = oof_preds
        TEST_PREDS[model_name] = test_preds
        FOLD_SCORES[model_name] = fold_scores

    return OOF_PREDS, TEST_PREDS, FOLD_SCORES


def train_cv_models_II(
    model_names: List[ModelName],
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    metric_config: Dict[str, Any] = {
        "name": "balanced_accuracy", 
        "fn": balanced_accuracy_score, 
        "needs_proba": False,
        "kwargs": {}
    },
    return_proba: bool = True,                 # True for soft probabilities, False for hard classes
    task_type: str = "classification",          # "classification" or "regression"
    te_cols: Optional[List[str]] = None,        # Target encoding columns
    params: dict[str, dict[str, Any]] = {},
    stopping_rounds: int = 150,
    seed: int = 42,
    split_config: dict = {
        "n_splits": 5,
        "binned_stratify_continuous": False,
        "n_bins": 10,
    },
    device: str = "cpu",
    verbose: bool = True,
    optuna_trial: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[float]]] | float:
    """
    Runs K-Fold cross-validation across multiple estimators, handling outlier capping,
    optional target encoding, dynamic metric evaluation, and soft/hard predictions.

    When ``optuna_trial`` is enabled, exactly one model must be supplied and the
    model's mean fold score is returned for Optuna to maximize or minimize.
    """

    # ------------------------------------------------------------------ Setup
    metric_fn = (
        partial(metric_config["fn"], **metric_config["kwargs"]) 
        if metric_config["kwargs"] 
        else metric_config["fn"]
    )

    n_samples = len(X)
    n_test = len(X_test)
    n_classes = int(y.nunique()) if task_type == "classification" else 1

    te_cols = te_cols or []
    categorical_features = (
        X.select_dtypes(include="category")
        .columns
        .difference([y.name])
        .tolist()
    )

    if optuna_trial and len(model_names) != 1:
        raise ValueError("optuna_trial requires exactly one model name")

    folds = get_cv_splits(
        X, y,
        task_type=task_type,
        random_state=seed,
        n_splits=split_config["n_splits"],
        binned_stratify_continuous=split_config["binned_stratify_continuous"],
        n_bins=split_config["n_bins"]
    )

    estimators = [
        (
            model_name, 
            build_classifier(
                model_name, 
                params.get(model_name, {}), 
                seed=seed,
                device=device,
                n_classes=n_classes
            )
        )
        for model_name in model_names # params.keys() would work too but what if i don't have params? It wouldn't start with models on default params
    ]

    # ------------------------------------------------------------------ Output storage
    OOF_PREDS: Dict[str, np.ndarray] = {}
    TEST_PREDS: Dict[str, np.ndarray] = {}
    FOLD_SCORES: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------ Start training loop

    for model_name, base_model in estimators:

        if verbose:
            print(f"\n{'='*20} Fitting {model_name} {'='*20}")

        # Determine output shapes based on soft (proba) vs hard predictions
        if task_type == "classification" and return_proba:
            oof_preds = np.zeros((n_samples, n_classes))
            test_preds = np.zeros((n_test, n_classes))
        else:
            oof_preds = np.zeros(n_samples)
            test_preds = np.zeros(n_test)

        fold_scores = []

        X_temp = X.copy()
        X_test_temp = X_test.copy()

        for fold, (train_idx, valid_idx) in enumerate(folds, start=1):

            if verbose:
                print(f"\n{'#'*10} Fold {fold}/{split_config['n_splits']} {'#'*10}")

            # Capping numerical outliers
            num_cols = (
                X_temp.select_dtypes(include=np.number)
                .columns
                .difference([y.name])
            )
            X_train, X_valid, _ = iqr_outlier_capping(
                X_temp.iloc[train_idx], 
                X_temp.iloc[valid_idx], 
                None, 
                columns=num_cols
            )
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            X_test_loop = X_test_temp.copy()

            # Target Encoding Operationalisation
            if te_cols:
                if verbose and fold == 0:
                    print(f"ℹ️ Target Encoding applied to: {te_cols}")
                
                target_enc = TargetEncoder(
                    target_type="auto",
                    smooth="auto",
                    cv=4,
                    shuffle=True,
                    random_state=seed,
                )
                
                X_train[te_cols] = target_enc.fit_transform(X_train[te_cols], y_train)
                X_valid[te_cols] = target_enc.transform(X_valid[te_cols])
                X_test_loop[te_cols] = target_enc.transform(X_test_loop[te_cols])

                # Target encoded features are now numeric; remove them from categorical list
                categorical_features_loop = [c for c in categorical_features if c not in te_cols]

            start = time.time()

            # Model Fitting
            model = clone(base_model)
            model = fit_model(
                model_name,
                model,
                X_train,
                y_train,
                X_valid,
                y_valid,
                categorical_features=categorical_features_loop,
                stopping_rounds=stopping_rounds,
                verbose=verbose
            )

            # Predictions & Probability Handling
            if task_type == "classification":
                val_proba = model.predict_proba(X_valid)
                val_hard = np.argmax(val_proba, axis=1)

                if return_proba:
                    oof_preds[valid_idx] = val_proba
                else:
                    oof_preds[valid_idx] = val_hard

                if not optuna_trial:
                    test_proba = model.predict_proba(X_test_loop)
                    test_hard = np.argmax(test_proba, axis=1)
                    test_preds += test_proba if return_proba else test_hard

                eval_preds = val_proba if metric_config['needs_proba'] else val_hard
            else:
                val_pred = model.predict(X_valid)

                oof_preds[valid_idx] = val_pred
                if not optuna_trial:
                    test_preds += model.predict(X_test_loop)
                eval_preds = val_pred

            # Dynamic Metric Calculation
            if metric_fn is not None:
                fold_score = metric_fn(y_valid, eval_preds)
                fold_scores.append(fold_score)
                if verbose:
                    print(f" Fold {fold+1} {metric_config['name']}: {fold_score:.5f}")

            end = time.time()
            if verbose:
                print(f"Fold {fold+1} finished in {end - start:.2f}s")

        if fold_scores and verbose:
            print(f"Mean Validation {metric_config['name']}: {np.mean(fold_scores):.5f}")

        if not optuna_trial:
            # Average test predictions across folds
            test_preds /= split_config['n_splits']

        OOF_PREDS[model_name] = oof_preds
        TEST_PREDS[model_name] = test_preds
        FOLD_SCORES[model_name] = fold_scores

    if optuna_trial:
        return float(np.mean(FOLD_SCORES[model_names[0]]))

    return OOF_PREDS, TEST_PREDS, FOLD_SCORES