from functools import partial
import time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Optional, Any, Literal

from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils import compute_sample_weight

import lightgbm as lgb

from src.utils.helpers import get_cv_splits

ModelName = Literal["xgboost", "lightgbm", "catboost"]


def fit_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    weight_strategy: Optional[str | np.ndarray] = None,
    stopping_rounds: int = 150,
    verbose: bool = False,
):
    """Fit a model using model-specific training settings."""

    categorical_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()
    if weight_strategy is not None:
        sample_weights = compute_sample_weight(
            class_weight=weight_strategy,
            y=y_train,
        )
    else:
        sample_weights = None
    
    if model_name == "xgboost":
        fit_params = {
            "eval_set": [(X_valid, y_valid)],
            "verbose": 200 if verbose else False,
            "sample_weight": sample_weights,
        }

    elif model_name == "lightgbm":
        fit_params = {
            "eval_set": [(X_valid, y_valid)],
            "categorical_feature": categorical_features if categorical_features else "auto",
            "sample_weight": sample_weights,
            "callbacks": [
                lgb.early_stopping(
                    stopping_rounds=stopping_rounds,
                    verbose=200 if verbose else False,
                )
            ],
        }

    elif model_name == "catboost":
        fit_params = {
            "eval_set": (X_valid, y_valid),
            "cat_features": categorical_features if categorical_features else None,
            "early_stopping_rounds": stopping_rounds,
            "verbose": 200 if verbose else False,
            "sample_weight": sample_weights,
        }

    model.fit(X_train, y_train, **fit_params)

    return model

# FIXME: objective is still kinda hardcoded
def build_classifier(
    model_name: ModelName,
    metric_config: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
    seed: int = 42,
    device: str = "cpu",
    n_classes: int = 2,
) -> Any:
    """Build a classifier instance for the requested model family."""
    params = params or {}
    metric_config = metric_config or {}

    # Extract evaluation metric if explicitly provided in metric_config, otherwise let the model default
    eval_metric_override = metric_config.get("eval_metric", {}).get(model_name)

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        model_kwargs = {
            **params,
            "objective": "binary:logistic" if n_classes == 2 else "multi:softprob",
            "num_class": None if n_classes == 2 else n_classes,
            "enable_categorical": True,
            "early_stopping_rounds": 100,
            "random_state": seed,
            "device": device,
        }
        if eval_metric_override:
            model_kwargs["eval_metric"] = eval_metric_override
            
        return XGBClassifier(**model_kwargs)

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping

        model_kwargs = {
            **params,
            "objective": "binary" if n_classes == 2 else "multiclass",
            "n_classes": n_classes if n_classes > 2 else None,
            "verbose": -1,
            "random_state": seed,
            "device": device,
        }
        if eval_metric_override:
            model_kwargs["metric"] = eval_metric_override

        model = LGBMClassifier(**model_kwargs)
        model._early_stopping_cb = early_stopping(100, verbose=False)
        return model

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        model_kwargs = {
            **params,
            "loss_function": "Logloss" if n_classes == 2 else "MultiClass",
            "random_state": seed,
            "verbose": 0,
            "task_type": device.upper(),
            "allow_writing_files": False,
        }
        if eval_metric_override:
            model_kwargs["eval_metric"] = eval_metric_override

        return CatBoostClassifier(**model_kwargs)

    raise ValueError(f"Unsupported model_name: {model_name}")


def predict_fold(
    model: Any,
    task_type: str,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return validation/test predictions in both supported formats."""

    test_is_valid_file = X_test is not None and not X_test.empty

    if task_type == "regression":
        return (
            model.predict(X_valid), 
            model.predict(X_test) if test_is_valid_file else None,
            None, 
            None
        )

    # FIXME: some models might not support predict_proba (?)
    valid_proba = model.predict_proba(X_valid)
    test_proba = model.predict_proba(X_test) if test_is_valid_file else None
    return (
        np.argmax(valid_proba, axis=1),
        np.argmax(test_proba, axis=1) if test_proba is not None else None,
        valid_proba,
        test_proba,
    )


def train_cv_models(
    model_names: List[ModelName],
    X: pd.DataFrame,
    y: pd.Series,
    X_test: Optional[pd.DataFrame] = None,      # Optional because it is not needed for optuna trials
    metric_config: Optional[Dict[str, Any]] = None,
    return_proba: bool = True,                  # True for soft probabilities, False for hard classes
    task_type: str = "classification",          # "classification" or "regression"
    params: Optional[dict[str, dict[str, Any]]] = None,
    stopping_rounds: int = 150,
    seed: int = 42,
    split_config: Optional[dict] = None,
    device: str = "cpu",
    verbose: bool = True,
    optuna_trial: bool = False,
    preprocessing_pipeline: Optional[Pipeline] = None,
    weight_strategy: Optional[str | np.ndarray] = "balanced",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[float]]] | float:
    """Run fold-safe cross-validation for one or more model families."""

    metric_config = metric_config or {
        "name": "balanced_accuracy",
        "fn": balanced_accuracy_score,
        "needs_proba": False,
        "kwargs": {},
        "eval_metric": {}
    }
    split_config = split_config or {
        "n_splits": 5,
        "binned_stratify_continuous": False,
        "n_bins": 10,
    }
    params = params or {}

    if X_test is None and not optuna_trial:
        raise ValueError("X_test is required unless optuna_trial=True")

    metric_fn = (
        partial(metric_config["fn"], **metric_config["kwargs"]) 
        if metric_config["kwargs"] else metric_config["fn"]
    )

    n_samples = len(X)
    n_test = len(X_test) if X_test is not None else 0
    n_classes = int(y.nunique()) if task_type == "classification" else 1

    if optuna_trial and len(model_names) != 1:
        raise ValueError("optuna_trial requires exactly one model name")

    folds = get_cv_splits(X, y, task_type=task_type, random_state=seed, **split_config)
    OOF_PREDS: Dict[str, np.ndarray] = {}
    TEST_PREDS: Dict[str, np.ndarray] = {}
    FOLD_SCORES: Dict[str, List[float]] = {}

    for model_name in model_names:
        if verbose:
            print(f"\n{'='*20} Fitting {model_name} {'='*20}")

        oof_preds = np.zeros((n_samples, n_classes)) if task_type == "classification" and return_proba else np.zeros(n_samples)
        test_preds = np.zeros((n_test, n_classes)) if task_type == "classification" and return_proba else np.zeros(n_test)
        test_vote_counts = (
            np.zeros((n_test, n_classes), dtype=int)
            if task_type == "classification" and not return_proba and not optuna_trial
            else None
        )

        base_model = build_classifier(
            model_name, 
            metric_config,
            params.get(model_name, {}), 
            seed, 
            device, 
            n_classes,
        )

        fold_scores = []
        X_test_fold = X_test.copy() if X_test is not None else None

        for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
            if verbose:
                print(f"\n{'#'*10} Fold {fold}/{split_config['n_splits']} {'#'*10}")

            X_train = X.iloc[train_idx].copy()
            X_valid = X.iloc[valid_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_valid = y.iloc[valid_idx].copy()
            
            start = time.time()

            if preprocessing_pipeline:

                fold_pipeline = clone(preprocessing_pipeline)
                X_train = fold_pipeline.fit_transform(X_train)
                X_valid = fold_pipeline.transform(X_valid)
                X_test_fold = fold_pipeline.transform(X_test) if X_test is not None else None

            model = fit_model(
                model_name=model_name, 
                model=clone(base_model), 
                X_train=X_train, 
                y_train=y_train, 
                X_valid=X_valid, 
                y_valid=y_valid, 
                stopping_rounds=stopping_rounds, 
                verbose=verbose,
                weight_strategy=weight_strategy
                )

            val_hard, test_hard, val_proba, test_proba = predict_fold(
                model, task_type, X_valid, X_test_fold
            )

            if task_type == "classification":
                oof_preds[valid_idx] = val_proba if return_proba else val_hard
                if not optuna_trial:
                    if return_proba:
                        test_preds += test_proba
                    else:
                        class_indices = pd.Index(model.classes_).get_indexer(test_hard)
                        # TODO: check np methods
                        np.add.at(
                            test_vote_counts,
                            (np.arange(n_test), class_indices),
                            1,
                        )
                if metric_config["needs_proba"]:
                    eval_preds = val_proba[:, 1] if n_classes == 2 else val_proba
                else:
                    eval_preds = val_hard

            else:
                oof_preds[valid_idx] = val_hard
                if not optuna_trial:
                    test_preds += test_hard
                eval_preds = val_hard

            fold_score = metric_fn(y_valid, eval_preds)
            fold_scores.append(fold_score)
            if verbose:
                print(f" Fold {fold} {metric_config['name']}: {fold_score:.5f}")
                print(f"Fold {fold} finished in {time.time() - start:.2f}s")

        if verbose:
            print(f"Mean Validation {metric_config['name']}: {np.mean(fold_scores):.5f}")

        if not optuna_trial:
            if task_type == "classification" and not return_proba:
                test_preds = model.classes_[np.argmax(test_vote_counts, axis=1)]
            else:
                test_preds /= split_config["n_splits"]

        OOF_PREDS[model_name] = oof_preds
        TEST_PREDS[model_name] = test_preds
        FOLD_SCORES[model_name] = fold_scores

    if optuna_trial:
        return float(np.mean(FOLD_SCORES[model_names[0]]))
        
    return OOF_PREDS, TEST_PREDS, FOLD_SCORES