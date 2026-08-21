import time
from functools import partial
from typing import Any, Dict, Optional, Tuple, Callable, Literal

import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.base import clone

from src.utils.helpers import get_cv_splits, iqr_outlier_capping
from src.modelling.train import train_cv_models_II


ModelName = Literal["xgboost", "lightgbm", "catboost"]


def _suggest_params(trial: optuna.Trial, model_name: ModelName) -> dict[str, Any]:
    """Suggest Optuna hyperparameters for a given model family."""
    # generic suggestions – will be filtered/renamed per model type
    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
    }

    if model_name == "xgboost":
        params.update(
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.9)
        )
    elif model_name == "lightgbm":
        params.update(
            num_leaves=trial.suggest_int("num_leaves", 20, 100),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 100),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 0.9)
        )
    elif model_name == "catboost":
        # Rename generic keys to CatBoost‑compatible ones
        cat_params = {
            "iterations": params.pop("n_estimators"),
            "depth": params.pop("max_depth"),
            "bagging_temperature": params.pop("subsample"),    # approximate mapping
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0),
        }
        params = cat_params
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return params

# TODO: consider removing this
def dynamic_default_params(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = 'classification',
        device: str = 'cpu',
        seed: int = 42,
        ) -> dict:
    n_rows, n_cols = X.shape

    if task_type == 'classification':
        if len(y.unique()) == 2:
            pos_count = (y == 1).sum()
            neg_count = (y == 0).sum()
            scale_pos_weight = pos_count / neg_count if neg_count > 0 else 1.0
        else:
            scale_pos_weight = 1

    is_small = n_rows < 5_000
    is_large = n_rows > 100_000
    is_wide = n_cols > 50

    base_depth = 4 if is_small else (8 if is_large else 6)
    row_subsample = 0.8 if is_large else 0.7
    col_subsample = 0.6 if is_wide else 0.7
    lr = 0.05 if is_small else 0.02

    common = {
        'random_state': seed,
        'n_estimators': 3000,
    }

    return {
        'xgboost': {
            **common,
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
            'enable_categorical': True,
            'tree_method': 'hist',
            'device': device,
            'learning_rate': lr,
            'max_depth': base_depth,
            'subsample': row_subsample,
            'colsample_bytree': col_subsample,
            'early_stopping_rounds': 50 if is_small else 100,
        },
        'lightgbm': {
            **common,
            'objective': 'multiclass',
            'class_weight': 'balanced',
            'device': 'cpu',
            'learning_rate': lr,
            'num_leaves': min(2**base_depth - 1, 255),
            'subsample': row_subsample,
            'colsample_bytree': col_subsample,
            'subsample_freq': 1,
            'min_child_samples': 10 if is_small else (100 if is_large else 20),
            'verbose': -1,
        },
        'catboost': {
            **common,
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'auto_class_weights': 'Balanced',
            'classes_count': len(y.unique()),
            'task_type': 'CPU' if device == 'cpu' else 'GPU',
            'learning_rate': lr,
            'depth': min(base_depth, 16),
            'l2_leaf_reg': 5.0 if is_small else 3.0,
            'bootstrap_type': 'Bernoulli',
            'subsample': row_subsample,
            'early_stopping_rounds': 50 if is_small else 100,
            'verbose': 0,
        }
    }

# TODO: n_classes works for classification, but let's also tackle regression, eventually
def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: ModelName,
    metric_config: Dict[str, Any] = {
        "name": "balanced_accuracy", 
        "fn": balanced_accuracy_score, 
        "needs_proba": False,
        "kwargs": {}
    },
    seed: int = 42,
    device: str = "cpu",
    target: str = "target",
    task_type: str = "classification",
    te_cols: Optional[list] = None,
    categorical_features: Optional[list] = None,
    stopping_rounds: int = 150,
    split_config: dict = {
        "n_splits": 3,
        "binned_stratify_continuous": False,
        "n_bins": 10,
    },
    verbose: bool = False,
) -> float:

    """Evaluate one Optuna trial using cross-validation."""

    params = _suggest_params(trial, model_name)
    return train_cv_models_II(
        model_names=[model_name],
        X=X,
        y=y,
        X_test=X.iloc[:0].copy(),
        metric_config=metric_config,
        return_proba=False,
        task_type=task_type,
        te_cols=te_cols,
        params={model_name: params},
        stopping_rounds=stopping_rounds,
        seed=seed,
        split_config=split_config,
        device=device,
        verbose=verbose,
        optuna_trial=True,
    )


def weights_optimization(
    trial: optuna.Trial,
    y: pd.Series,
    oof_probs: dict,
    metric_config: Tuple[str, Callable, Optional[dict]] = ("balanced_accuracy", balanced_accuracy_score, None),
    verbose: bool = False
) -> float:
    """Optimize ensemble weights using Optuna."""
    model_keys = list(oof_probs.keys())
    if not model_keys:
        raise ValueError("oof_probs must contain at least one model's predictions")

    metric_name, metric_fn, metric_kwargs = metric_config
    metric_kwargs = metric_kwargs or {}
    metric_fn = partial(metric_fn, **metric_kwargs) if metric_kwargs else metric_fn

    if len(model_keys) == 1:
        preds = np.argmax(oof_probs[model_keys[0]], axis=1)
        score = metric_fn(y, preds)
        return score

    weights = []
    remaining = 1.0
    for key in model_keys[:-1]:
        w = trial.suggest_float(f"w_{key}", 0.0, remaining)
        weights.append(w)
        remaining -= w
    weights.append(remaining)

    ensemble_probs = np.zeros_like(next(iter(oof_probs.values())))
    for w, key in zip(weights, model_keys):
        ensemble_probs += w * oof_probs[key]

    preds = np.argmax(ensemble_probs, axis=1)
    score = float(metric_fn(y, preds))
    if verbose:
        print(f"{metric_name.upper()} | ensemble: {score:.4f}")
    return score