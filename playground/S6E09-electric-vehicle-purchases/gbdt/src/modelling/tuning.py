import time
import json
from pathlib import Path
from functools import partial
from typing import Any, Dict, Optional, Tuple, Callable, Literal, List

import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.modelling.train import train_cv_models


ModelName = Literal["xgboost", "lightgbm", "catboost"]


def _suggest_params(trial: optuna.Trial, model_name: ModelName) -> dict[str, Any]:
    """Suggest Optuna hyperparameters for a given model family."""
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        }
    elif model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        }
    elif model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0),
        }
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


# TODO: n_classes works for classification, but let's also tackle regression, eventually
def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: ModelName,
    sampling_fraction: float = 1.0,
    metric_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    device: str = "cpu",
    task_type: str = "classification",
    stopping_rounds: int = 150,
    split_config: Optional[dict] = None,
    preprocessing_pipeline: Optional[Pipeline] = None,
    verbose: bool = False,
) -> float:

    """Evaluate one Optuna trial using cross-validation."""

    metric_config = metric_config or {
        "name": "balanced_accuracy",
        "fn": balanced_accuracy_score,
        "needs_proba": False,
        "kwargs": {},
    }
    split_config = split_config or {
        "n_splits": 3,
        "binned_stratify_continuous": False,
        "n_bins": 10,
    }
    params = _suggest_params(trial, model_name)

    X_tune, _, y_tune, _ = train_test_split(
        X, y,
        train_size=sampling_fraction,
        random_state=seed,
        stratify=y
    )

    return train_cv_models(
        model_names=[model_name],
        X=X_tune,
        y=y_tune,
        X_test=None,
        metric_config=metric_config,
        return_proba=False,
        task_type=task_type,
        params={model_name: params},
        stopping_rounds=stopping_rounds,
        seed=seed,
        split_config=split_config,
        device=device,
        verbose=verbose,
        optuna_trial=True,
        preprocessing_pipeline=preprocessing_pipeline
    )


def save_optimization_results(
    study_results: dict[str, dict[str, Any]],
    evals_directory: Path,
) -> None:
    """Save Optuna optimization results and best parameters."""

    if not study_results:
        print("No study results to save.")
        return

    if not evals_directory.exists():
        print(f"Directory {evals_directory} does not exist. Creating it.")
        evals_directory.mkdir(parents=True, exist_ok=True)
    
    best_params_dict = {
        model_name: {k: v for k, v in result["Best params"].items() if pd.notna(v)}
        for model_name, result in study_results.items()
    }

    study_df = pd.DataFrame(
        [
            {"Model": model_name, "Best_Value": result["Best value"], **result["Best params"]}
            for model_name, result in study_results.items()
        ]
    )
    
    with open(evals_directory / "best_params.json", "w") as f:
        json.dump(best_params_dict, f, indent=4)
        
    study_df.to_csv(evals_directory / "optuna_results.csv", index=False)

    print(f"✅ Saved best parameters to {evals_directory / 'best_params.json'}")
    print(f"✅ Saved optimization summary to {evals_directory / 'optuna_results.csv'}")


def load_optimization_results(
    evals_directory: Path,
    best_params: Optional[dict[str, dict[str, Any]]] = {},
) -> Optional[dict[str, dict[str, Any]]]:
    """Load optimized parameters from current session or disk."""
    params = None

    if best_params:
        print("ℹ️ Using parameters from the current Optuna session.")
        params = best_params
    else:
        params_path = evals_directory / "best_params.json"
        if params_path.exists():
            print(f"ℹ️ Loading parameters from {params_path}")
            with open(params_path) as f:
                params = json.load(f)
        else:
            print(f"⚠️ Optimized params requested but {params_path} not found.")

    return params

# TODO: add threshold optimization here ot make it into its own method
def weights_optimization(
    trial: optuna.Trial,
    y: pd.Series,
    oof_probs: dict,
    metric_config: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> float:
    """Optimize ensemble weights using Optuna."""

    model_keys = list(oof_probs.keys())
    if not model_keys:
        raise ValueError("oof_probs must contain at least one model's predictions")

    metric_config = metric_config or {
        "name": "balanced_accuracy",
        "fn": balanced_accuracy_score,
        "needs_proba": False,
        "kwargs": {},
    }

    metric_fn = (
        partial(metric_config["fn"], **metric_config["kwargs"]) 
        if metric_config["kwargs"] else metric_config["fn"]
    )

    if len(model_keys) == 1:
        preds = np.argmax(oof_probs[model_keys[0]], axis=1)
        score = metric_fn(y, preds)
        return score

    weights = []
    remaining = 1.0
    for key in model_keys[:-1]:
        w = trial.suggest_float(key, 0.0, remaining)
        weights.append(w)
        remaining -= w
    weights.append(remaining)

    ensemble_probs = np.zeros_like(next(iter(oof_probs.values())))
    for w, key in zip(weights, model_keys):
        ensemble_probs += w * oof_probs[key]

    preds = np.argmax(ensemble_probs, axis=1)
    score = float(metric_fn(y, preds))
    if verbose:
        print(f"{metric_config['name'].upper()} | ensemble: {score:.4f}")
    return score