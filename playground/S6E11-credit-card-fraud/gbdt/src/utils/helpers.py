import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, KFold


def seed_everything(seed: int) -> None:
    """
    Seeds the random number generators.

    Args:
        seed: The seed to set.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ Seed set to {seed}")


def print_with_sep(text, sep="=", n=40):
    print("\n")
    print(sep * n)
    print('\t', text)
    print(sep * n)


def save_predictions(
    oof_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    preds_directory: str|Path,
) -> None:
    """Save OOF and test predictions using Parquet format, creating directory if needed."""
    preds_directory = Path(preds_directory)

    if not preds_directory.exists():
        print(f"Directory {preds_directory} does not exist. Creating it.")
        preds_directory.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------ Save OOF predictions
    for model_name, preds in oof_preds.items():
        oof_path = preds_directory / f"oof_{model_name}.parquet"
        if preds is not None: # and not oof_path.exists():
            pd.DataFrame(preds).to_parquet(oof_path)
        
    # ------------------------------------------ Save Test predictions
    for model_name, preds in test_preds.items():
        test_path = preds_directory / f"test_{model_name}.parquet"
        if preds is not None: # and not test_path.exists():
            pd.DataFrame(preds).to_parquet(test_path)

    print(f"✅ Saved predictions to {preds_directory}")

def load_predictions(
    model_names: list[str] | None = None,
    preds_directory: str|Path = "predictions",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load OOF and test predictions from disk."""

    preds_directory = Path(preds_directory)
    oof_preds = {}
    test_preds = {}
    
    if model_names is None:
        model_names = []
        if preds_directory.exists():
            for path in preds_directory.glob("oof_*.parquet"):
                name = path.stem[4:]  # remove "oof_"
                if name not in model_names:
                    model_names.append(name)
            for path in preds_directory.glob("test_*.parquet"):
                name = path.stem[5:]  # remove "test_"
                if name not in model_names:
                    model_names.append(name)
        model_names = sorted(model_names)

    for model_name in model_names:
        oof_path = preds_directory / f"oof_{model_name}.parquet"
        test_path = preds_directory / f"test_{model_name}.parquet"
        
        if oof_path.exists():
            oof_preds[model_name] = pd.read_parquet(oof_path).to_numpy()
        else:
            print(f"⚠️ OOF predictions for {model_name} not found at {oof_path}")
            
        if test_path.exists():
            test_preds[model_name] = pd.read_parquet(test_path).to_numpy()
        else:
            print(f"⚠️ Test predictions for {model_name} not found at {test_path}")
            
    print(f"ℹ️ Loaded predictions from {preds_directory}")
    return oof_preds, test_preds


def get_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "classification",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    binned_stratify_continuous: bool = False,
    n_bins: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates cross-validation splits for classification or regression.

    Args:
        X: Features DataFrame.
        y: Target Series.
        task_type: "classification" or "regression".
        n_splits: Number of folds.
        shuffle: Whether to shuffle the data.
        random_state: Seed for reproducibility.
        binned_stratify_continuous: If True, uses binned stratification for regression.
        n_bins: Number of bins for continuous stratification.

    Returns:
        List of (train_idx, valid_idx) tuples.
    """
    if task_type == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(splitter.split(X, y))

    if binned_stratify_continuous:
        # Create discrete quantile bins for regression target stratification
        y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return list(splitter.split(X, y_binned))

    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(splitter.split(X))
