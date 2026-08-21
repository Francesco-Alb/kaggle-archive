import random
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


def iqr_outlier_capping(
        train: pd.DataFrame,
        valid: pd.DataFrame = None,
        test: pd.DataFrame = None,
        columns: list = None
        ) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Applies IQR-based outlier capping to specified columns of one, two, or three DataFrames.

    Parameters:
        train (pd.DataFrame): The training DataFrame used to calculate IQR thresholds.
        valid (pd.DataFrame, optional): The validation DataFrame to cap using train thresholds.
        test (pd.DataFrame, optional): The test DataFrame to cap using train thresholds.
        columns (list, optional): List of column names to apply capping to. If None, applies to all numerical columns.

    Returns:
        tuple: A tuple containing:
            - train_capped (pd.DataFrame): Capped training DataFrame.
            - valid_capped (pd.DataFrame or None): Capped validation DataFrame (if provided).
            - test_capped (pd.DataFrame or None): Capped test DataFrame (if provided).

    Note: Make sure there are no nans
    """
    train_capped = train.copy()
    valid_capped = valid.copy() if valid is not None else None
    test_capped = test.copy() if test is not None else None

    if columns is None:
        columns = train.select_dtypes(include='number').columns.tolist()

    for col in columns:
        Q1 = np.percentile(train[col].dropna(), 25)
        Q3 = np.percentile(train[col].dropna(), 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        train_capped[col] = np.clip(train_capped[col], lower_bound, upper_bound)

        if valid is not None:
            valid_capped[col] = np.clip(valid[col], lower_bound, upper_bound)

        if test is not None:
            test_capped[col] = np.clip(test[col], lower_bound, upper_bound)

    return train_capped, valid_capped, test_capped


def print_with_sep(text, sep="=", n=40):
    print("\n")
    print(sep * n)
    print('\t', text)
    print(sep * n)


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
