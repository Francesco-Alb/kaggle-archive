# utils/data_setup.py

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

def get_dataset_paths(
    competition_name: str,
    is_kaggle: bool,
    extra_dataset_handle: str | None = None,
    extra_csv_name: str = "PLACEHOLDER.csv"
) -> Tuple[str, Optional[str]]:
    """
    Determine dataset paths based on the execution environment.

    Args:
        competition_name: The identifier for the Playground Series competition.
        is_kaggle: Boolean flag indicating if running inside Kaggle notebooks.
        add_extra_data: Flag to enable downloading supplementary datasets.
        extra_dataset_handle: Kaggle dataset handle (e.g., "username/dataset-name").
        extra_csv_name: Filename to extract from the extra dataset.

    Returns:
        A tuple containing:
            - main_path (str): Path to the competition data.
            - extra_path (Optional[str]): Path to extra data, or None.
    """

    main_path = None
    extra_path: Optional[str] = None

    if not is_kaggle:
        import kagglehub
        
        main_path = kagglehub.competition_download(competition_name)
        if extra_dataset_handle:
            try:
                base_path = kagglehub.dataset_download(extra_dataset_handle)
                extra_path = f"{base_path}/{extra_csv_name}"
                print("✅ Extra data loaded.")
            except Exception as e:
                print(f"ℹ️ Extra data not found: {e}")
        
    else:
        main_path = f'/kaggle/input/competitions/{competition_name}' 
        if extra_dataset_handle:
            try:
                extra_path = f"/kaggle/input/competitions/{extra_dataset_handle}/{extra_csv_name}"
                print("✅ Extra data loaded.")
            except Exception as e:
                print(f"ℹ️ Extra data not found: {e}")

    return main_path, extra_path


def load_data(
    data_path: str | Path,
    target: str,
    smoke_test: bool = False,
    sample_size: int = 1000,
    seed: int = 42,
    extra_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load train, test, and optional extra datasets, standardizing column names and alignments.

    Args:
        data_path: Path to the directory containing 'train.csv' and 'test.csv'.
        target: Name of the target label column.
        smoke_test: If True, sub-samples datasets to 1000 rows for fast execution.
        sample_size: Number of rows to sample when `smoke_test` is enabled.
        seed: Random state seed used for sampling in smoke test mode.
        extra_path: Path to the supplementary CSV file.

    Returns:
        dict[str, pd.DataFrame]: A dictionary containing keys 'train', 'test',
            and optionally 'extra' mapped to their cleaned DataFrames.
    """
    # ------------------------------------------------------------------ Normalize paths
    if type(data_path) == str:
        data_path = Path(data_path)

    if type(extra_path) == str:
        extra_path = Path(extra_path)

    # ------------------------------------------------------------------ Load Data
    train_path = data_path / "train.csv"
    test_path = data_path / "test.csv"

    train_df = pd.read_csv(train_path, index_col=[0])
    test_df = pd.read_csv(test_path, index_col=[0])
    train_extra = (
        pd.read_csv(extra_path)
        if extra_path
        else None
    )

    if smoke_test:
        print("ℹ️ Running in smoke test mode.")
        train_df = train_df.sample(n=sample_size, random_state=seed)
        test_df = test_df.sample(n=sample_size, random_state=seed)

    # ------------------------------------------------------------------ Clean column names
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        return df
    
    train_df = _normalize(train_df)
    test_df = _normalize(test_df)
    if train_extra is not None:
        train_extra = _normalize(train_extra)

    # ------------------------------------------------------------------ Define dataset dictionary
    datasets: dict[str, pd.DataFrame] = {
        "train": train_df,
        "test": test_df,
    }

    if train_extra is not None:
        # drop columns that exist only in extra to keep feature set consistent
        extra_only = set(train_extra.columns) - set(train_df.columns)
        if extra_only:
            print("Adding data from extra path")
            print(f"Columns in 'extra' that are missing from 'train': {extra_only}")
            print(f"Dropping {extra_only} col from extra data")
            train_extra = train_extra.drop(columns=extra_only)
        datasets["extra"] = train_extra

    else:
        print("ℹ️ No extra data path found or configured.")

    assert target in datasets["train"].columns, f"Target column '{target}' not found in the dataset."

    return datasets