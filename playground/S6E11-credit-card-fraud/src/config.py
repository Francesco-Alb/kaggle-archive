from pathlib import Path

import torch
from typing import Callable
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass, asdict, field

@dataclass
class EnvConfig:
    competition: str = "playground-series-s6e11"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    smoke_test: bool = True


# TODO: consider adding a ProcessingConfig or FeatureConfig for stuff like feature_selection, folds, use_te
@dataclass
class DataConfig:
    data_path: str = ""
    extra_path: str | None = None
    target: str = "is_fraud"
    metric: str = "roc_auc_score"
    evals_directory: Path = Path("evals")
    folds: int = 5


@dataclass
class SplitConfig:
    n_splits: int = 5
    binned_stratify_continuous: bool = False
    n_bins: int = 10


@dataclass
class MetricConfig:
    name: str = "roc-auc"
    fn: Callable = roc_auc_score
    needs_proba: bool = True
    kwargs: dict = field(default_factory=dict)
    eval_metric: dict = field(default_factory=lambda: {
        "xgboost": "auc",
        "lightgbm": "auc",
        "catboost": "AUC",
    })


@dataclass
class PreprocessingConfig:
    add_extra_data: bool = False
    add_features: bool = False
    feature_selection: bool = False
    use_te: bool = False


@dataclass
class TrainConfig:
    task_type: str = "classification"
    return_proba: bool = True
    stopping_rounds: int = 150
    weight_strategy: str | None = "balanced"
    verbose: bool = True


@dataclass
class OptunaConfig:
    hp_search: bool = False
    search_weights: bool = True
    optuna_trials: int = 30
    stopping_rounds: int = 150
    sampling_fraction: float = 1.0
    split: SplitConfig = field(default_factory=lambda: SplitConfig(n_splits=3))
    verbose: bool = False

@dataclass
class TrackConfig:
    report_to_wandb: bool = False
    wandb_project: str = "s6e11-credit-card-fraud"
    wandb_entity: str = ""


# NOTE: Use default_factory for mutable default values to prevent unintended shared state.
@dataclass
class CFG:
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    tracking: TrackConfig = field(default_factory=TrackConfig)

    def to_dict(self):
        return asdict(self)

def load_configs() -> CFG:
    return CFG()