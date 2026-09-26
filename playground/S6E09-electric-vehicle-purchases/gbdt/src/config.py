from pathlib import Path

import torch
from dataclasses import dataclass, asdict, field

@dataclass
class EnvConfig:
    competition: str = "playground-series-s6e9"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    smoke_test: bool = True
    report_to_wandb: bool = False

# TODO: consider adding a ProcessingConfig  for stuff like feature_selection, folds, use_te
@dataclass
class DataConfig:
    data_path: str = ""
    extra_path: str | None = None
    target: str = "will_buy_ev"
    metric: str = "roc_auc_score"
    evals_directory: Path = Path("evals")
    folds: int = 5
    add_extra_data: bool = False
    add_features: bool = False
    feature_selection: bool = False
    use_te: bool = False # Target Encoding

@dataclass
class ModelConfig:
    task_type: str = "classification"
    model_type: str = "tree-based"
    hp_search: bool = False
    search_weights: bool = True
    optuna_trials: int = 30

@dataclass
class TrackConfig:
    wandb_project: str = "s6e09-electric-vehicle-purchases"
    wandb_entity: str = ""

@dataclass
# NOTE: Use default_factory for mutable default values to prevent unintended shared state.
class CFG:
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackConfig = field(default_factory=TrackConfig)

    def to_dict(self):
        return asdict(self)

def load_configs() -> CFG:
    return CFG()