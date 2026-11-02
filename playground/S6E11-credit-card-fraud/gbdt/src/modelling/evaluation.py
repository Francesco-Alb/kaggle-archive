import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import optuna

def plot_fold_scores(fold_scores: Dict[str, List[float]], save_path: Optional[Path|str] = None):
    """Plot and optionally save score distribution across folds."""

    if isinstance(save_path, str):
        save_path = Path(save_path)

    scores_df = pd.DataFrame(fold_scores)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=scores_df, palette="mako", orient='h')
    plt.title('Score Distribution Across Folds', fontsize=16, weight='bold')
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Models', fontsize=14)
    plt.tight_layout()

    if save_path and not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()

def create_ensemble_predictions(
    test_preds: dict[str, np.ndarray],
    weight_study: Optional[optuna.Study] = None,
    use_weights: bool = False
) -> np.ndarray:
    """
    Creates an ensemble of model probabilities.
    Automatically handles weights optimization (last weight is implicit).
    """
    model_names = list(test_preds.keys())
    
    if use_weights and weight_study is not None and weight_study.best_params:
        weights = weight_study.best_params.copy()
        # Add the last weight that was implicit in the optimization
        assigned_sum = sum(weights.values())
        last_model = [m for m in model_names if m not in weights][0]
        weights[last_model] = max(0.0, 1.0 - assigned_sum)
        print(f"Using optimized weights: {weights}")
    else:
        weights = {model_name: 1.0 / len(model_names) for model_name in model_names}
        print(f"Using equal weights: {weights}")

    # Weighted ensemble of model probabilities
    ensemble_probabilities = sum(
        weights[model_name] * test_preds[model_name] 
        for model_name in model_names
    )
    
    return ensemble_probabilities
