from typing import Any, Dict

import wandb
import subprocess


def get_commit_hash() -> Any:
    """Return the current short Git commit hash, if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    

def get_model_config(model: Any, framework: str = "sklearn") -> Dict[str, Any]:
    """Extract model configuration from the specified framework.

    Args:
        model (Any): The model instance to extract configuration from.
        framework (str, optional): The machine learning framework used. Defaults to "sklearn".

    Returns:
        Dict[str, Any]: Dictionary containing model parameters.

    Raises:
        NotImplementedError: If the framework is not supported.
    """

    if framework == "sklearn":
        return {
            key: value
            for key, value in model.get_params(deep=True).items()
            if isinstance(value, (str, int, float, bool, type(None), tuple, list))
        }

    raise NotImplementedError(f"Unsupported framework: {framework}")


def init_run(
        project: str, 
        name: str, 
        group: str, 
        config: Dict, 
        model: Any=None, 
        framework="sklearn"
    ) -> wandb.run:
    """Initialize a Weights & Biases (wandb) tracking run with metadata and configuration.

    Args:
        project (str): The W&B project name.
        name (str): The name of the specific run.
        group (str): The group name to aggregate related runs.
        config (Dict): Dictionary containing hyperparameters and configuration parameters.
        model (Any, optional): The model instance to extract configuration from. Defaults to None.
        framework (str, optional): The machine learning framework used (currently supports "sklearn"). Defaults to "sklearn".

    Returns:
        wandb.run: The active W&B run object.
    """

    run_config = {**config, "git_commit": get_commit_hash()}

    if model is not None:
        run_config["model_config"] = get_model_config(model, framework)
        run_config["model_class"] = model.__class__.__name__

    return wandb.init(
        project=project,
        name=name,
        group=group,
        config=run_config,
    )
