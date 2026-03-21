from omegaconf import DictConfig

from src.core.runner import run_experiment


def train_loop(cfg: DictConfig) -> float:
    """Run one minimal training execution unit.

    Args:
        cfg: Resolved runtime configuration for the current training run.

    Returns:
        float: Test score when `test_after_train=true`, otherwise validation score.

    Raises:
        ValueError: Raised when test mode is requested but no test score is produced.
    """

    result = run_experiment(cfg)

    if cfg.get("test_after_train", False):
        if result.summary.test_score is not None:
            return result.summary.test_score
        else:
            raise ValueError("Test score was not computed.")

    return result.summary.val_score
