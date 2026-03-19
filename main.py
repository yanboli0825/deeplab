import hydra
from omegaconf import DictConfig, open_dict

from src.config import validate_app_config
from src.core.bootstrap import bootstrap_app
from src.core.cv import cv_loop
from src.core.train import train_loop


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """Hydra entrypoint.

    The Python runtime intentionally exposes only two first-class modes:
    `train` for a single execution unit and `cv` for repeated train units.
    More complex workflows should be assembled outside this file.
    """

    cfg = validate_app_config(cfg)
    bootstrap = bootstrap_app(cfg)
    with open_dict(cfg):
        cfg.runtime = {
            "output_dir": bootstrap.output_dir,
            "resolved_config_path": bootstrap.resolved_config_path,
        }

    # mode dispatch
    if cfg.mode.name == "train":
        return train_loop(cfg)
    if cfg.mode.name == "cv":
        return cv_loop(cfg)
    raise ValueError(f"Unsupported mode: {cfg.mode.name}")


if __name__ == "__main__":
    main()
