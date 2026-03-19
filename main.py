import os
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from typing import List
import sys
from src.utils.utils import load_dotenv, snapshot_code
from src.core.train import train_loop
from src.core.cv import cv_loop


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    # =================================================================================
    # 加载环境变量
    # =================================================================================
    load_dotenv()

    # =================================================================================
    # 设置随机种子
    # =================================================================================
    L.seed_everything(cfg.get('seed', 42), workers=True)

    # =================================================================================
    # 保存实验配置和代码快照，便于复现
    # =================================================================================
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    with open(os.path.join(cfg.paths.output_dir, "config.yaml"), "w", encoding="utf-8") as f:
        OmegaConf.save(config=cfg, f=f, resolve=True)

    snapshot_code(cfg.paths.output_dir)

    # =================================================================================
    # 实验运行模式入口
    # =================================================================================
    # train -> validate -> test (optional)
    if cfg.mode.name == 'train':
        return train_loop(cfg)
    # cross validation only on validation set
    # if the averaged performance on test set is needed, use multirun on train mode to retrieve
    elif cfg.mode.name == 'cv':
        return cv_loop(cfg)
    else:
        return float('inf')


if __name__ == "__main__":
    main()
