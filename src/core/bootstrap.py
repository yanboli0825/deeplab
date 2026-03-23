"""Bootstrap helpers kept outside the mode dispatch layer."""

import os
import typing
from dataclasses import dataclass

import torch
import lightning as L
import omegaconf
from omegaconf import DictConfig, OmegaConf

from src.utils.misc import load_dotenv, snapshot_code, write_resolved_config


@dataclass
class BootstrapArtifacts:
    """Artifacts created once for the Hydra invocation."""

    output_dir: str
    resolved_config_path: str
    artifact_index_path: str


def bootstrap_app(cfg: DictConfig) -> BootstrapArtifacts:
    """Prepare the runtime environment before mode dispatch.

    Args:
        cfg: Resolved Hydra configuration for the current invocation.

    Returns:
        BootstrapArtifacts: Paths to bootstrap artifacts created once per invocation.
    """

    load_dotenv()
    L.seed_everything(int(cfg.get("seed", 42)), workers=True)
    # torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig, omegaconf.base.ContainerMetadata, typing.Any])


    output_dir = cfg.paths.output_dir
    os.makedirs(output_dir, exist_ok=True)

    resolved_config_path = os.path.join(output_dir, "config.yaml")
    artifact_index_path = os.path.join(
        output_dir,
        str(cfg.artifacts.get("index_name", "artifacts.json")),
    )
    write_resolved_config(cfg, resolved_config_path)
    snapshot_code(output_dir)

    return BootstrapArtifacts(
        output_dir=output_dir,
        resolved_config_path=resolved_config_path,
        artifact_index_path=artifact_index_path,
    )
