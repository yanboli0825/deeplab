import json

from omegaconf import OmegaConf

from src.config import validate_app_config
from src.core.cv import cv_loop
from src.core.runner import run_experiment


def _make_test_cfg(tmp_path):
    """Build a minimal runtime config used by smoke-style tests.

    Args:
        tmp_path: Temporary pytest directory used for outputs.

    Returns:
        DictConfig: Validated test configuration.
    """

    return validate_app_config(
        OmegaConf.create(
            {
                "project_name": "deeplab",
                "experiment_name": "test-exp",
                "run_name": "smoke",
                "seed": 123,
                "monitor": "val/loss",
                "test_after_train": True,
                "resume_ckpt": None,
                "mode": {"name": "train"},
                "paths": {"output_dir": str(tmp_path)},
                "artifacts": {
                    "summary_name": "run_summary.json",
                    "index_name": "artifacts.json",
                    "workflow_summary_name": "workflow_summary.json",
                    "workflow_index_name": "workflow_artifacts.json",
                    "split_manifest_name": "split_manifest.yaml",
                },
                "split": {"method": None},
                "workflow": {"root_dir": str(tmp_path / "workflows")},
                "model": {
                    "_target_": "src.models.dummy_model.DummyModel",
                    "_recursive_": False,
                    "init_args": {
                        "model_cfg": {"model_name": "dummy_model", "num_classes": 2},
                        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-4, "weight_decay": 1e-4},
                        "scheduler": {
                            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                            "T_max": 1,
                            "eta_min": 1e-6,
                        },
                    },
                },
                "datamodule": {
                    "_target_": "src.datamodules.dummy_dm.DummyDataModule",
                    "init_args": {
                        "data_cfg": {
                            "dataset_name": "dummy_dataset",
                            "batch_size": 8,
                            "num_workers": 0,
                            "num_samples": 48,
                        }
                    },
                },
                "logger": {
                    "items": [
                        {
                            "_target_": "lightning.pytorch.loggers.CSVLogger",
                            "save_dir": str(tmp_path),
                            "name": "csv",
                        }
                    ]
                },
                "callbacks": {
                    "items": {
                        "model_checkpoint": {
                            "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                            "dirpath": "${paths.output_dir}/checkpoints",
                            "filename": "best",
                            "monitor": "${monitor}",
                            "mode": "min",
                            "save_last": True,
                            "save_top_k": 1,
                            "auto_insert_metric_name": False,
                        }
                    }
                },
                "trainer": {
                    "_target_": "lightning.pytorch.Trainer",
                    "init_args": {
                        "accelerator": "cpu",
                        "devices": 1,
                        "max_epochs": 1,
                        "enable_progress_bar": False,
                        "log_every_n_steps": 1,
                        "limit_train_batches": 2,
                        "limit_val_batches": 2,
                        "limit_test_batches": 2,
                        "num_sanity_val_steps": 0,
                        "default_root_dir": "${paths.output_dir}",
                    },
                },
                "hydra": {},
                "hpo": None,
            }
        )
    )


def test_run_experiment_writes_run_summary_and_artifact_index(tmp_path) -> None:
    """Verify that one training unit writes both summary and artifact index files.

    Args:
        tmp_path: Temporary pytest directory used for outputs.

    Returns:
        None: Assertions validate the runtime contract.
    """

    cfg = _make_test_cfg(tmp_path)
    result = run_experiment(cfg)

    assert result.context.summary_path.endswith("run_summary.json")
    with open(result.context.summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    with open(result.context.artifact_index_path, "r", encoding="utf-8") as f:
        artifact_payload = json.load(f)

    assert payload["mode"] == "train"
    assert payload["run_name"] == "smoke"
    assert "val_score" in payload
    assert payload["best_metrics"]["monitor"] == "val/loss"
    assert "epoch" in payload["best_metrics"]
    assert "val" in payload["best_metrics"]
    assert "loss" in payload["best_metrics"]["val"]
    assert "test" in payload["best_metrics"]
    assert payload["artifact_index_path"].endswith("artifacts.json")
    assert artifact_payload["metrics"]["summary_path"].endswith("run_summary.json")


def test_cv_loop_writes_aggregate_summary_and_artifact_index(tmp_path) -> None:
    """Verify that CV mode writes aggregate workflow artifacts.

    Args:
        tmp_path: Temporary pytest directory used for outputs.

    Returns:
        None: Assertions validate the aggregate runtime contract.
    """

    cfg = _make_test_cfg(tmp_path)
    cfg.mode.name = "cv"
    cfg.mode.n_folds = 2

    score = cv_loop(cfg)
    assert isinstance(score, float)

    with open(tmp_path / "workflow_summary.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    with open(tmp_path / "workflow_artifacts.json", "r", encoding="utf-8") as f:
        artifact_payload = json.load(f)

    assert payload["mode"] == "cv"
    assert payload["n_folds"] == 2
    assert len(payload["fold_summary_paths"]) == 2
    assert len(payload["fold_artifact_paths"]) == 2
    assert len(artifact_payload["children"]) == 2
