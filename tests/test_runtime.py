import json

from omegaconf import OmegaConf

from src.core.cv import cv_loop
from src.core.runner import run_experiment


def _make_test_cfg(tmp_path):
    return OmegaConf.create(
        {
            "project_name": "deeplab",
            "experiment_name": "test-exp",
            "run_name": "smoke",
            "seed": 123,
            "monitor": "val/loss",
            "test_after_train": True,
            "resume_ckpt": None,
            "mode": {"name": "train", "n_folds": 2},
            "paths": {"output_dir": str(tmp_path), "data_dir": "data"},
            "model": {
                "_target_": "src.models.dummy_model.DummyModel",
                "_recursive_": False,
                "model_cfg": {"model_name": "dummy_model", "num_classes": 2},
                "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-4, "weight_decay": 1e-4},
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                    "T_max": 1,
                    "eta_min": 1e-6,
                },
            },
            "datamodule": {
                "_target_": "src.datamodules.dummy_dm.DummyDataModule",
                "data_cfg": {
                    "dataset_name": "dummy_dataset",
                    "batch_size": 8,
                    "num_workers": 0,
                    "total_samples": 48,
                },
            },
            "logger": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": str(tmp_path),
                "name": "csv",
            },
            "callbacks": {
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
            },
            "trainer": {
                "_target_": "lightning.pytorch.Trainer",
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
            "hydra": {},
            "hpo": None,
        }
    )


def test_run_experiment_writes_run_summary(tmp_path) -> None:
    cfg = _make_test_cfg(tmp_path)
    result = run_experiment(cfg)

    assert result.context.summary_path.endswith("run_summary.json")
    with open(result.context.summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["mode"] == "train"
    assert payload["run_name"] == "smoke"
    assert "val_score" in payload
    assert payload["resolved_config_path"].endswith("config.yaml")


def test_cv_loop_writes_aggregate_summary(tmp_path) -> None:
    cfg = _make_test_cfg(tmp_path)
    cfg.mode.name = "cv"
    cfg.mode.n_folds = 2

    score = cv_loop(cfg)
    assert isinstance(score, float)

    with open(tmp_path / "run_summary.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["mode"] == "cv"
    assert payload["n_folds"] == 2
    assert len(payload["fold_summary_paths"]) == 2
