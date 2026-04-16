# Training Service

This directory contains a thin HTTP service layer around the deeplab training engine.

## Scope

- Does not modify `src/` or `conf/`
- Accepts HTTP job requests as JSON dicts
- Adapts requests to framework CLI overrides
- Runs `main.py` in background subprocesses
- Persists job metadata under `service/runtime`

## Install

```bash
pip install -e .[service]
```

## Run

```bash
uvicorn service.app.main:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health`
- `POST /jobs/train`
- `POST /jobs/cv`
- `POST /jobs/hpo`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/summary`
- `GET /jobs/{job_id}/logs`
- `POST /jobs/{job_id}/cancel`

## Request shape

```json
{
  "platform": {
    "project_name": "cpath-demo",
    "submitted_by": "user_a",
    "resource_request": {
      "gpu_count": 1,
      "gpu_ids": [0]
    }
  },
  "dataset": {
    "manifest_path": "/data/manifest.csv"
  },
  "framework": {
    "experiment_name": "demo-exp",
    "run_name": "manual-run",
    "test_after_train": true,
    "mode": {
      "name": "train"
    },
    "model": {
      "_target_": "src.models.cpath.abmil.ABMIL",
      "init_args": {
        "model_name": "abmil",
        "model_cfg": {
          "num_classes": 2
        }
      }
    },
    "datamodule": {
      "_target_": "src.datamodules.frozen_section_dm.FrozenSectionDataModule",
      "init_args": {
        "data_cfg": {
          "batch_size": 1
        }
      }
    },
    "split": {
      "method": "stratified_group_holdout",
      "group_id_column": "patient_id",
      "label_column": "label",
      "test_ratio": 0.1,
      "seed": 42
    },
    "trainer": {
      "_target_": "lightning.pytorch.Trainer",
      "init_args": {
        "max_epochs": 5,
        "devices": [0]
      }
    }
  }
}
```
