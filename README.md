# DeepLab Training Framework

Hydra + PyTorch Lightning based training framework with two runtime units only:

- `train`: one concrete training run
- `cv`: repeated training runs across folds

Higher-level experiment flows are implemented in a Python workflow layer and exposed through thin shell launchers.

## Core Principles

- Hydra drives construction and runtime parameters.
- The runtime only exposes `train` and `cv`.
- Split policy is independent from datamodules.
- Every run writes `run_summary.json` and `artifacts.json`.
- Every workflow writes `workflow_summary.json` and `workflow_artifacts.json`.

## Recommended Code Layout

The recommended extension shape is:

```text
src/
|-- models/
|   |-- backbones/
|   |-- heads/
|   `-- tasks/
`-- datamodules/
    |-- datasets/
    |-- transforms/
    `-- manifests/
```

This repository already includes reference implementations for that layout:

- [`src/models/backbones/tabular_mlp.py`](/E:/projects/deeplab/src/models/backbones/tabular_mlp.py)
- [`src/models/heads/classification_head.py`](/E:/projects/deeplab/src/models/heads/classification_head.py)
- [`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)
- [`src/datamodules/datasets/tabular_dataset.py`](/E:/projects/deeplab/src/datamodules/datasets/tabular_dataset.py)
- [`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)
- [`src/datamodules/manifests/schema.py`](/E:/projects/deeplab/src/datamodules/manifests/schema.py)

## Recommended Data Layout

Prefer manifest-driven data access over implicit directory scanning.

Example:

```text
data/
`-- manifests/
    `-- example_tabular.csv
```

Recommended manifest columns:

- `sample_id`
- `label`
- `group`
- `path` when samples live on disk

Reference file:

- [`example_tabular.csv`](/E:/projects/deeplab/data/manifests/example_tabular.csv)

## Minimal Usage

Single train run:

```bash
python main.py mode=train
```

Single CV run:

```bash
python main.py mode=cv mode.n_folds=5
```

Run the reference tabular baseline:

```bash
python main.py +experiment=tabular_baseline
```

Override model or trainer parameters:

```bash
python main.py \
  +experiment=tabular_baseline \
  model.init_args.optimizer.lr=3e-4 \
  trainer.init_args.max_epochs=20
```

## Runtime Outputs

Each run writes artifacts under Hydra's output directory.

- `config.yaml`: resolved config snapshot
- `code/`: source snapshot
- `run_summary.json`: stable high-frequency summary
- `artifacts.json`: artifact index for data, checkpoints, figures, and logger handles
- `checkpoints/`: Lightning checkpoints

CV and workflow aggregations additionally write:

- `workflow_summary.json`
- `workflow_artifacts.json`

## Config Structure

The root config is [`conf/config.yaml`](/E:/projects/deeplab/conf/config.yaml).

Main config groups:

- `mode`
- `artifacts`
- `split`
- `workflow`
- `model`
- `datamodule`
- `logger`
- `callbacks`
- `trainer`
- `paths`
- `hydra`
- `hpo`
- `experiment`

The framework-owned object surface follows:

- `model._target_` + `model.init_args`
- `datamodule._target_` + `datamodule.init_args`
- `trainer._target_` + `trainer.init_args`
- `logger.items`
- `callbacks.items`

## Workflow Entry Points

Flat CV:

```bash
bash scripts/run_flat_cv.sh --experiment-name my_exp --run-prefix flatcv --n-folds 5
```

Nested CV:

```bash
bash scripts/run_nested_cv.sh --experiment-name my_exp --split-root splits/nested
```

HPO + refit:

```bash
bash scripts/hpo.sh --experiment-name my_exp
```

## Related Docs

- [`FRAMEWORK_ARCHITECTURE.md`](/E:/projects/deeplab/FRAMEWORK_ARCHITECTURE.md)
- [`FRAMEWORK_ARCHITECTURE_ZH.md`](/E:/projects/deeplab/FRAMEWORK_ARCHITECTURE_ZH.md)
- [`DEVELOPMENT_GUIDE.md`](/E:/projects/deeplab/DEVELOPMENT_GUIDE.md)
