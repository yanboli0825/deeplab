# DeepLab Training Framework

Hydra + PyTorch Lightning + MLflow/WandB based training framework.

The framework is built around two Python runtime modes only:

- `train`: one minimal training execution unit
- `cv`: repeated training units across folds

More complex workflows such as flat CV, nested CV, and HPO + refit are composed outside the runtime through scripts.


## 1. Core Principles

- Hydra drives object construction and runtime parameters.
- Python only exposes `train` and `cv`.
- Complex workflows belong to shell orchestration.
- Every run should produce a stable `run_summary.json`.


## 2. Project Layout

```text
deeplab/
├── main.py
├── conf/
├── src/
├── scripts/
├── tests/
├── CODEX.md
├── FRAMEWORK_ARCHITECTURE.md
├── FRAMEWORK_ARCHITECTURE_ZH.md
└── DEVELOPMENT_GUIDE.md
```

Important directories:

- [`conf`](/E:/projects/deeplab/conf): Hydra config groups
- [`src/core`](/E:/projects/deeplab/src/core): bootstrap, mode dispatch, runtime, artifact contracts
- [`src/datamodules`](/E:/projects/deeplab/src/datamodules): datamodules and split helpers
- [`src/models`](/E:/projects/deeplab/src/models): model base class and example model
- [`src/loggers`](/E:/projects/deeplab/src/loggers): backend adapters
- [`scripts`](/E:/projects/deeplab/scripts): orchestration scripts
- [`tests`](/E:/projects/deeplab/tests): minimal regression tests


## 3. Minimal Usage

### Single train run

```bash
python main.py mode=train
```

### Single CV run

```bash
python main.py mode=cv mode.n_folds=5
```

### Override model or trainer parameters

```bash
python main.py \
  mode=train \
  experiment_name=my_exp \
  run_name=trial_001 \
  model.optimizer.lr=3e-4 \
  trainer.max_epochs=20
```


## 4. Runtime Outputs

Each run writes artifacts under Hydra's output directory.

Important files:

- `config.yaml`: resolved config snapshot
- `code/`: source snapshot
- `run_summary.json`: stable runtime summary
- `checkpoints/`: Lightning checkpoints

`run_summary.json` is the artifact contract used by shell scripts and aggregation tools.


## 5. Config Structure

The root config is [`conf/config.yaml`](/E:/projects/deeplab/conf/config.yaml).

Main config groups:

- `mode`
- `model`
- `datamodule`
- `logger`
- `callbacks`
- `trainer`
- `paths`
- `hydra`
- `hpo`

Default runtime values:

- `experiment_name: ${project_name}`
- `run_name: ${mode.name}`


## 6. Available Scripts

### Flat CV style orchestration

[`scripts/run_flat_cv.sh`](/E:/projects/deeplab/scripts/run_flat_cv.sh)

Example:

```bash
bash scripts/run_flat_cv.sh my_exp flatcv 5 0.2 3e-4 1e-4
```

### Nested CV style orchestration

[`scripts/run_nested_cv.sh`](/E:/projects/deeplab/scripts/run_nested_cv.sh)

This script expects split manifests such as:

```text
splits/nested/outer_0/inner_0.yaml
splits/nested/outer_0/inner_1.yaml
splits/nested/outer_0/refit.yaml
```

### HPO entry

[`scripts/hpo.sh`](/E:/projects/deeplab/scripts/hpo.sh)

### CV search + manual refit helper

[`scripts/run_cv_refit.sh`](/E:/projects/deeplab/scripts/run_cv_refit.sh)

### Aggregate multiple summaries

[`scripts/aggregate_json_metrics.py`](/E:/projects/deeplab/scripts/aggregate_json_metrics.py)

Example:

```bash
python scripts/aggregate_json_metrics.py "outputs/my_exp/**/run_summary.json" val_score
```


## 7. Extension Entry Points

If you want to extend the framework:

- add models under [`src/models`](/E:/projects/deeplab/src/models)
- add datamodules under [`src/datamodules`](/E:/projects/deeplab/src/datamodules)
- add split policies in [`src/datamodules/split.py`](/E:/projects/deeplab/src/datamodules/split.py)
- add logger adapters under [`src/loggers`](/E:/projects/deeplab/src/loggers)

Detailed guidance:

- architecture report: [`FRAMEWORK_ARCHITECTURE.md`](/E:/projects/deeplab/FRAMEWORK_ARCHITECTURE.md)
- 中文架构说明: [`FRAMEWORK_ARCHITECTURE_ZH.md`](/E:/projects/deeplab/FRAMEWORK_ARCHITECTURE_ZH.md)
- developer guide: [`DEVELOPMENT_GUIDE.md`](/E:/projects/deeplab/DEVELOPMENT_GUIDE.md)


## 8. Recommended Workflow

1. Use `train` to validate a single execution unit.
2. Use `cv` when you need fold-level repetition inside the runtime.
3. Use scripts for flat CV, nested CV, HPO, and refit workflows.
4. Consume `run_summary.json` instead of depending on implicit internal paths.
