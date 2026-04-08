# Call Flow Guide

## Summary

This document explains three things:

1. what happens after `python main.py ...`
2. how `train`, `cv`, and workflows differ
3. where runtime outputs are written

## 1. Overall Call Chain

The framework always returns to the same smallest runtime unit:

```text
Hydra CLI
  -> main.py
  -> validate_app_config()
  -> bootstrap_app()
  -> train_loop() or cv_loop()
     -> run_experiment()
        -> build_split_provider()
        -> build_model()
        -> build_datamodule()
        -> build_logger()
        -> build_callbacks()
        -> build_trainer()
        -> trainer.fit()
        -> optional trainer.test()
        -> write run_summary.json
        -> write artifacts.json
```

## 2. `train` Path

Command:

```bash
python main.py mode=train
```

Call order:

```text
main.py:main
  -> validate_app_config(cfg)
  -> bootstrap_app(cfg)
  -> train_loop(cfg)
  -> run_experiment(cfg)
```

Key files:

- [`main.py`](E:/projects/deeplab/main.py)
- [`src/core/train.py`](E:/projects/deeplab/src/core/train.py)
- [`src/core/runner.py`](E:/projects/deeplab/src/core/runner.py)

`run_experiment()` is the smallest runtime execution unit. `train_loop()` is only a thin wrapper.

## 3. `cv` Path

Command:

```bash
python main.py mode=cv mode.n_folds=5
```

Call order:

```text
main.py:main
  -> validate_app_config()
  -> bootstrap_app()
  -> cv_loop(cfg)
     -> for fold in range(n_folds):
          run_experiment(cfg, fold=fold)
     -> aggregate fold results
     -> write workflow_summary.json
     -> write workflow_artifacts.json
```

Key files:

- [`src/core/cv.py`](E:/projects/deeplab/src/core/cv.py)
- [`src/core/runner.py`](E:/projects/deeplab/src/core/runner.py)

`cv` is repeated `run_experiment()` plus aggregation. It is not a separate training framework.

## 4. Workflow Path

Higher-level experiment orchestration is implemented in:

- [`src/workflows/flat_cv.py`](E:/projects/deeplab/src/workflows/flat_cv.py)
- [`src/workflows/nested_cv.py`](E:/projects/deeplab/src/workflows/nested_cv.py)
- [`src/workflows/hpo_refit.py`](E:/projects/deeplab/src/workflows/hpo_refit.py)

Common helpers:

- [`src/workflows/common.py`](E:/projects/deeplab/src/workflows/common.py)

Workflow call pattern:

```text
workflow main()
  -> generate overrides
  -> run_main(overrides)
     -> subprocess: python main.py ...
  -> read child summaries and artifacts
  -> aggregate
  -> write workflow outputs
```

Workflows use runtime as a child process. They do not become a new runtime mode.

## 5. Where Outputs Are Written

### Single run

[`src/core/runner.py`](E:/projects/deeplab/src/core/runner.py) writes:

- `config.yaml`
- `run_summary.json`
- `artifacts.json`

### CV / workflow aggregation

[`src/core/cv.py`](E:/projects/deeplab/src/core/cv.py) and [`src/workflows/common.py`](E:/projects/deeplab/src/workflows/common.py) write:

- `workflow_summary.json`
- `workflow_artifacts.json`

The aggregation layer keeps child output paths as references rather than duplicating all contents.

## 6. Split Stage Reminder

The split stage is part of runtime setup. It happens before dataloader construction and before `trainer.fit()`.

For the current framework:

- `split.provider` is the primary split entry
- `stratified_*` methods resolve labels from `split.data_file + split.label_column`
- `stratified_group_*` methods additionally resolve group ids from `split.data_file + split.group_id_column`
- `stratified_group_*` split should fail early if the requested split cannot keep every class present in every split

That means split failures should happen before metric calculation, not after the model starts running.
