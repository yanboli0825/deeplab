# Development Guide

## Summary

This document is for people extending the framework with their own project code. The goal is not to explain every implementation detail, but to clarify:

- what should be changed
- what should not be changed
- which boundary each layer owns

The current framework is intentionally small at runtime. The main execution units are only `train` and `cv`. More complex experiment orchestration belongs to `src/workflows/`, not to the runtime entrypoint.

## 1. Add a model

Recommended structure:

```text
src/models/
|-- backbones/
|-- heads/
`-- tasks/
```

Recommended split of responsibility:

- `backbones/`: feature extraction
- `heads/`: task-specific prediction head
- `tasks/`: LightningModule-level task logic

Recommended practice:

- inherit the common training scaffold from [`src/models/base_model.py`](E:/projects/deeplab/src/models/base_model.py)
- keep the task model focused on network assembly and `forward()`
- configure optimizer and scheduler through Hydra config

Avoid:

- writing output-directory logic inside the model
- checking whether the runtime is `cv` inside the model
- branching on logger backend inside the model

## 2. Add data support

Recommended structure:

```text
src/datamodules/
|-- datasets/
|-- transforms/
|-- manifests/
`-- <your_datamodule>.py
```

Recommended split of responsibility:

- dataset: per-sample reading
- datamodule: read manifest, resolve split, build dataloaders
- split provider: define holdout / kfold / group / stratified split policy

Recommended practice:

- prepare a manifest first
- let the datamodule consume `split_provider` or `_default_split()`
- let the datamodule write the actual split artifact used in the run

Avoid:

- hard-coding fold logic inside the datamodule
- moving nested CV control flow into the datamodule
- letting the dataset decide train/val/test itself
- reintroducing `split_indices` or `split_file` as public entry points

Reference implementations:

- [`src/datamodules/tabular_dm.py`](E:/projects/deeplab/src/datamodules/tabular_dm.py)
- [`src/datamodules/datasets/tabular_dataset.py`](E:/projects/deeplab/src/datamodules/datasets/tabular_dataset.py)

## 3. Add configuration

New components should be introduced by adding config, not by changing runtime entrypoints.

Typical config groups:

- `conf/model/<your_model>.yaml`
- `conf/datamodule/<your_dm>.yaml`
- `conf/experiment/<your_experiment>.yaml`

The framework expects the following object contract:

- `model._target_ + model.init_args`
- `datamodule._target_ + datamodule.init_args`
- `trainer._target_ + trainer.init_args`

`experiment` config is responsible for combining:

- model
- datamodule
- logger
- trainer
- callbacks
- split

## 4. When to add a workflow

Use a workflow when the requirement is orchestration across multiple runtime runs, not a new training primitive.

Typical workflow-level needs:

- nested CV
- HPO + refit
- multi-group candidate comparison
- batch execution driven by shell scripts

Do not push the following into runtime:

- adding another `main.py` mode for a one-off experiment pattern
- embedding candidate selection into `runner.py`
- mixing workflow control flow into model or datamodule code

## 5. Recommended onboarding order

The cleanest way to add a new project is:

1. prepare the manifest
2. implement the dataset
3. implement the datamodule
4. implement the task model
5. add Hydra config
6. validate with `mode=train`
7. expand to `mode=cv`
8. add workflow only if orchestration is needed

This order keeps the smallest executable unit easy to debug.

## 6. Debugging checklist

If a run does not behave as expected, check in this order:

1. whether `config.yaml` matches the intended override
2. whether `run_summary.json` has the expected monitor / score / checkpoint path
3. whether `artifacts.json` has valid data artifact and checkpoint paths
4. whether the datamodule wrote the expected `split_manifest.yaml`
5. whether the model and datamodule agree on input and output shapes

Do not start by editing `main.py`. Most issues are not there.
