# Framework Architecture

## Summary

This repository is a Hydra-driven training framework built on top of PyTorch Lightning. The runtime intentionally exposes only two first-class execution units:

- `train`: one concrete training run
- `cv`: repeated `train`-style runs across folds

Higher-level experiment patterns such as flat cross-validation, nested cross-validation, or HPO + refit are implemented outside the runtime in `src/workflows/` and launched through thin scripts in `scripts/`.

The design goal is to keep the runtime surface small, make configuration the primary control plane, and keep split policy, data loading, task logic, logging, and workflow orchestration cleanly separated.

## Layered Design

### 1. Entry Layer

- [`main.py`](/E:/projects/deeplab/main.py)

Responsibilities:

- compose Hydra config
- validate the framework-owned config surface
- bootstrap runtime paths and output directories
- dispatch only to `train` or `cv`

This file deliberately avoids training logic, fold loops, and object construction.

### 2. Config and Build Layer

- [`src/config/schema.py`](/E:/projects/deeplab/src/config/schema.py)
- [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)

Responsibilities:

- normalize Hydra config into a stable framework contract
- validate the built-in mode contract
- convert `_target_ + init_args` plugin configs into instantiate-ready object configs
- build split providers, datamodules, models, callbacks, loggers, and trainers

The framework-owned config surface is standardized as:

- `model._target_ + model.init_args`
- `datamodule._target_ + datamodule.init_args`
- `trainer._target_ + trainer.init_args`
- `logger.items`
- `callbacks.items`

### 3. Runtime Layer

- [`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)
- [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)
- [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)

Responsibilities:

- build one concrete `RunContext`
- inject runtime-only values into a local config
- resolve the final execution config
- run `trainer.fit()` and optional `trainer.test()`
- write `run_summary.json` and `artifacts.json`

`run_experiment()` in [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py) is the smallest runtime execution unit in the framework.

### 4. Extension Layer

- [`src/datamodules/`](/E:/projects/deeplab/src/datamodules)
- [`src/models/`](/E:/projects/deeplab/src/models)
- [`src/loggers/`](/E:/projects/deeplab/src/loggers)

Responsibilities:

- datamodules consume split results and expose dataloaders
- task models define forward/loss/metrics/optimizer behavior
- logger adapters isolate backend-specific logging details

Recommended model organization:

```text
src/models/
|-- backbones/
|-- heads/
`-- tasks/
```

Recommended data organization:

```text
src/datamodules/
|-- datasets/
|-- transforms/
|-- manifests/
`-- <your_datamodule>.py
```

### 5. Workflow Layer

- [`src/workflows/common.py`](/E:/projects/deeplab/src/workflows/common.py)
- [`src/workflows/flat_cv.py`](/E:/projects/deeplab/src/workflows/flat_cv.py)
- [`src/workflows/nested_cv.py`](/E:/projects/deeplab/src/workflows/nested_cv.py)
- [`src/workflows/hpo_refit.py`](/E:/projects/deeplab/src/workflows/hpo_refit.py)

Responsibilities:

- orchestrate multiple runtime runs
- call `main.py` through subprocesses
- collect child summaries and artifact indexes
- write workflow-level outputs

Workflows are orchestration layers, not new runtime modes.

## Main Contracts

### RunContext

Defined in [`src/core/contracts.py`](/E:/projects/deeplab/src/core/contracts.py), `RunContext` captures one concrete run:

- mode
- experiment name
- run name
- output directory
- config path
- summary path
- artifact index path
- optional fold

### RunSummary

The stable high-frequency output of one run. It stores:

- monitor name
- `val_score`
- `last_val_score`
- optional `test_score`
- best checkpoint path

`val_score` is the best checkpoint score, not the last validation value. The last observed validation value is preserved separately in `last_val_score`.

### ArtifactIndex

The stable file and handle index for one run. It covers:

- config paths
- checkpoints
- metrics
- data artifacts
- figures
- logger identifiers

The workflow layer uses the same contract and points to child artifacts through `children`.

## Split and Data Contract

The framework keeps split policy outside datamodules.

- split policy lives in [`src/datamodules/split.py`](/E:/projects/deeplab/src/datamodules/split.py)
- datamodules consume `SplitIndices` or a `SplitProvider`
- datamodules may still define a local fallback split, but they are not the owner of cross-validation semantics

This separation allows the same datamodule to be reused in:

- one train run
- CV mode
- higher-level workflows

## Recommended Usage Pattern

When adapting a new project, the expected path is:

1. prepare a manifest file
2. implement a dataset
3. implement a datamodule that consumes split results
4. implement a task model
5. define Hydra config groups
6. validate with `mode=train`
7. scale to `mode=cv` or workflows

The framework is designed to make this path straightforward without modifying the runtime entrypoint.
