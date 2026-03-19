# Training Framework Architecture Report

## 1. Overview

This training framework is now built around four explicit design rules:

1. Configuration is composed by Hydra and consumed at runtime through a small, stable contract.
2. Python only exposes two first-class execution modes: `train` and `cv`.
3. More complex workflows such as flat CV, nested CV, HPO + refit, and metric aggregation are orchestrated outside the runtime through shell scripts.
4. Runtime outputs are standardized so that orchestration code consumes artifacts instead of reverse-engineering internal paths.

At a high level, the framework flow is:

`Hydra config -> bootstrap -> mode runner (train/cv) -> single-run runtime -> standardized artifacts`


## 2. Runtime Architecture

### 2.1 Entry Layer

The entrypoint is [`main.py`](/E:/projects/deeplab/main.py).

Its responsibility is intentionally small:

- load and validate the top-level Hydra config
- run bootstrap logic once
- dispatch to `train` or `cv`

It no longer contains experiment construction logic, logger-specific branches, or CV loop details.


### 2.2 Bootstrap Layer

The bootstrap logic lives in [`src/core/bootstrap.py`](/E:/projects/deeplab/src/core/bootstrap.py).

This layer is responsible for:

- loading `.env`
- setting the random seed
- creating the run output directory
- saving the resolved config snapshot
- snapshotting source code into the run directory

This separation keeps environment preparation independent from experiment execution.


### 2.3 Config Validation Layer

The runtime-critical schema is defined in [`src/config/schema.py`](/E:/projects/deeplab/src/config/schema.py).

This layer validates the fields the framework itself depends on, especially:

- `mode.name`
- `mode.n_folds`
- `project_name`
- `experiment_name`
- `run_name`
- `paths.output_dir`
- `monitor`

The design is intentionally partial rather than fully strict. The framework validates its own contract without blocking user-defined extension fields.


### 2.4 Mode Layer

The mode layer contains only:

- [`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)
- [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)

Their responsibilities are:

- `train`: run exactly one execution unit
- `cv`: repeat the same execution unit across folds and write an aggregate summary

This is the core architectural boundary of the framework. Anything more complex than that should be composed outside the Python mode layer.


### 2.5 Single-Run Runtime Layer

The runtime core is [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py).

This is the most important module in the framework. It is responsible for:

- building the run context
- applying runtime-specific overrides such as fold-specific output directories
- instantiating model, datamodule, logger, callbacks, and trainer
- running `trainer.fit`
- optionally running `trainer.test`
- writing the standardized `run_summary.json`

The runtime is now organized around explicit contracts instead of backend-specific string branching.


### 2.6 Artifact Contract Layer

The artifact contract is defined in [`src/core/contracts.py`](/E:/projects/deeplab/src/core/contracts.py).

There are three main objects:

- `RunContext`: immutable metadata for one concrete execution unit
- `RunSummary`: stable artifact for one `train` run
- `CvSummary`: aggregate artifact for one `cv` run

Each run now writes a stable summary artifact containing fields such as:

- `mode`
- `experiment_name`
- `run_name`
- `output_dir`
- `resolved_config_path`
- `summary_path`
- `monitor`
- `val_score`
- `test_score`
- `best_ckpt_path`
- `fold`

This contract is what shell orchestration should consume.


## 3. Data and Split Architecture

### 3.1 Split Policy

Split logic is isolated in [`src/datamodules/split.py`](/E:/projects/deeplab/src/datamodules/split.py).

This module is intentionally framework-agnostic. It only returns numpy index arrays wrapped in `SplitIndices`.

The supported policies are:

- random holdout
- group-aware holdout
- K-fold
- group-aware K-fold
- dev/test holdout
- group-aware dev/test holdout

The module now includes basic validation for:

- illegal ratios
- invalid `fold`
- invalid `n_splits`
- empty partitions caused by pathological inputs
- mismatched group lengths


### 3.2 Datamodule Boundary

The example datamodule is [`src/datamodules/dummy_dm.py`](/E:/projects/deeplab/src/datamodules/dummy_dm.py).

Its boundary is now explicit:

- the datamodule consumes split artifacts
- it does not own fold policy
- it can read a `split_file`
- it can consume injected `split_indices`
- it can fall back to a simple default split for smoke usage

This is a deliberate change from the older pattern where datamodule config also carried fold-specific policy state.


## 4. Model and Logging Architecture

### 4.1 Model Layer

The model base class is [`src/models/base_model.py`](/E:/projects/deeplab/src/models/base_model.py).

Its responsibilities are now narrower:

- task forward contract
- generic training/validation/test steps
- metric setup
- confusion matrix generation
- optimizer and scheduler construction from config

The example implementation is [`src/models/dummy_model.py`](/E:/projects/deeplab/src/models/dummy_model.py).


### 4.2 Logger Adapter Layer

Logger backend differences are isolated in [`src/loggers/__init__.py`](/E:/projects/deeplab/src/loggers/__init__.py) and the handler modules under [`src/loggers`](/E:/projects/deeplab/src/loggers).

Current handlers:

- [`src/loggers/mlflow_handler.py`](/E:/projects/deeplab/src/loggers/mlflow_handler.py)
- [`src/loggers/wandb_handler.py`](/E:/projects/deeplab/src/loggers/wandb_handler.py)
- [`src/loggers/base.py`](/E:/projects/deeplab/src/loggers/base.py)

This means `BaseModel` no longer needs to branch directly on `MLFlowLogger` or `WandbLogger`. Backend-specific figure logging is delegated to adapters.


## 5. Hydra Configuration Structure

The Hydra root config is [`conf/config.yaml`](/E:/projects/deeplab/conf/config.yaml).

The config tree is organized by concern:

- [`conf/mode`](/E:/projects/deeplab/conf/mode)
- [`conf/model`](/E:/projects/deeplab/conf/model)
- [`conf/datamodule`](/E:/projects/deeplab/conf/datamodule)
- [`conf/logger`](/E:/projects/deeplab/conf/logger)
- [`conf/callbacks`](/E:/projects/deeplab/conf/callbacks)
- [`conf/trainer`](/E:/projects/deeplab/conf/trainer)
- [`conf/paths`](/E:/projects/deeplab/conf/paths)
- [`conf/hydra`](/E:/projects/deeplab/conf/hydra)
- [`conf/hpo`](/E:/projects/deeplab/conf/hpo)

Important runtime defaults:

- `experiment_name` defaults to `project_name`
- `run_name` defaults to `mode.name`
- Hydra still controls the physical output directory layout

This keeps the framework configuration-driven while avoiding implicit `null/null` run paths.


## 6. Orchestration Strategy

The shell orchestration layer lives in [`scripts`](/E:/projects/deeplab/scripts).

Current scripts:

- [`run_flat_cv.sh`](/E:/projects/deeplab/scripts/run_flat_cv.sh)
- [`run_nested_cv.sh`](/E:/projects/deeplab/scripts/run_nested_cv.sh)
- [`run_cv_refit.sh`](/E:/projects/deeplab/scripts/run_cv_refit.sh)
- [`hpo.sh`](/E:/projects/deeplab/scripts/hpo.sh)
- [`aggregate_json_metrics.py`](/E:/projects/deeplab/scripts/aggregate_json_metrics.py)

The intended split of responsibilities is:

- Python runtime: execute one unit or a simple fold loop
- shell scripts: compose higher-order workflows
- aggregation script: consume `run_summary.json`

This is consistent with the framework goal of keeping the Python mode layer minimal.


## 7. Project Structure

The current project structure is:

```text
deeplab/
├── main.py
├── CODEX.md
├── FRAMEWORK_ARCHITECTURE.md
├── conf/
│   ├── config.yaml
│   ├── callbacks/
│   ├── datamodule/
│   ├── hpo/
│   ├── hydra/
│   ├── logger/
│   ├── mode/
│   ├── model/
│   ├── paths/
│   └── trainer/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── schema.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── bootstrap.py
│   │   ├── contracts.py
│   │   ├── cv.py
│   │   ├── runner.py
│   │   └── train.py
│   ├── datamodules/
│   │   ├── dummy_dm.py
│   │   └── split.py
│   ├── loggers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── mlflow_handler.py
│   │   └── wandb_handler.py
│   ├── models/
│   │   ├── base_model.py
│   │   └── dummy_model.py
│   └── utils/
│       ├── __init__.py
│       ├── build.py
│       └── misc.py
├── scripts/
│   ├── aggregate_json_metrics.py
│   ├── hpo.sh
│   ├── run_cv_refit.sh
│   ├── run_flat_cv.sh
│   └── run_nested_cv.sh
└── tests/
    ├── test_runtime.py
    └── test_split.py
```


## 8. Responsibilities by Directory

### `conf/`

Hydra configuration groups. This directory defines what can be swapped without editing Python code.

### `src/config/`

Framework-owned schema validation for runtime-critical config fields.

### `src/core/`

Execution semantics and runtime orchestration. This is where mode dispatch, bootstrap, summary contract, and single-run execution live.

### `src/datamodules/`

Data loading and split consumption. Split generation utilities are kept separate from datamodule implementation.

### `src/models/`

Task logic and model templates.

### `src/loggers/`

Backend adapters for MLflow and WandB specific behavior.

### `src/utils/`

Shared helpers for object construction, config persistence, hyperparameter logging, and code snapshotting.

### `scripts/`

Workflow composition outside the Python runtime boundary.

### `tests/`

Minimal regression coverage for split helpers and runtime artifact contracts.


## 9. Strengths of the Current Architecture

- The runtime boundary is now much clearer than before.
- `train` and `cv` are the only first-class Python modes.
- The framework produces standardized artifacts instead of relying on implicit path conventions.
- Split policy is separated from datamodule construction.
- Logger-specific behavior is isolated behind adapters.
- The config layout is clean and consistent with Hydra composition.
- High-order workflows are correctly pushed to scripts instead of expanding Python mode count.


## 10. Remaining Architectural Gaps

The framework is in a much better state, but it is not fully finished.

The main remaining gaps are:

- config schema is still partial, not fully structured end-to-end
- the example datamodule is still a template, not a production data pipeline
- orchestration scripts still encode workflow assumptions and are not yet abstracted into a workflow engine
- there is not yet a richer artifact index beyond `run_summary.json`
- model-level extension patterns are present but still minimal


## 11. Recommended Extension Rules

To keep the architecture stable, future changes should follow these rules:

1. Do not add new Python top-level modes unless the runtime boundary itself changes.
2. New experiment workflows should be built in shell or workflow tooling, not by adding `mode=nested_cv`-style branches.
3. New datamodules should consume split artifacts rather than reintroducing fold policy into datamodule config.
4. New logger backends should be added through adapters, not inside model code or `runner.py`.
5. New runtime outputs should extend the existing summary contract instead of creating ad hoc artifacts with implicit names.


## 12. Conclusion

The framework has now moved from a loose prototype toward a real training runtime with explicit contracts.

Its defining architectural characteristics are:

- Hydra-driven composition
- minimal Python execution surface
- explicit run artifacts
- split policy isolation
- logger backend isolation
- shell-based composition for complex workflows

This is a sound foundation for continuing toward a more complete research or production training framework.
