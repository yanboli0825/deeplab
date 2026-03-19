# Development Guide

## 1. Design Rules

When extending this framework, follow these rules first:

1. Do not add new Python top-level modes unless the framework boundary itself changes.
2. Prefer Hydra config composition over hard-coded runtime branching.
3. Datamodules consume split artifacts. They should not own split policy.
4. Logger-specific behavior belongs in adapters, not in model code or `runner.py`.
5. Shell scripts orchestrate complex workflows. Python runtime executes minimal units.


## 2. How to Add a New Model

### Step 1

Create a model file under [`src/models`](/E:/projects/deeplab/src/models).

The model should either:

- inherit from [`BaseModel`](/E:/projects/deeplab/src/models/base_model.py)
- or implement a custom `LightningModule` if the task semantics are very different

### Step 2

Create a Hydra config under [`conf/model`](/E:/projects/deeplab/conf/model).

The config should define:

- `_target_`
- task-specific model config
- optimizer config
- scheduler config

### Step 3

Run with overrides such as:

```bash
python main.py model=<your_model_group>
```

### Rule

Do not put logger branching, artifact naming, or output path logic into model code.


## 3. How to Add a New Datamodule

### Step 1

Create the datamodule under [`src/datamodules`](/E:/projects/deeplab/src/datamodules).

### Step 2

Accept `split_indices` and optionally `split_file` as inputs.

That means your datamodule constructor should be compatible with the runtime builder:

```python
def __init__(self, data_cfg, split_indices=None, *args, **kwargs):
    ...
```

### Step 3

Consume the split to build dataset partitions in `setup()`.

### Rule

Do not reintroduce `fold`, `n_splits`, or split policy decisions into datamodule config unless the datamodule is the explicit owner of a standalone dataset preparation workflow.


## 4. How to Add a New Logger Backend

### Step 1

Add a new handler under [`src/loggers`](/E:/projects/deeplab/src/loggers).

### Step 2

Implement the same interface as [`BaseLoggerHandler`](/E:/projects/deeplab/src/loggers/base.py).

### Step 3

Register the Lightning logger type and handler in [`src/loggers/__init__.py`](/E:/projects/deeplab/src/loggers/__init__.py).

### Rule

Do not add backend-specific `if isinstance(logger, ...)` logic into:

- [`src/models/base_model.py`](/E:/projects/deeplab/src/models/base_model.py)
- [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)


## 5. How to Add a New Split Policy

### Step 1

Add the policy to [`src/datamodules/split.py`](/E:/projects/deeplab/src/datamodules/split.py).

### Step 2

Keep it framework-agnostic:

- input is metadata or candidate indices
- output is `SplitIndices`

### Step 3

If the policy needs to be persisted, store it as a split manifest and let the datamodule read it through `split_file`.

### Rule

Do not bury split rules inside:

- shell scripts only
- datamodule internals only
- mode-specific ad hoc logic


## 6. What Belongs in Config

Use Hydra config for things that define object construction or runtime parameters:

- model class and hyperparameters
- datamodule class and data parameters
- logger backend selection
- trainer parameters
- callbacks
- monitor metric
- whether to run test after training


## 7. What Belongs in Shell Orchestration

Use shell scripts for workflow composition such as:

- running multiple folds as separate train jobs
- nested CV
- HPO search followed by refit
- aggregation across multiple `run_summary.json` files
- comparisons between multiple candidate override sets

If the logic is “repeat multiple minimal runs and compare outputs”, it usually belongs in scripts.


## 8. What Belongs in Runtime Contracts

Add data to the runtime contract only when orchestration code needs to consume it reliably.

Examples:

- best checkpoint path
- resolved config path
- val/test metrics
- fold id
- output directory

If a script depends on some value, that value should usually be written into `run_summary.json`.


## 9. Anti-Patterns to Avoid

Do not introduce these patterns again:

- adding `mode=nested_cv` or similar high-order Python modes
- branching on logger `_target_` inside runtime code
- putting split ownership back into datamodules
- making scripts depend on hidden file names that are not part of the artifact contract
- mixing bootstrap logic with experiment execution logic
- letting model code know too much about external experiment backends


## 10. Recommended Workflow for New Features

1. Decide whether the feature belongs to config, runtime, datamodule/model, or shell orchestration.
2. If scripts need to consume the result, extend the runtime contract first.
3. Keep the Python mode surface unchanged unless absolutely necessary.
4. Add or update a minimal test around split behavior or artifact contracts.


## 11. Reference Files

Useful files to follow when extending the framework:

- [`main.py`](/E:/projects/deeplab/main.py)
- [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)
- [`src/core/contracts.py`](/E:/projects/deeplab/src/core/contracts.py)
- [`src/datamodules/dummy_dm.py`](/E:/projects/deeplab/src/datamodules/dummy_dm.py)
- [`src/datamodules/split.py`](/E:/projects/deeplab/src/datamodules/split.py)
- [`src/models/base_model.py`](/E:/projects/deeplab/src/models/base_model.py)
- [`src/loggers/__init__.py`](/E:/projects/deeplab/src/loggers/__init__.py)
- [`scripts/run_flat_cv.sh`](/E:/projects/deeplab/scripts/run_flat_cv.sh)
- [`scripts/run_nested_cv.sh`](/E:/projects/deeplab/scripts/run_nested_cv.sh)
