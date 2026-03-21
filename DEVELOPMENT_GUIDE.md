# Development Guide

## 1. Design Rules

When extending this framework, follow these rules first:

1. Do not add new Python top-level modes unless the runtime boundary itself changes.
2. Prefer Hydra composition over hard-coded branching.
3. Datamodules consume split artifacts. They do not own split policy.
4. Logger-specific behavior belongs in adapters, not in model code or `runner.py`.
5. Workflow orchestration belongs in `src/workflows`, not in ad hoc shell logic.

## 2. Recommended Project Shape

Use the following layout for real projects:

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

Recommended responsibilities:

- `models/backbones`: reusable feature extractors
- `models/heads`: task heads
- `models/tasks`: Lightning task modules
- `datamodules/datasets`: sample-level reading
- `datamodules/transforms`: optional preprocessing and augmentation
- `datamodules/manifests`: manifest column schema and helpers

## 3. How to Add a New Model

### Step 1

Add reusable network pieces under:

- [`src/models/backbones`](/E:/projects/deeplab/src/models/backbones)
- [`src/models/heads`](/E:/projects/deeplab/src/models/heads)

### Step 2

Add the task-level Lightning module under:

- [`src/models/tasks`](/E:/projects/deeplab/src/models/tasks)

Use [`tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py) as the reference structure.

### Step 3

Create a Hydra config under [`conf/model`](/E:/projects/deeplab/conf/model).

The preferred shape is:

```yaml
_target_: src.models.tasks.your_task.YourTaskModel
_recursive_: false

init_args:
  model_cfg: {...}
  metrics_cfg: {...}
  optimizer: {...}
  scheduler: {...}
```

### Rule

Do not put split policy, logger branching, artifact naming, or output path logic into model code.

## 4. How to Add a New Datamodule

### Step 1

Add sample reading logic under [`src/datamodules/datasets`](/E:/projects/deeplab/src/datamodules/datasets).

### Step 2

Add the datamodule under [`src/datamodules`](/E:/projects/deeplab/src/datamodules).

Use [`tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py) as the reference implementation.

### Step 3

Keep the constructor compatible with the runtime builder:

```python
def __init__(self, data_cfg, split_indices=None, split_provider=None, *args, **kwargs):
    ...
```

### Step 4

Build train/val/test partitions in `setup()` by consuming split artifacts instead of deriving fold logic locally.

### Rule

Do not put `fold`, `n_splits`, `outer_fold`, or `inner_fold` into datamodule behavior.

## 5. Recommended Data Entry Contract

Prefer a manifest file over implicit directory scanning.

Example CSV:

```text
sample_id,label,group,f0,f1,f2,f3
s001,0,p001,0.10,0.20,0.30,0.40
```

Recommended columns:

- `sample_id`
- `label`
- `group`
- `path` when data lives on disk

See [`schema.py`](/E:/projects/deeplab/src/datamodules/manifests/schema.py) for the recommended manifest column contract.

## 6. Hydra Config Conventions

Model config:

- [`tabular_classification.yaml`](/E:/projects/deeplab/conf/model/cpath/tabular_classification.yaml)

Datamodule config:

- [`tabular_manifest.yaml`](/E:/projects/deeplab/conf/datamodule/cpath/tabular_manifest.yaml)

Experiment config:

- [`tabular_baseline.yaml`](/E:/projects/deeplab/conf/experiment/tabular_baseline.yaml)

Recommended ownership:

- `conf/model`: architecture and optimization
- `conf/datamodule`: data reading parameters
- `conf/experiment`: project-level experiment bundles

## 7. What Belongs in Runtime Contracts

Extend runtime contracts only when orchestration or downstream consumers need stable access.

Typical examples:

- best checkpoint path
- summary path
- artifact index path
- data artifact paths
- fold id

## 8. Anti-Patterns to Avoid

Do not introduce these patterns:

- model code branching on logger type
- datamodule code choosing fold strategy
- workflow logic hidden inside datamodule or model
- dataset parsing hard-coded directly inside task models
- relying on implicit directory names instead of manifest or artifact contracts

## 9. Reference Files

- [`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)
- [`src/models/backbones/tabular_mlp.py`](/E:/projects/deeplab/src/models/backbones/tabular_mlp.py)
- [`src/models/heads/classification_head.py`](/E:/projects/deeplab/src/models/heads/classification_head.py)
- [`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)
- [`src/datamodules/datasets/tabular_dataset.py`](/E:/projects/deeplab/src/datamodules/datasets/tabular_dataset.py)
- [`src/datamodules/manifests/schema.py`](/E:/projects/deeplab/src/datamodules/manifests/schema.py)
