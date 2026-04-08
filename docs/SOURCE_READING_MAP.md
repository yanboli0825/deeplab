# Source Reading Map

## Summary

This document is for people reading the codebase for the first time.
It does not try to explain everything. It answers:

- which file owns which responsibility
- what to read first
- what to look for in each file

If you have not built the call chain yet, start with [`docs/CALL_FLOW.md`](E:/projects/deeplab/docs/CALL_FLOW.md).

## 1. Read the main chain first

Recommended order:

1. [`main.py`](E:/projects/deeplab/main.py)
2. [`src/config/schema.py`](E:/projects/deeplab/src/config/schema.py)
3. [`src/core/bootstrap.py`](E:/projects/deeplab/src/core/bootstrap.py)
4. [`src/core/train.py`](E:/projects/deeplab/src/core/train.py)
5. [`src/core/runner.py`](E:/projects/deeplab/src/core/runner.py)
6. [`src/utils/build.py`](E:/projects/deeplab/src/utils/build.py)

What you should understand after this pass:

- `main.py` is only the entrypoint
- `run_experiment()` is the smallest runtime unit
- `build.py` owns instantiate boundaries
- runtime consumes standardized Hydra config, not raw YAML structure

## 2. Read datamodule and model contracts next

Recommended order:

1. [`src/datamodules/base_dm.py`](E:/projects/deeplab/src/datamodules/base_dm.py)
2. [`src/datamodules/tabular_dm.py`](E:/projects/deeplab/src/datamodules/tabular_dm.py)
3. [`src/datamodules/datasets/tabular_dataset.py`](E:/projects/deeplab/src/datamodules/datasets/tabular_dataset.py)
4. [`src/models/base_model.py`](E:/projects/deeplab/src/models/base_model.py)
5. [`src/models/tasks/tabular_classification.py`](E:/projects/deeplab/src/models/tasks/tabular_classification.py)
6. [`src/models/backbones/tabular_mlp.py`](E:/projects/deeplab/src/models/backbones/tabular_mlp.py)
7. [`src/models/heads/classification_head.py`](E:/projects/deeplab/src/models/heads/classification_head.py)

What to pay attention to:

- datamodules consume split results and build dataloaders
- task models own forward/loss/metrics/optimizer behavior
- the framework no longer forces datamodules to expose summary metadata such as `input_shape` or `label_space`

## 3. Read workflow code last

Recommended order:

1. [`src/core/contracts.py`](E:/projects/deeplab/src/core/contracts.py)
2. [`src/core/cv.py`](E:/projects/deeplab/src/core/cv.py)
3. [`src/workflows/common.py`](E:/projects/deeplab/src/workflows/common.py)
4. [`src/workflows/flat_cv.py`](E:/projects/deeplab/src/workflows/flat_cv.py)
5. [`src/workflows/nested_cv.py`](E:/projects/deeplab/src/workflows/nested_cv.py)
6. [`src/workflows/hpo_refit.py`](E:/projects/deeplab/src/workflows/hpo_refit.py)

What to understand here:

- `cv` is repeated `run_experiment()` plus aggregation
- workflows orchestrate multiple runtime runs through subprocesses
- workflow code reads child artifacts instead of reimplementing runtime logic

## 4. Read split code with the new semantics in mind

Relevant files:

- [`src/datamodules/split.py`](E:/projects/deeplab/src/datamodules/split.py)
- [`src/utils/build.py`](E:/projects/deeplab/src/utils/build.py)

Important ideas:

- split policy is owned outside the datamodule
- `stratified_*` methods resolve `label_column` from `split.data_file`
- `stratified_group_*` methods resolve both `label_column` and `group_id_column`
- `stratified_group_*` must fail early if a requested split cannot keep all classes present in every split

## 5. Use config as the map back to code

When you see a field in config, trace it back as follows:

- `_target_` tells you which Python class is instantiated
- `init_args` tells you which constructor arguments matter
- runtime values such as `fold` are injected by the framework, not directly by raw YAML

Useful config files:

1. [`conf/config.yaml`](E:/projects/deeplab/conf/config.yaml)
2. [`conf/model/cpath/tabular_classification.yaml`](E:/projects/deeplab/conf/model/cpath/tabular_classification.yaml)
3. [`conf/datamodule/cpath/tabular_manifest.yaml`](E:/projects/deeplab/conf/datamodule/cpath/tabular_manifest.yaml)
4. [`conf/trainer/default.yaml`](E:/projects/deeplab/conf/trainer/default.yaml)
5. [`conf/callbacks/default.yaml`](E:/projects/deeplab/conf/callbacks/default.yaml)
6. [`conf/logger/mlflow.yaml`](E:/projects/deeplab/conf/logger/mlflow.yaml)

## 6. Practical reading strategy

If you only have a short time:

1. read `main.py` and `runner.py`
2. read `build.py`
3. read one datamodule and one task model
4. then read `cv.py` and workflow code if needed

This order keeps you focused on the main path before branching into orchestration.
