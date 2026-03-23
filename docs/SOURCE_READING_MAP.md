# Source Reading Map

## Summary

这份文档是给第一次读源码的人用的。它不回答“代码怎么跑起来”，而是回答：

- 每个文件在架构里的位置是什么
- 读这个文件时应先抓什么，再抓什么
- 哪些函数和变量最值得盯住

如果你还没建立整体调用链，先看 [`docs/CALL_FLOW.md`](/E:/projects/deeplab/docs/CALL_FLOW.md)。

## 1. 第一轮：先读主链

目标：先建立一次 `train` 运行是怎么串起来的。

建议顺序：

1. [`main.py`](/E:/projects/deeplab/main.py)
2. [`src/config/schema.py`](/E:/projects/deeplab/src/config/schema.py)
3. [`src/core/bootstrap.py`](/E:/projects/deeplab/src/core/bootstrap.py)
4. [`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)
5. [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)
6. [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)

### 读完这一轮后，你应该建立的认知

- `main.py` 只是入口，不做训练细节
- `run_experiment()` 是最小训练执行单元
- `build.py` 负责 instantiate 边界
- runtime 消费的是标准化后的 Hydra config

## 2. 第二轮：再读对象层

目标：理解 datamodule 和 model 是如何被 runtime 使用的。

建议顺序：

1. [`src/datamodules/base_dm.py`](/E:/projects/deeplab/src/datamodules/base_dm.py)
2. [`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)
3. [`src/datamodules/datasets/tabular_dataset.py`](/E:/projects/deeplab/src/datamodules/datasets/tabular_dataset.py)
4. [`src/models/base_model.py`](/E:/projects/deeplab/src/models/base_model.py)
5. [`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)
6. [`src/models/backbones/tabular_mlp.py`](/E:/projects/deeplab/src/models/backbones/tabular_mlp.py)
7. [`src/models/heads/classification_head.py`](/E:/projects/deeplab/src/models/heads/classification_head.py)

### [`src/datamodules/base_dm.py`](/E:/projects/deeplab/src/datamodules/base_dm.py)

先看：

- 构造函数参数
- `resolve_split()`
- `emit_data_artifacts()`
- `dataset_summary()`

要抓住的点：

- datamodule 消费 split，而不是拥有 split policy
- runtime 会把 artifact 输出配置注入到 `data_cfg.runtime`

### [`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)

先看：

- `prepare_data()`
- `setup()`
- `_default_split()`
- `train_dataloader() / val_dataloader() / test_dataloader()`

要抓住的点：

- manifest 如何被读入
- split 如何被解析成 train/val/test rows
- dataset summary 如何扩展

### [`src/models/base_model.py`](/E:/projects/deeplab/src/models/base_model.py)

先看：

- `__init__`
- `training_step()`
- `validation_step()`
- `test_step()`
- `configure_optimizers()`

要抓住的点：

- 通用训练语义放在 base class
- task model 只需要补网络语义
- optimizer / scheduler 仍然由 config 驱动

### [`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)

先看：

- `__init__`
- `forward()`

要抓住的点：

- task model 如何组合 backbone 和 head
- task model 为什么不关心 split、output dir、workflow

## 3. 第三轮：最后读聚合与 workflow

目标：理解 `cv` 和 workflow 如何在不破坏最小执行单元边界的前提下完成复杂实验。

建议顺序：

1. [`src/core/contracts.py`](/E:/projects/deeplab/src/core/contracts.py)
2. [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)
3. [`src/workflows/common.py`](/E:/projects/deeplab/src/workflows/common.py)
4. [`src/workflows/flat_cv.py`](/E:/projects/deeplab/src/workflows/flat_cv.py)
5. [`src/workflows/nested_cv.py`](/E:/projects/deeplab/src/workflows/nested_cv.py)
6. [`src/workflows/hpo_refit.py`](/E:/projects/deeplab/src/workflows/hpo_refit.py)

### [`src/core/contracts.py`](/E:/projects/deeplab/src/core/contracts.py)

先看：

- `RunContext`
- `RunSummary`
- `ArtifactIndex`
- `CvSummary`
- `WorkflowSummary`

要抓住的点：

- 单次 run 和聚合层的 contract 如何区分
- 为什么聚合层通过 child artifact path 做索引，而不是猜目录

### [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)

先看：

- fold loop
- `scores` / `test_scores`
- `CvSummary`

要抓住的点：

- `cv` 本质上仍然是多次 `run_experiment()`
- 它新增的是聚合语义，而不是一套新的训练内核

### [`src/workflows/common.py`](/E:/projects/deeplab/src/workflows/common.py)

先看：

- `run_main()`
- `latest_json()`
- `latest_path()`
- `write_workflow_outputs()`

要抓住的点：

- workflow 为什么通过 subprocess 调 `main.py`
- workflow 如何消费 child run outputs

## 4. 配置文件如何配合阅读

源码阅读时，建议同时打开这些配置：

1. [`conf/config.yaml`](/E:/projects/deeplab/conf/config.yaml)
2. [`conf/model/cpath/tabular_classification.yaml`](/E:/projects/deeplab/conf/model/cpath/tabular_classification.yaml)
3. [`conf/datamodule/cpath/tabular_manifest.yaml`](/E:/projects/deeplab/conf/datamodule/cpath/tabular_manifest.yaml)
4. [`conf/trainer/default.yaml`](/E:/projects/deeplab/conf/trainer/default.yaml)
5. [`conf/callbacks/default.yaml`](/E:/projects/deeplab/conf/callbacks/default.yaml)
6. [`conf/logger/mlflow.yaml`](/E:/projects/deeplab/conf/logger/mlflow.yaml)

对照方法：

- 看到一个 Python 构造函数，就回头看它对应的 `_target_`
- 看到一个对象字段，就回头看它来自哪个 `init_args`
- 看到一个 runtime 行为，就回头看是否由 config 控制

## 5. 最实用的阅读策略

如果你只想最短时间读懂：

1. 先读 [`main.py`](/E:/projects/deeplab/main.py) 和 [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)
2. 再读 [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)
3. 然后选一个 datamodule 和一个 task model 读通
4. 最后再看 `cv` 和 workflow

这样先掌握的是主干，而不是枝节。
