# Development Guide

## Summary

这份文档面向要在当前框架上扩展自己项目的人。重点不是解释架构，而是说明：

- 应该改哪里
- 不应该改哪里
- 新增 model、datamodule、config 时推荐遵守什么边界

## 1. 新增模型

推荐结构：

```text
src/models/
|-- backbones/
|-- heads/
`-- tasks/
```

建议分工：

- `backbones/`：特征提取
- `heads/`：任务输出头
- `tasks/`：LightningModule 级别任务

推荐做法：

- task model 继承 [`src/models/base_model.py`](/E:/projects/deeplab/src/models/base_model.py) 的通用训练骨架
- 在 task model 中只补网络组装和 `forward()`
- optimizer 和 scheduler 通过 config 注入

不推荐做法：

- 在 model 中写输出目录逻辑
- 在 model 中判断当前是不是 cv
- 在 model 中分支判断不同 logger backend

参考实现：

- [`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)

## 2. 新增数据接入

推荐结构：

```text
src/datamodules/
|-- datasets/
|-- transforms/
|-- manifests/
`-- <your_datamodule>.py
```

建议分工：

- dataset：单样本读取
- datamodule：读取 manifest、消费 split、构建 dataloader
- split provider：定义 holdout / kfold / group kfold 等策略

推荐做法：

- 先准备 manifest
- datamodule 消费 `split_indices` 或 `split_provider`
- datamodule 通过 `emit_data_artifacts()` 写 split 和 dataset 摘要

不推荐做法：

- 在 datamodule 内部写死 fold 语义
- 在 datamodule 内部判断 nested-cv
- 让 dataset 自己决定 train/val/test 分割

参考实现：

- [`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)
- [`src/datamodules/datasets/tabular_dataset.py`](/E:/projects/deeplab/src/datamodules/datasets/tabular_dataset.py)

## 3. 新增配置

新增组件时，优先补配置，而不是改 runtime。

你通常需要新增：

- `conf/model/<your_model>.yaml`
- `conf/datamodule/<your_dm>.yaml`
- `conf/experiment/<your_experiment>.yaml`

配置应遵守：

- `model._target_ + model.init_args`
- `datamodule._target_ + datamodule.init_args`
- `trainer._target_ + trainer.init_args`

experiment 配置负责组合：

- model
- datamodule
- logger
- trainer
- 可选 callbacks

参考：

- [`conf/model/cpath/tabular_classification.yaml`](/E:/projects/deeplab/conf/model/cpath/tabular_classification.yaml)
- [`conf/datamodule/cpath/tabular_manifest.yaml`](/E:/projects/deeplab/conf/datamodule/cpath/tabular_manifest.yaml)
- [`conf/experiment/tabular_baseline.yaml`](/E:/projects/deeplab/conf/experiment/tabular_baseline.yaml)

## 4. 什么时候该改 workflow

当需求属于“多次训练的组合逻辑”时，应改 workflow，不应改 runtime。

典型属于 workflow 的需求：

- nested CV
- HPO + refit
- 多组 candidate 对比
- shell 驱动的批量实验

典型不应放到 runtime 的需求：

- 给 `main.py` 增加新的训练 mode
- 在 `runner.py` 中内嵌 candidate selection
- 在 datamodule/model 中混入 workflow 控制流

## 5. 推荐开发顺序

建议按下面顺序接项目：

1. 准备数据 manifest
2. 写 dataset
3. 写 datamodule
4. 写 task model
5. 写 config
6. 跑 `mode=train`
7. 跑 `mode=cv`
8. 最后再做 workflow

这样做的原因是：最小执行单元总是最容易定位问题。

## 6. 调试建议

如果训练没有按预期工作，优先按下面顺序检查：

1. `config.yaml` 是否符合预期
2. `run_summary.json` 的 `monitor`、`val_score`、`best_ckpt_path` 是否合理
3. `artifacts.json` 中 data artifacts 和 checkpoint 路径是否完整
4. datamodule 的 `dataset_summary.json` 和 `split_manifest.yaml` 是否正确
5. model 与 datamodule 是否遵守了输入输出边界

不要先去改 `main.py`。大多数问题不在那里。
