# 框架架构说明

## 总览

这个仓库是一个基于 Hydra + PyTorch Lightning 的训练框架。它的 runtime 只保留两个一级执行单元：

- `train`
- `cv`

更复杂的实验流程放在 `src/workflows/` 中编排，不会继续往 runtime 入口里堆新的模式。

这样做的目的很明确：

- runtime surface 保持小而稳定
- 配置是主要控制面
- split policy、数据读取、任务逻辑、日志、workflow 编排各自分层

## 分层设计

### 1. 入口层

- [`main.py`](E:/projects/deeplab/main.py)

职责：

- 组合 Hydra 配置
- 校验框架拥有的配置契约
- 初始化运行目录和输出路径
- 只分发到 `train` 或 `cv`

这里不放训练细节、不放 fold 循环、不放对象实例化逻辑。

### 2. 配置与构建层

- [`src/config/schema.py`](E:/projects/deeplab/src/config/schema.py)
- [`src/utils/build.py`](E:/projects/deeplab/src/utils/build.py)

职责：

- 把 Hydra 配置标准化为框架自己的 contract
- 校验内建 mode
- 把 `_target_ + init_args` 转成 instantiate 友好的对象配置
- 构建 split provider、datamodule、model、callbacks、logger、trainer

框架当前标准化的配置面包括：

- `model._target_ + model.init_args`
- `datamodule._target_ + datamodule.init_args`
- `trainer._target_ + trainer.init_args`
- `logger.items`
- `callbacks.items`
- `split.method + split.data_file + split.group_id_column + split.label_column`

### 3. runtime 层

- [`src/core/train.py`](E:/projects/deeplab/src/core/train.py)
- [`src/core/cv.py`](E:/projects/deeplab/src/core/cv.py)
- [`src/core/runner.py`](E:/projects/deeplab/src/core/runner.py)

职责：

- 构造一次具体运行的 `RunContext`
- 注入 runtime-only 配置
- 解析最终执行态配置
- 调用 Lightning 的 `fit()` 和可选 `test()`
- 写出 `run_summary.json` 和 `artifacts.json`

[`src/core/runner.py`](E:/projects/deeplab/src/core/runner.py) 里的 `run_experiment()` 是整个框架最小的运行单元。

### 4. 扩展层

- [`src/datamodules/`](E:/projects/deeplab/src/datamodules)
- [`src/models/`](E:/projects/deeplab/src/models)
- [`src/loggers/`](E:/projects/deeplab/src/loggers)

职责：

- datamodule：读取数据、消费 split、返回 dataloader
- model：定义任务语义、损失、指标、优化器
- logger adapter：隔离 MLflow / WandB 一类后端差异

推荐的代码组织方式：

```text
src/models/
|-- backbones/
|-- heads/
`-- tasks/

src/datamodules/
|-- datasets/
|-- transforms/
|-- manifests/
`-- <your_datamodule>.py
```

### 5. workflow 层

- [`src/workflows/common.py`](E:/projects/deeplab/src/workflows/common.py)
- [`src/workflows/flat_cv.py`](E:/projects/deeplab/src/workflows/flat_cv.py)
- [`src/workflows/nested_cv.py`](E:/projects/deeplab/src/workflows/nested_cv.py)
- [`src/workflows/hpo_refit.py`](E:/projects/deeplab/src/workflows/hpo_refit.py)

职责：

- 组织多次 runtime run
- 通过 subprocess 调用 `main.py`
- 收集子运行的 summary 和 artifacts
- 写出 workflow 级输出

workflow 是编排层，不是新的 runtime mode。

## 核心契约

### RunContext

[`src/core/contracts.py`](E:/projects/deeplab/src/core/contracts.py) 里的 `RunContext` 描述一次具体运行的上下文：

- mode
- experiment name
- run name
- output directory
- config path
- summary path
- artifact index path
- optional fold

### RunSummary

`RunSummary` 是单次运行的高频结果摘要，主要给 train / cv / workflow 消费：

- monitor name
- `val_score`
- `last_val_score`
- optional `test_score`
- best checkpoint path

### ArtifactIndex

`ArtifactIndex` 是单次运行的文件与句柄索引，包含：

- config paths
- checkpoints
- metrics
- data artifacts
- figures
- logger identifiers

workflow 层通过 child artifact path 进行索引，不会把所有子运行内容复制一遍。

## split 与 datamodule 的边界

这套框架明确把 split policy 从 datamodule 中分离出来：

- split policy 由 [`src/datamodules/split.py`](E:/projects/deeplab/src/datamodules/split.py) 负责
- datamodule 只消费 `SplitIndices` 或 `SplitProvider`
- datamodule 可以保留本地 `_default_split()` 作为 fallback，但它不是 cross-validation 的 owner

新的 split 语义包括：

- `stratified_*` 方法从 `split.data_file + split.label_column` 解析标签
- `stratified_group_*` 方法额外从 `split.data_file + split.group_id_column` 解析 group
- `stratified_group_*` 不是“松散近似分层”，而是：
  - group 不泄漏
  - 每个 split 必须含全部类别
  - 标签分布尽量接近整体分布
  - 样本量平衡是次级目标

如果数据在给定 `n_folds` 或 `test_ratio` 下无法满足这些约束，split 阶段应该直接失败，而不是把问题留到 metrics 阶段。

## 推荐扩展顺序

如果你要接入自己的项目，建议按这个顺序：

1. 准备 manifest
2. 实现 dataset
3. 实现 datamodule
4. 实现 task model
5. 添加 Hydra config
6. 先验证 `train`
7. 再扩展到 `cv`
8. 最后才考虑 workflow

这个顺序的核心价值是：先把最小执行单元做稳，再做编排。
