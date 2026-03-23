# 框架架构说明

## 总览

这个仓库是一套基于 Hydra + PyTorch Lightning 的训练框架。它有一个非常明确的架构选择：

- Python runtime 只保留两个一级执行单元：`train` 和 `cv`
- 更复杂的实验流程放到 `src/workflows/` 和 `scripts/` 中组合执行

这意味着框架不会继续在 `main.py` 里增加新的训练模式，而是把复杂度上收至 workflow 层，把基础运行时保持稳定。

## 分层设计

### 1. 入口层

- [`main.py`](/E:/projects/deeplab/main.py)

职责：

- 组合 Hydra 配置
- 调用 `validate_app_config()` 标准化框架配置
- 调用 bootstrap 准备运行目录和基础产物路径
- 根据 `cfg.mode.name` 分发到 `train` 或 `cv`

这里有意不放训练细节、不放 fold 循环、不放对象实例化逻辑。

### 2. 配置与构建层

- [`src/config/schema.py`](/E:/projects/deeplab/src/config/schema.py)
- [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)

职责：

- 把 Hydra 原始配置标准化为框架自己的 contract
- 校验 `train` / `cv` 这两个内建 mode
- 把 `_target_ + init_args` 结构转成 Hydra instantiate 能消费的对象配置
- 构建 split provider、model、datamodule、logger、callbacks、trainer

框架自己约束的配置接口是：

- `model._target_ + model.init_args`
- `datamodule._target_ + datamodule.init_args`
- `trainer._target_ + trainer.init_args`
- `logger.items`
- `callbacks.items`

### 3. runtime 层

- [`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)
- [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)
- [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)

职责：

- 生成一次具体运行的 `RunContext`
- 注入 runtime-only 配置
- resolve 最终执行态配置
- 调用 Lightning 的 `fit` 和可选 `test`
- 写 `run_summary.json` 和 `artifacts.json`

其中 [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py) 的 `run_experiment()` 是整个框架最小、最核心的训练执行单元。

### 4. 扩展层

- [`src/datamodules/`](/E:/projects/deeplab/src/datamodules)
- [`src/models/`](/E:/projects/deeplab/src/models)
- [`src/loggers/`](/E:/projects/deeplab/src/loggers)

职责：

- datamodule：读取数据、消费 split、返回 dataloader
- model：定义任务语义、损失、指标、优化器配置
- logger adapter：隔离 MLflow / WandB 之类 backend 的差异

推荐代码组织方式：

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

- [`src/workflows/common.py`](/E:/projects/deeplab/src/workflows/common.py)
- [`src/workflows/flat_cv.py`](/E:/projects/deeplab/src/workflows/flat_cv.py)
- [`src/workflows/nested_cv.py`](/E:/projects/deeplab/src/workflows/nested_cv.py)
- [`src/workflows/hpo_refit.py`](/E:/projects/deeplab/src/workflows/hpo_refit.py)

职责：

- 编排多次 runtime 运行
- 通过 subprocess 调用 `main.py`
- 读取子运行的 summary 和 artifacts
- 写 workflow 级别的输出

这里的关键是：workflow 不是新的 runtime mode，而是建立在 `train/cv` 之上的上层调度。

## 核心契约

### RunContext

`RunContext` 定义一次具体运行的上下文：

- mode
- experiment name
- run name
- output directory
- resolved config path
- summary path
- artifact index path
- 可选 fold

### RunSummary

`RunSummary` 是单次运行的高频摘要，主要给 train/cv/workflow 消费。

它包含：

- monitor
- `val_score`
- `last_val_score`
- 可选 `test_score`
- `best_ckpt_path`

其中：

- `val_score` 是 best checkpoint 对应的监控分数
- `last_val_score` 是最后一轮验证值，用于诊断

### ArtifactIndex

`ArtifactIndex` 是完整 artifact 索引，包含：

- config
- checkpoints
- metrics
- data artifacts
- figures
- logger ids
- workflow children

它的作用不是给人直接阅读，而是给脚本和 workflow 稳定消费。

## split 与 datamodule 的边界

这套框架明确把 split policy 从 datamodule 中分离出来：

- split policy 在 [`src/datamodules/split.py`](/E:/projects/deeplab/src/datamodules/split.py)
- datamodule 只消费 `SplitIndices` 或 `SplitProvider`
- datamodule 可以有默认 split fallback，但它不是 CV 语义的 owner

这样 datamodule 不需要知道：

- 当前是不是 CV
- 当前是 outer fold 还是 inner fold
- workflow 在做什么

它只需要知道当前该消费哪组 train/val/test indices。

## 推荐扩展路径

如果你要把自己的项目接入这套框架，推荐路径是：

1. 先整理 manifest
2. 写 dataset
3. 写 datamodule
4. 写 task model
5. 写 Hydra config
6. 先验证 `train`
7. 再扩展到 `cv` 或 workflow

这个顺序能最大限度降低调试复杂度，因为你始终是在验证最小执行单元，而不是一开始就进入复杂编排。
