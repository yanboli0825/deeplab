# Training Framework 架构说明

## 1. 架构目标

当前训练框架围绕四个核心目标设计：

1. 通过 Hydra 组合配置，保持代码与配置分离。
2. Python 层只保留两个一级执行模式：`train` 和 `cv`。
3. 复杂流程通过 shell 脚本编排，而不是继续堆叠新的 Python mode。
4. 运行产物必须标准化，供上层脚本稳定消费。


## 2. 整体执行流程

当前框架的主执行链路是：

`Hydra 配置 -> bootstrap -> mode(train/cv) -> 单次运行 runtime -> 标准化产物`

对应的职责分层如下：

- 入口层：[`main.py`](/E:/projects/deeplab/main.py)
- 启动层：[`src/core/bootstrap.py`](/E:/projects/deeplab/src/core/bootstrap.py)
- 模式层：[`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)、[`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)
- 运行层：[`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)
- 契约层：[`src/core/contracts.py`](/E:/projects/deeplab/src/core/contracts.py)
- 配置校验层：[`src/config/schema.py`](/E:/projects/deeplab/src/config/schema.py)


## 3. 各层职责

### 3.1 入口层

[`main.py`](/E:/projects/deeplab/main.py) 现在只做三件事：

- 校验运行时关键配置
- 执行 bootstrap
- 根据 `cfg.mode.name` 分发到 `train` 或 `cv`

入口层不再负责：

- logger 特判
- fold 循环
- trainer/model/datamodule 的实例化
- artifact 命名规则


### 3.2 Bootstrap 层

[`src/core/bootstrap.py`](/E:/projects/deeplab/src/core/bootstrap.py) 负责一次 Hydra 调用只做一次的初始化动作：

- 加载 `.env`
- 设置随机种子
- 创建输出目录
- 保存解析后的配置文件
- 复制代码快照

这保证了环境准备与实验运行本身分离。


### 3.3 Mode 层

Mode 层只保留两个文件：

- [`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)
- [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)

其中：

- `train` 表示一次最小训练执行单元
- `cv` 表示对同一执行单元做多次 fold 重复，并写出聚合结果

这就是当前框架最重要的边界。像 `flat-cv`、`nested-cv`、`cv-refit`、`hpo+refit` 这类流程，都不应该再做成新的 Python 一级 mode。


### 3.4 Runtime 层

[`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py) 是当前框架的核心。

它负责：

- 构建当前运行的 `RunContext`
- 注入运行期配置
- 实例化 model/datamodule/logger/callbacks/trainer
- 执行 `fit`
- 可选执行 `test`
- 生成单次运行的 `run_summary.json`

相比旧结构，当前 runtime 的改进点是：

- 不再根据 logger `_target_` 写死分支
- 不再把命名规则散落在多个文件里
- fold-specific 输出路径通过 `RunContext` 显式决定


### 3.5 契约层

[`src/core/contracts.py`](/E:/projects/deeplab/src/core/contracts.py) 定义了当前框架的运行期契约：

- `RunContext`
- `RunSummary`
- `CvSummary`

这是 shell 编排层与 Python runtime 之间的稳定接口。

单次运行 summary 里会包含：

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

`cv` 模式还会额外输出聚合 summary，包括每个 fold 的 summary 路径和各 fold 分数。


## 4. 数据与划分架构

### 4.1 Split 工具层

[`src/datamodules/split.py`](/E:/projects/deeplab/src/datamodules/split.py) 现在是纯工具模块，不依赖 Lightning。

支持的能力：

- holdout
- group holdout
- k-fold
- group k-fold
- dev/test holdout
- group dev/test holdout

它的输出统一是 `SplitIndices`，只描述索引，不掺杂 datamodule 行为。

这样做的价值是：

- split policy 可以独立测试
- datamodule 不需要再定义 fold 规则
- shell 编排也可以直接消费 split manifest


### 4.2 Datamodule 边界

[`src/datamodules/dummy_dm.py`](/E:/projects/deeplab/src/datamodules/dummy_dm.py) 体现了当前 datamodule 的边界设计：

- datamodule 只消费 split
- split 可以来自 `split_indices`
- 也可以来自 `split_file`
- 如果两者都没有，再使用默认 split 兜底

这意味着 datamodule 不再承担 split policy owner 的角色，只负责数据准备、dataset 构造和 dataloader 输出。


## 5. 模型与日志架构

### 5.1 模型层

模型基类是 [`src/models/base_model.py`](/E:/projects/deeplab/src/models/base_model.py)。

当前它主要负责：

- forward 接口约束
- 通用 train/val/test step
- metric 管理
- confusion matrix 生成
- optimizer/scheduler 从 Hydra 配置构建

示例模型是 [`src/models/dummy_model.py`](/E:/projects/deeplab/src/models/dummy_model.py)。


### 5.2 Logger 适配层

Logger 差异通过 [`src/loggers`](/E:/projects/deeplab/src/loggers) 下的 adapter 处理：

- [`src/loggers/base.py`](/E:/projects/deeplab/src/loggers/base.py)
- [`src/loggers/mlflow_handler.py`](/E:/projects/deeplab/src/loggers/mlflow_handler.py)
- [`src/loggers/wandb_handler.py`](/E:/projects/deeplab/src/loggers/wandb_handler.py)
- [`src/loggers/__init__.py`](/E:/projects/deeplab/src/loggers/__init__.py)

现在 `BaseModel` 不再直接判断 MLflow/WandB 类型，而是通过 `LoggerFactory` 获取 handler。

这让后续新增 logger backend 时，不需要再去改模型层逻辑。


## 6. Hydra 配置结构

Hydra 顶层配置在 [`conf/config.yaml`](/E:/projects/deeplab/conf/config.yaml)。

配置分组如下：

- [`conf/mode`](/E:/projects/deeplab/conf/mode)
- [`conf/model`](/E:/projects/deeplab/conf/model)
- [`conf/datamodule`](/E:/projects/deeplab/conf/datamodule)
- [`conf/logger`](/E:/projects/deeplab/conf/logger)
- [`conf/callbacks`](/E:/projects/deeplab/conf/callbacks)
- [`conf/trainer`](/E:/projects/deeplab/conf/trainer)
- [`conf/paths`](/E:/projects/deeplab/conf/paths)
- [`conf/hydra`](/E:/projects/deeplab/conf/hydra)
- [`conf/hpo`](/E:/projects/deeplab/conf/hpo)

关键默认值：

- `experiment_name: ${project_name}`
- `run_name: ${mode.name}`

这样可以避免 Hydra 输出目录出现 `null/null` 这种无意义路径。


## 7. Shell 编排层

脚本目录在 [`scripts`](/E:/projects/deeplab/scripts)。

当前保留的编排脚本：

- [`run_flat_cv.sh`](/E:/projects/deeplab/scripts/run_flat_cv.sh)
- [`run_nested_cv.sh`](/E:/projects/deeplab/scripts/run_nested_cv.sh)
- [`run_cv_refit.sh`](/E:/projects/deeplab/scripts/run_cv_refit.sh)
- [`hpo.sh`](/E:/projects/deeplab/scripts/hpo.sh)
- [`aggregate_json_metrics.py`](/E:/projects/deeplab/scripts/aggregate_json_metrics.py)

这里的原则是：

- Python runtime 只负责最小执行单元
- shell 脚本负责高阶实验流程编排
- 聚合逻辑基于 `run_summary.json`，不再猜测内部目录规则


## 8. 当前项目结构

```text
deeplab/
├── main.py
├── CODEX.md
├── FRAMEWORK_ARCHITECTURE.md
├── FRAMEWORK_ARCHITECTURE_ZH.md
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
│   ├── core/
│   ├── datamodules/
│   ├── loggers/
│   ├── models/
│   └── utils/
├── scripts/
└── tests/
```


## 9. 各目录职责

### `conf/`

Hydra 配置分组目录，决定哪些对象和参数可以通过配置切换。

### `src/config/`

运行时关键配置的 schema 校验层。

### `src/core/`

执行语义层，包含 bootstrap、mode dispatch、runtime、artifact contract。

### `src/datamodules/`

数据加载和 split 消费层。

### `src/models/`

模型任务逻辑和模型模板。

### `src/loggers/`

日志后端适配层。

### `src/utils/`

构建、配置持久化、hyperparameter logging、代码快照等通用工具。

### `scripts/`

复杂实验编排层。

### `tests/`

最小回归测试，包括 split 与 runtime contract 的测试骨架。


## 10. 当前架构的优点

- 顶层执行边界清晰，只保留 `train/cv`
- 运行期 contract 明确，脚本可以稳定消费 summary
- split policy 和 datamodule 解耦
- logger backend 差异从模型层抽离
- Hydra 配置结构比较规整
- 高阶实验流程通过脚本编排，方向正确


## 11. 当前仍保留的限制

- schema 目前还是“关键字段校验”，不是完整 structured config
- dummy datamodule 仍然只是模板
- 脚本编排仍是 shell 级别，还没有更高阶 workflow engine
- artifact contract 目前以 `run_summary.json` 为主，尚未形成更丰富的 artifact index


## 12. 结论

当前 training framework 已经从“能跑的原型”进入“边界明确的最小训练框架”阶段。

它的核心特征现在是：

- Hydra 组合配置驱动
- Python 仅保留 `train/cv`
- 单次运行有显式 `RunContext`
- 运行结果有显式 `run_summary.json`
- split policy 独立
- logger backend 独立
- 复杂流程通过 shell 编排

这是一套可以继续扩展的基础架构，而且扩展方向已经比较清晰。
