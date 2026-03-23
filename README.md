# DeepLab Training Framework

这是一个基于 Hydra + PyTorch Lightning 的训练框架。它的核心目标不是把所有训练流程都做成 Python 入口，而是把训练系统收敛成两个最小执行单元：

- `train`: 一次具体训练运行
- `cv`: 多次 `train` 的 fold 循环

更复杂的实验流程，例如 `flat-cv`、`nested-cv`、`hpo + refit`，不再继续扩充 runtime mode，而是放在 `src/workflows/` 和 `scripts/` 中组合执行。这样做的目的，是让训练框架的边界稳定、配置清晰、扩展成本可控。

## 1. 这套框架的设计原则

### 1.1 完全配置驱动

框架通过 Hydra 组合配置来构建对象和控制运行参数。框架自有的配置接口统一为：

- `model._target_ + model.init_args`
- `datamodule._target_ + datamodule.init_args`
- `trainer._target_ + trainer.init_args`
- `logger.items`
- `callbacks.items`

这意味着：

- 你新增模型时，主要工作是写 Python 类和 Hydra 配置
- 你切换 datamodule、logger、trainer、optimizer 时，主要通过 override 完成
- 主入口 `main.py` 不需要不断加 if/else 分支来适配新任务

### 1.2 只保留两个一等运行模式

`main.py` 只认识 `train` 和 `cv`。这不是功能缺失，而是架构选择。

- `train` 负责执行一次完整训练单元
- `cv` 负责重复调用 `train` 风格的执行单元
- 更高阶流程由 workflow 层组合，而不是再添加 `nested_cv`、`flatten_cv` 之类的一级 mode

这样做的收益是：

- runtime 行为更容易理解
- artifact contract 更稳定
- 配置和代码边界更清晰

### 1.3 split policy 和 datamodule 解耦

datamodule 负责“读取数据并构造 dataloader”，不负责决定“如何做 holdout / kfold / group kfold”。

split policy 由顶层 `split` 配置和 split provider 控制，runner 在运行时把 split 结果注入 datamodule。这样 datamodule 可以复用到：

- 单次 train
- CV
- workflow 组合实验

而不需要把 fold 逻辑写死在数据类内部。

### 1.4 artifact contract 是一等公民

每个具体 run 都会产出一组稳定的基础 artifacts：

- `config.yaml`
- `run_summary.json`
- `artifacts.json`
- `checkpoints/`
- datamodule 写出的数据相关摘要

聚合层也会写稳定的 workflow artifacts，供 shell 脚本和上层工作流消费。

## 2. 架构总览

可以先把这套框架理解成四层：

### 2.1 入口层

- [`main.py`](/E:/projects/deeplab/main.py)

职责：

- 读取 Hydra 配置
- 调用 `validate_app_config()` 标准化框架配置面
- 调用 bootstrap 准备输出目录和运行时路径
- 根据 `cfg.mode.name` 分发到 `train` 或 `cv`

这里有意保持很薄，不承载复杂业务逻辑。

### 2.2 runtime 层

- [`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)
- [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)
- [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)

职责：

- 组装一次具体 run 的上下文
- 注入 runtime 字段
- 构建 model / datamodule / logger / callbacks / trainer
- 调用 Lightning 的 `fit` / `test`
- 写 `run_summary.json` 和 `artifacts.json`

其中 [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py) 是最关键的单次执行主链。

### 2.3 实例化与配置适配层

- [`src/config/schema.py`](/E:/projects/deeplab/src/config/schema.py)
- [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)

职责：

- 把 Hydra 原始配置规范化为框架统一 contract
- 把 `_target_ + init_args` 形式转成 Hydra instantiate 可消费的对象配置
- 构建 split provider、model、datamodule、logger、callbacks、trainer

这层的核心价值是：框架自己的 contract 和第三方库对象构造解耦。

### 2.4 扩展层

- [`src/datamodules/`](/E:/projects/deeplab/src/datamodules)
- [`src/models/`](/E:/projects/deeplab/src/models)
- [`src/workflows/`](/E:/projects/deeplab/src/workflows)

职责：

- datamodule：接数据
- model：定义任务
- workflow：组合复杂实验

这三层是你后续接自己项目时最常修改的地方。

## 3. 项目结构怎么读

新用户第一次读这个仓库，建议重点看下面这些目录，而不是一开始就通读所有文件。

```text
conf/                Hydra 配置入口和配置组
src/config/          框架配置 schema 与标准化逻辑
src/core/            train/cv/run_experiment 主链
src/utils/           Hydra instantiate 适配与通用工具
src/datamodules/     数据接入层、split、dataset
src/models/          任务模型、backbone、head
src/workflows/       更高阶实验编排
scripts/             workflow 的薄启动脚本
data/manifests/      示例 manifest
tests/               当前最小测试骨架
```

如果你只想先抓住主干，建议按这个顺序读：

1. [`main.py`](/E:/projects/deeplab/main.py)
2. [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)
3. [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)
4. [`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)
5. [`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)
6. [`conf/experiment/tabular_baseline.yaml`](/E:/projects/deeplab/conf/experiment/tabular_baseline.yaml)

## 4. 这套框架推荐你怎么组织自己的数据和模型

这是最重要的部分。你接自己的项目时，不要直接改 `runner.py` 或 `main.py`，而是按下面的边界扩展。

### 4.1 数据侧推荐组织方式

推荐采用 manifest-driven data access，而不是让 datamodule 直接从目录层级猜数据结构。

建议目录：

```text
data/
`-- manifests/
    `-- your_dataset.csv

src/datamodules/
|-- datasets/
|   `-- your_dataset.py
|-- transforms/
|   `-- your_transforms.py
`-- your_datamodule.py
```

推荐职责分工：

- `datasets/`：定义单样本读取逻辑
- `your_datamodule.py`：构建 train/val/test dataloader
- `split.py`：继续作为 split policy 层使用，不要把 fold 逻辑写死进 datamodule

manifest 至少建议包含：

- `sample_id`
- `label`
- `group`，如果你有 group-aware split 需求
- `path`，如果样本来自文件
- 任务特定 feature 列

参考实现：

- [`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)
- [`src/datamodules/datasets/tabular_dataset.py`](/E:/projects/deeplab/src/datamodules/datasets/tabular_dataset.py)
- [`data/manifests/example_tabular.csv`](/E:/projects/deeplab/data/manifests/example_tabular.csv)

### 4.2 模型侧推荐组织方式

推荐把模型拆成 backbone、head、task model 三层。

建议目录：

```text
src/models/
|-- backbones/
|   `-- your_backbone.py
|-- heads/
|   `-- your_head.py
`-- tasks/
    `-- your_task_model.py
```

推荐职责分工：

- `backbones/`：特征提取
- `heads/`：任务输出头
- `tasks/`：LightningModule 级别的训练任务

这样做的好处是：

- backbone 可复用
- head 可替换
- task model 只负责任务语义，而不是把所有东西揉成一个大类

参考实现：

- [`src/models/backbones/tabular_mlp.py`](/E:/projects/deeplab/src/models/backbones/tabular_mlp.py)
- [`src/models/heads/classification_head.py`](/E:/projects/deeplab/src/models/heads/classification_head.py)
- [`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)

### 4.3 配置侧推荐组织方式

你真正接项目时，核心不是改入口文件，而是补这些配置：

```text
conf/
|-- model/
|   `-- your_model.yaml
|-- datamodule/
|   `-- your_datamodule.yaml
`-- experiment/
    `-- your_experiment.yaml
```

推荐流程：

- `conf/model/...`：定义 `_target_` 和 `init_args`
- `conf/datamodule/...`：定义数据文件、列名、batch size、num_workers 等
- `conf/experiment/...`：把 model/datamodule/logger/trainer 组合成一次完整实验

参考配置：

- [`conf/model/cpath/tabular_classification.yaml`](/E:/projects/deeplab/conf/model/cpath/tabular_classification.yaml)
- [`conf/datamodule/cpath/tabular_manifest.yaml`](/E:/projects/deeplab/conf/datamodule/cpath/tabular_manifest.yaml)
- [`conf/experiment/tabular_baseline.yaml`](/E:/projects/deeplab/conf/experiment/tabular_baseline.yaml)

## 5. 如果我要在自己的数据和模型上跑训练，应该怎么做

下面是一条最短可落地路径。

### 第一步：准备 manifest

先把你的原始数据整理成一张 manifest 表，而不是一上来写复杂 datamodule。

对于表格任务，至少需要：

- 样本标识列
- 标签列
- 特征列

对于图像/病理/文本等任务，也建议用 manifest 管理样本，而不是让 datamodule 直接扫目录。

### 第二步：写 dataset

在 [`src/datamodules/datasets/`](/E:/projects/deeplab/src/datamodules/datasets) 下新增你的 dataset，实现单样本读取逻辑。

关键原则：

- dataset 负责一个 sample 怎么读
- 不要在 dataset 里处理 k-fold
- 不要在 dataset 里判断当前是不是 train/cv mode

### 第三步：写 datamodule

新增你的 datamodule，负责：

- 读取 manifest
- 根据注入的 split 结果切 train/val/test
- 返回 dataloader

如果你没有显式 split provider，也可以先像参考 datamodule 一样提供一个默认 split 逻辑，但推荐后续把 split 上收至顶层配置。

### 第四步：写 task model

在 [`src/models/tasks/`](/E:/projects/deeplab/src/models/tasks) 下新增你的任务模型。

最重要的边界是：

- model 负责 `forward`、loss、metric、optimizer/scheduler 配置
- model 不负责路径命名、logger 类型判断、cv 编排

### 第五步：写 Hydra 配置

至少补三份配置：

1. model 配置
2. datamodule 配置
3. experiment 配置

其中 experiment 配置负责把一组组件拼成一次能运行的实验。

### 第六步：先跑一次 `train`

建议第一步只验证最小执行单元：

```bash
python main.py mode=train experiment=tabular_baseline
```

如果你有自己的 experiment：

```bash
python main.py mode=train experiment=your_experiment
```

### 第七步：再跑 `cv`

单次 train 跑通后，再切到 CV：

```bash
python main.py mode=cv experiment=your_experiment mode.n_folds=5
```

不要反过来先跑复杂 workflow。先把最小执行单元跑通，后面的组合流程才可控。

## 6. 最小可运行示例

### 6.1 跑一次单次训练

```bash
python main.py mode=train
```

### 6.2 跑一次 tabular baseline

```bash
python main.py experiment=tabular_baseline
```

### 6.3 覆盖 trainer 和 optimizer 参数

```bash
python main.py experiment=tabular_baseline trainer.init_args.max_epochs=20 model.init_args.optimizer.lr=3e-4
```

### 6.4 跑一次 CV

```bash
python main.py mode=cv experiment=tabular_baseline mode.n_folds=5
```

## 7. 当前项目里真实可用的参考实现

如果你想快速照着改，建议直接参考下面这组文件：

- 入口：[`main.py`](/E:/projects/deeplab/main.py)
- 单次运行主链：[`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)
- 参考 datamodule：[`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)
- 参考 dataset：[`src/datamodules/datasets/tabular_dataset.py`](/E:/projects/deeplab/src/datamodules/datasets/tabular_dataset.py)
- 参考 task model：[`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)
- 参考 experiment：[`conf/experiment/tabular_baseline.yaml`](/E:/projects/deeplab/conf/experiment/tabular_baseline.yaml)

如果你是第一次接这个项目，这组文件已经足够帮助你理解整个最短链路。

## 8. 运行后会生成什么

每个具体 run 都会在 Hydra 输出目录下生成一组产物。

常见基础产物包括：

- `config.yaml`：最终 resolved config
- `run_summary.json`：本次 run 的核心摘要
- `artifacts.json`：完整 artifact 索引
- `checkpoints/`：Lightning checkpoint
- datamodule 相关的数据摘要产物
- `confusion_matrices/`：如果模型导出了混淆矩阵

其中：

- `run_summary.json` 适合被 train/cv/workflow 直接消费
- `artifacts.json` 适合做更完整的结果发现与后处理

CV 和 workflow 聚合层还会额外写：

- `workflow_summary.json`
- `workflow_artifacts.json`

## 9. 更高阶流程怎么用

当前仓库已经把复杂实验流程收到了 workflow 层和脚本层。

可用脚本入口在 [`scripts/`](/E:/projects/deeplab/scripts)：

- `run_flat_cv.sh`
- `run_nested_cv.sh`
- `hpo.sh`

这些脚本本质上是 workflow 的薄启动器，不应该替代 `train` / `cv` 本身。

推荐理解方式是：

- `train` / `cv` 是基础执行单元
- workflow 是组合逻辑
- shell 只是调用 workflow 的外壳

## 10. 这套框架鼓励什么，不鼓励什么

### 鼓励

- 用 Hydra 配置来组合对象和运行参数
- 用 manifest 驱动数据读取
- 只把 `train` 和 `cv` 当成一级 runtime mode
- 用 workflow 组合复杂实验
- 保持 `_target_ + init_args` 的插件式对象定义
- 依赖 `run_summary.json` 和 `artifacts.json` 作为稳定 contract

### 不鼓励

- 为每种实验都再加一个新的 runtime mode
- 在 datamodule 里写死 fold / nested-cv 逻辑
- 在 model 里混入 logger 分支、输出目录规则、workflow 控制流
- 把复杂业务逻辑继续塞进 `main.py`
- 绕开 config group，直接在代码里硬编码对象切换逻辑

## 11. 相关文档

如果你在读完 README 后还想继续深入：

- 架构说明：[`docs/FRAMEWORK_ARCHITECTURE.md`](/E:/projects/deeplab/docs/FRAMEWORK_ARCHITECTURE.md)
- 中文架构说明：[`docs/FRAMEWORK_ARCHITECTURE_ZH.md`](/E:/projects/deeplab/docs/FRAMEWORK_ARCHITECTURE_ZH.md)
- 开发扩展指南：[`docs/DEVELOPMENT_GUIDE.md`](/E:/projects/deeplab/docs/DEVELOPMENT_GUIDE.md)
- 调用流程：[`docs/CALL_FLOW.md`](/E:/projects/deeplab/docs/CALL_FLOW.md)
- 源码阅读地图：[`docs/SOURCE_READING_MAP.md`](/E:/projects/deeplab/docs/SOURCE_READING_MAP.md)

如果你的目标是“把我自己的数据和模型接进来并跑通一次训练”，建议先读这份 README，再去看参考 datamodule、参考 task model 和 experiment 配置。
