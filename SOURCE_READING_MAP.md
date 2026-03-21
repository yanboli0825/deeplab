# Source Reading Map

这份文档是 [`CALL_FLOW.md`](/E:/projects/deeplab/CALL_FLOW.md) 的补充版。

如果说 `CALL_FLOW.md` 回答的是“代码怎么跑起来”，那这份文档回答的是：

- 每个文件在整体架构里的位置是什么
- 阅读这个文件时先看什么，后看什么
- 哪些函数和值最值得盯住
- 读完这个文件后，你应该建立什么认知


## 1. 推荐阅读策略

不要一上来全仓库跳着看。建议按下面三轮阅读：

### 第一轮：只看主链

目标：搞清楚一次 `train` 是怎么跑起来的。

顺序：

1. [`main.py`](/E:/projects/deeplab/main.py)
2. [`src/config/schema.py`](/E:/projects/deeplab/src/config/schema.py)
3. [`src/core/bootstrap.py`](/E:/projects/deeplab/src/core/bootstrap.py)
4. [`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)
5. [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)
6. [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)

### 第二轮：看对象怎么工作

目标：搞清楚 model 和 datamodule 是怎么被 runtime 调起来的。

顺序：

1. [`src/datamodules/base.py`](/E:/projects/deeplab/src/datamodules/base.py)
2. [`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)
3. [`src/datamodules/datasets/tabular_dataset.py`](/E:/projects/deeplab/src/datamodules/datasets/tabular_dataset.py)
4. [`src/models/base_model.py`](/E:/projects/deeplab/src/models/base_model.py)
5. [`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)
6. [`src/models/backbones/tabular_mlp.py`](/E:/projects/deeplab/src/models/backbones/tabular_mlp.py)
7. [`src/models/heads/classification_head.py`](/E:/projects/deeplab/src/models/heads/classification_head.py)

### 第三轮：看高阶编排

目标：搞清楚 `cv` 和 workflow 为什么没有破坏最小执行单元边界。

顺序：

1. [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)
2. [`src/core/contracts.py`](/E:/projects/deeplab/src/core/contracts.py)
3. [`src/workflows/common.py`](/E:/projects/deeplab/src/workflows/common.py)
4. [`src/workflows/flat_cv.py`](/E:/projects/deeplab/src/workflows/flat_cv.py)
5. [`src/workflows/nested_cv.py`](/E:/projects/deeplab/src/workflows/nested_cv.py)
6. [`src/workflows/hpo_refit.py`](/E:/projects/deeplab/src/workflows/hpo_refit.py)


## 2. 入口层怎么读

### [`main.py`](/E:/projects/deeplab/main.py)

这个文件的定位：

- 框架唯一的 Hydra 入口
- 只做配置校验、bootstrap、mode 分发
- 不做训练细节

建议阅读顺序：

1. 看 `main(cfg)` 的前 3 行
2. 看 `cfg.runtime` 是怎么被注入的
3. 看 mode dispatch

阅读时重点盯住：

- `cfg = validate_app_config(cfg)`
- `bootstrap = bootstrap_app(cfg)`
- `cfg.runtime = {...}`
- `if cfg.mode.name == "train"`
- `if cfg.mode.name == "cv"`

读完后你应该得到的结论：

- `main.py` 只是“入口控制器”
- 真正执行训练的代码不在这里


### [`src/config/schema.py`](/E:/projects/deeplab/src/config/schema.py)

这个文件的定位：

- 框架级 config contract
- 把 Hydra 的灵活配置收紧成 runtime 能稳定消费的结构

建议阅读顺序：

1. 先看 dataclass 定义，不要先看函数
2. 看 `PluginConfig / LoggerCollectionConfig / CallbackCollectionConfig`
3. 看 `_normalize_*`
4. 最后看 `validate_app_config()`

阅读时重点盯住：

- `AppConfig`
- `PluginConfig`
- `_normalize_object_config()`
- `_normalize_logger_config()`
- `_normalize_callback_config()`
- `_validate_mode()`
- `validate_app_config()`

读完后你应该得到的结论：

- runtime 并不是直接吃原始 Hydra config
- 它吃的是“框架归一化后的 config”


### [`src/core/bootstrap.py`](/E:/projects/deeplab/src/core/bootstrap.py)

这个文件的定位：

- 一次 Hydra 调用只执行一次的初始化层

建议阅读顺序：

1. 看 `BootstrapArtifacts`
2. 看 `bootstrap_app()`

阅读时重点盯住：

- `output_dir`
- `resolved_config_path`
- `artifact_index_path`

读完后你应该得到的结论：

- bootstrap 只准备环境和初始产物
- bootstrap 不执行训练循环


## 3. runtime 主链怎么读

### [`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)

这个文件的定位：

- 最薄的一层 mode 封装

怎么读：

1. 先看 `run_experiment(cfg)`
2. 再看返回逻辑是 `val_score` 还是 `test_score`

重点问题：

- 为什么这里不实例化 trainer
- 为什么这里只返回一个 float

读完后应有的理解：

- `train` mode 只是 runtime 单元的轻包装


### [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)

这是全仓库最值得反复读的文件。

这个文件的定位：

- 单次最小训练执行单元
- runtime 的核心

建议阅读顺序：

1. 看 `RunResult`
2. 看 `_build_run_context()`
3. 看 `_with_runtime_context()`
4. 看 `run_experiment()`
5. 最后回头看 `_build_artifact_index()`

不要上来就从头逐行看。先抓骨架。

阅读时最关键的变量：

- `context`
- `local_cfg`
- `provider`
- `model`
- `datamodule`
- `logger`
- `callbacks`
- `trainer`
- `summary`
- `artifact_index`

阅读时最关键的问题：

1. run 的名字和输出目录在哪定下来的
2. fold 信息在哪注入的
3. datamodule 为什么能拿到 runtime artifact 设置
4. trainer 是在哪一行真正启动的
5. summary 和 artifact index 是在哪一行落盘的

读完后你应该得到的结论：

- `run_experiment()` 就是整个框架的“最小训练原子”
- 以后任何复杂流程，本质上都是在复用它


### [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)

这个文件的定位：

- Hydra config 到 Python 对象的装配层

建议阅读顺序：

1. 看 `_materialize_object_config()`
2. 看 `build_model()`
3. 看 `build_datamodule()`
4. 看 `build_split_provider()`
5. 看 `build_logger() / build_callbacks() / build_trainer()`

阅读时重点盯住：

- `_target_`
- `init_args`
- `split_provider`
- `items`

读完后你应该得到的结论：

- runtime 不直接知道对象实现细节
- runtime 只负责“规范化 config -> instantiate”


## 4. 数据层怎么读

### [`src/datamodules/base.py`](/E:/projects/deeplab/src/datamodules/base.py)

这个文件的定位：

- datamodule 共同契约层

建议阅读顺序：

1. 看构造函数签名
2. 看 `resolve_split()`
3. 看 `_load_split_manifest()`
4. 看 `emit_data_artifacts()`
5. 看 `dataset_summary()` 和 `data_artifacts()`

阅读时重点盯住：

- `split_indices`
- `split_provider`
- `self._active_split`
- `runtime = self.hparams.data_cfg.get("runtime", {})`

读完后你应该得到的结论：

- datamodule 的角色是“消费 split”
- datamodule 不是 split policy owner


### [`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)

这个文件的定位：

- 真实项目接入模板
- manifest 驱动 datamodule 的参考实现

建议阅读顺序：

1. 看 `prepare_data()`
2. 看 `setup()`
3. 看 `_default_split()`
4. 看 `dataset_summary()`
5. 看 `train_dataloader() / val_dataloader() / test_dataloader()`

阅读时重点盯住：

- `self.rows`
- `feature_columns`
- `label_column`
- `split = self.resolve_split()`

读完后你应该得到的结论：

- 真正的数据接入逻辑应该集中在 datamodule 和 dataset
- 训练语义不应该渗入数据层


### [`src/datamodules/datasets/tabular_dataset.py`](/E:/projects/deeplab/src/datamodules/datasets/tabular_dataset.py)

这个文件的定位：

- 最底层单样本读取逻辑

怎么读：

1. 看 `__init__`
2. 看 `__getitem__`

阅读重点：

- 一条 manifest row 怎么变成 tensor

读完后你应该得到的结论：

- dataset 只解决样本读取，不解决 split，不解决 batch，不解决训练


### [`src/datamodules/split.py`](/E:/projects/deeplab/src/datamodules/split.py)

这个文件的定位：

- split policy 工具层
- 与 Lightning 无关

建议阅读顺序：

1. 看 `SplitIndices`
2. 看 `SplitProvider` 及其几个子类
3. 看 `make_holdout_split()`
4. 看 `make_kfold_split()`
5. 看 group-aware 版本

阅读时重点盯住：

- `candidate_indices`
- `group_ids`
- `fold`
- `n_splits`

读完后你应该得到的结论：

- split 逻辑已经从 datamodule 和 workflow 中抽出来了
- 这使它可以被独立测试和复用


## 5. 模型层怎么读

### [`src/models/base_model.py`](/E:/projects/deeplab/src/models/base_model.py)

这个文件的定位：

- 任务模型的共享 Lightning scaffold

建议阅读顺序：

1. 看 `__init__`
2. 看 `_setup_metrics()`
3. 看 `training_step()`
4. 看 `validation_step()` 和 `on_validation_epoch_end()`
5. 看 `test_step()` 和 `on_test_epoch_end()`
6. 看 `_log_confusion_matrix()`
7. 看 `configure_optimizers()`

阅读时重点盯住：

- `self.monitor`
- `self.best_metric`
- `self.train_metrics / self.val_metrics / self.test_metrics`
- `self.val_cm / self.test_cm`

阅读时要思考的问题：

- 为什么 metric 和 confusion matrix 放在 base class
- 为什么 logger backend 没有写 `if mlflow / if wandb`
- 为什么 optimizer/scheduler 直接从 `self.hparams` instantiate

读完后你应该得到的结论：

- `BaseModel` 负责通用训练语义
- task model 只需要把网络前向补上


### [`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)

这个文件的定位：

- 任务模型模板
- backbone/head/task 拆分的示例

建议阅读顺序：

1. 看 `__init__`
2. 看 `forward()`

阅读重点：

- `self.backbone`
- `self.head`
- `model_cfg["input_dim"]`
- `model_cfg["num_classes"]`

读完后你应该得到的结论：

- 任务模型应该负责组合网络部件
- 不应该把 dataloader、split、artifact 细节写进来


### [`src/models/backbones/tabular_mlp.py`](/E:/projects/deeplab/src/models/backbones/tabular_mlp.py)

这个文件的定位：

- 纯特征提取器

怎么读：

1. 看构造时怎么把 `hidden_dims` 拼成层
2. 看 `self.output_dim`
3. 看 `forward()`

读完后你应该得到的结论：

- backbone 应该尽量纯，不携带训练流程知识


### [`src/models/heads/classification_head.py`](/E:/projects/deeplab/src/models/heads/classification_head.py)

这个文件的定位：

- 最简单的任务头

怎么读：

1. 看 `__init__`
2. 看 `forward()`

读完后你应该得到的结论：

- head 的职责就是把 feature 映射成任务输出


## 6. contract 层怎么读

### [`src/core/contracts.py`](/E:/projects/deeplab/src/core/contracts.py)

这个文件的定位：

- runtime 和 workflow 的产物契约层

建议阅读顺序：

1. `RunContext`
2. `RunSummary`
3. `ArtifactIndex`
4. `CvSummary`
5. `WorkflowSummary`

阅读时重点盯住：

- 哪些字段是“单次 run”
- 哪些字段是“聚合层”
- 哪些字段是给脚本/workflow 稳定消费的

读完后你应该得到的结论：

- 当前框架的输出不是隐式目录结构
- 而是显式 contract


## 7. 聚合层怎么读

### [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)

这个文件的定位：

- runtime 内建聚合层

怎么读：

1. 看 fold 循环
2. 看 `scores`、`test_scores`
3. 看 `CvSummary`
4. 看聚合 `ArtifactIndex`

阅读重点：

- `run_experiment(cfg, fold=fold)`
- `fold_summary_paths`
- `fold_artifact_paths`

读完后你应该得到的结论：

- `cv` 本质上只是“多次 run_experiment + 聚合”


### [`src/workflows/common.py`](/E:/projects/deeplab/src/workflows/common.py)

这个文件的定位：

- workflow 层的公共基础设施

建议阅读顺序：

1. `run_main()`
2. `latest_json()`
3. `latest_path()`
4. `write_workflow_outputs()`

阅读重点：

- workflow 为什么通过 subprocess 调 `main.py`
- workflow 是怎么消费 child summary 的

读完后你应该得到的结论：

- workflow 没有侵入 runtime
- workflow 是 runtime 之上的编排层


### [`src/workflows/flat_cv.py`](/E:/projects/deeplab/src/workflows/flat_cv.py)

这个文件的定位：

- 最容易读懂的 workflow 示例

怎么读：

1. 看 argparse
2. 看 fold 循环里生成的 override
3. 看如何读取 `run_summary.json`
4. 看如何写 `workflow_summary.json`

读完后你应该得到的结论：

- flat CV 不是新 mode
- 它只是一个重复调用 `train` 的 workflow


### [`src/workflows/nested_cv.py`](/E:/projects/deeplab/src/workflows/nested_cv.py)

这个文件的定位：

- 最完整的 workflow 控制流示例

建议阅读顺序：

1. `_parse_candidate()`
2. outer loop
3. candidate loop
4. inner split loop
5. refit run
6. workflow summary 写出

阅读重点：

- `best_score`
- `best_override`
- `split_file`

读完后你应该得到的结论：

- nested CV 的复杂性属于 workflow，不属于 runtime


### [`src/workflows/hpo_refit.py`](/E:/projects/deeplab/src/workflows/hpo_refit.py)

这个文件的定位：

- “workflow 调 cv，再调 train”的嵌套示例

怎么读：

1. 看 candidate loop 里为什么跑的是 `mode=cv`
2. 看怎么从 `workflow_summary.json` 取分数
3. 看 best candidate 怎么再进入 `mode=train`

读完后你应该得到的结论：

- workflow 可以组合 runtime mode
- 但不会改变 runtime 的边界


## 8. 配置文件要怎么对照着读

源码阅读时，建议同时开着这几个配置文件：

1. [`conf/config.yaml`](/E:/projects/deeplab/conf/config.yaml)
2. [`conf/model/cpath/tabular_classification.yaml`](/E:/projects/deeplab/conf/model/cpath/tabular_classification.yaml)
3. [`conf/datamodule/cpath/tabular_manifest.yaml`](/E:/projects/deeplab/conf/datamodule/cpath/tabular_manifest.yaml)
4. [`conf/trainer/default.yaml`](/E:/projects/deeplab/conf/trainer/default.yaml)
5. [`conf/callbacks/default.yaml`](/E:/projects/deeplab/conf/callbacks/default.yaml)
6. [`conf/logger/mlflow.yaml`](/E:/projects/deeplab/conf/logger/mlflow.yaml)

推荐对照方法：

- 看一个 Python 构造函数，就回头看对应 `_target_`
- 看一个对象属性，就回头看它从哪个 `init_args` 进入
- 看一个 runtime 行为，就回头看它有没有配置入口


## 9. 你读源码时最值得记下来的核心对象

建议你一边读，一边把下面这些对象和职责记下来：

- `cfg`: Hydra 配置对象
- `local_cfg`: 注入 runtime 字段后的本地 config
- `RunContext`: 单次 run 的上下文
- `RunSummary`: 单次 run 的摘要
- `ArtifactIndex`: 单次 run 的 artifact 索引
- `CvSummary`: `cv` 聚合摘要
- `WorkflowSummary`: workflow 聚合摘要
- `BaseDataModule`: datamodule 契约
- `BaseModel`: 模型训练语义契约


## 10. 一个最实用的阅读办法

如果你只想最快读懂，不要平均用力。建议你这样做：

1. 先把 [`main.py`](/E:/projects/deeplab/main.py) 和 [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py) 读通。
2. 再把 [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py) 看明白，搞清楚 config 是怎么变成对象的。
3. 然后挑一个 datamodule 和一个 model 读通。
4. 最后再看 `cv` 和 workflow。

这样读的收益最大，因为你先掌握的是“主干”，不是“枝叶”。


## 11. 配合文档的使用方式

建议你同时开三份文档：

- 调用链：[`CALL_FLOW.md`](/E:/projects/deeplab/CALL_FLOW.md)
- 源码地图：[`SOURCE_READING_MAP.md`](/E:/projects/deeplab/SOURCE_READING_MAP.md)
- 扩展规则：[`DEVELOPMENT_GUIDE.md`](/E:/projects/deeplab/DEVELOPMENT_GUIDE.md)

三者分别回答的问题是：

- `CALL_FLOW.md`：代码怎么跑
- `SOURCE_READING_MAP.md`：文件怎么读
- `DEVELOPMENT_GUIDE.md`：以后怎么扩展

