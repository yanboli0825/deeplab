# Call Flow Guide

这份文档面向“对照源码阅读”的场景，目标不是介绍概念，而是回答三个问题：

1. `python main.py ...` 之后，代码按什么顺序执行
2. `train`、`cv`、workflow 三条路径分别走到哪里
3. 训练结果是在哪些位置被写成 `run_summary.json`、`artifacts.json` 和 workflow summary


## 1. 建议阅读顺序

如果你想最快建立整体认知，建议按下面顺序阅读源码：

1. [`main.py`](/E:/projects/deeplab/main.py)
2. [`src/config/schema.py`](/E:/projects/deeplab/src/config/schema.py)
3. [`src/core/bootstrap.py`](/E:/projects/deeplab/src/core/bootstrap.py)
4. [`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)
5. [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)
6. [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)
7. [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)
8. [`src/datamodules/base.py`](/E:/projects/deeplab/src/datamodules/base.py)
9. [`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)
10. [`src/models/base_model.py`](/E:/projects/deeplab/src/models/base_model.py)
11. [`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)
12. [`src/workflows/common.py`](/E:/projects/deeplab/src/workflows/common.py)
13. [`src/workflows/flat_cv.py`](/E:/projects/deeplab/src/workflows/flat_cv.py)
14. [`src/workflows/nested_cv.py`](/E:/projects/deeplab/src/workflows/nested_cv.py)
15. [`src/workflows/hpo_refit.py`](/E:/projects/deeplab/src/workflows/hpo_refit.py)


## 2. 总体调用链

无论你最终跑的是 `train`、`cv`，还是更高阶 workflow，最底层都会回到同一个最小执行单元：

```text
Hydra CLI
  -> main.py
  -> validate_app_config()
  -> bootstrap_app()
  -> mode dispatch
     -> train_loop() or cv_loop()
        -> run_experiment()
           -> build_model()
           -> build_datamodule()
           -> build_logger()
           -> build_callbacks()
           -> build_trainer()
           -> trainer.fit()
           -> optional trainer.test()
           -> write run_summary.json
           -> write artifacts.json
```

可以把它理解成三层：

- 入口层：决定“这次运行是什么模式”
- runtime 层：执行“一次最小训练单元”
- workflow 层：组合多次最小训练单元


## 3. `train` 路径调用流程

### 3.1 从命令行进入

你执行：

```bash
python main.py mode=train
```

调用链是：

```text
main.py:main
  -> validate_app_config(cfg)
  -> bootstrap_app(cfg)
  -> train_loop(cfg)
  -> run_experiment(cfg)
```

对应源码位置：

- 入口：[`main.py`](/E:/projects/deeplab/main.py)
- 模式分发：[`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)
- 单次运行：[`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)


### 3.2 `main.py` 做了什么

`main()` 只做四件事：

1. 让 Hydra 先把配置组合成 `cfg`
2. 调 `validate_app_config(cfg)` 规范化框架关心的配置结构
3. 调 `bootstrap_app(cfg)` 做一次性的环境准备
4. 根据 `cfg.mode.name` 分发到 `train_loop()` 或 `cv_loop()`

这里要特别注意：

- `main.py` 不实例化 model/datamodule/trainer
- `main.py` 不做 fold 循环
- `main.py` 不写训练 summary

所以读源码时，不要在 `main.py` 里找训练细节，它只是入口和分发层。


### 3.3 `validate_app_config()` 做了什么

位置：[`src/config/schema.py`](/E:/projects/deeplab/src/config/schema.py)

这一步负责把 Hydra 组合出来的原始 config 变成框架能稳定消费的形态。重点动作：

- 校验 `mode` 是否是 `train` 或 `cv`
- 把 `model` 规范成 `_target_ + init_args`
- 把 `datamodule` 规范成 `_target_ + init_args`
- 把 `trainer` 规范成 `_target_ + init_args`
- 把 `logger` 规范成 `items`
- 把 `callbacks` 规范成 `items`

读到这里时，你要建立一个关键认知：

> 后面 runtime 代码默认接收的是“规范化之后的 config”，不是原始随意形态的 config。


### 3.4 `bootstrap_app()` 做了什么

位置：[`src/core/bootstrap.py`](/E:/projects/deeplab/src/core/bootstrap.py)

这一步是一次性准备动作：

- 加载 `.env`
- 设随机种子
- 创建输出目录
- 保存解析后的 `config.yaml`
- 复制源码快照到 `code/`

注意边界：

- bootstrap 只执行一次
- bootstrap 不启动训练
- bootstrap 不负责 fold-specific 输出目录


### 3.5 `train_loop()` 做了什么

位置：[`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)

`train_loop()` 很薄，只做：

```text
train_loop(cfg)
  -> result = run_experiment(cfg)
  -> 如果 test_after_train=true，返回 test_score
  -> 否则返回 val_score
```

所以真正的训练逻辑都在 `run_experiment()`


### 3.6 `run_experiment()` 是最重要的函数

位置：[`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)

你可以把它理解成：

> “一次最小训练执行单元”

调用步骤如下：

```text
run_experiment()
  -> _build_run_context()
  -> _with_runtime_context()
  -> write_resolved_config()
  -> build_split_provider()
  -> build_model()
  -> build_datamodule()
  -> build_logger()
  -> build_callbacks()
  -> build_trainer()
  -> log_hyperparameters()
  -> trainer.fit()
  -> optional trainer.test()
  -> build RunSummary
  -> save_json(run_summary.json)
  -> _build_artifact_index()
  -> save_json(artifacts.json)
  -> return RunResult
```


### 3.7 `RunContext` 是怎么生成的

先看 `_build_run_context()`。

它负责决定这次 run 的几个关键属性：

- `experiment_name`
- `run_name`
- `output_dir`
- `resolved_config_path`
- `summary_path`
- `artifact_index_path`
- `fold`

这意味着：

- run 名字不是 logger 自己决定的
- summary 路径不是 shell 猜出来的
- runtime 层显式定义了一次 run 的输出边界


### 3.8 runtime config 是怎么注入的

再看 `_with_runtime_context()`。

它做了两类注入：

1. 顶层 runtime 字段
2. datamodule 运行时 artifact 配置

注入后，`local_cfg.runtime` 里会出现：

- `mode`
- `fold`
- `output_dir`
- `run_name`
- `experiment_name`
- `resolved_config_path`
- `summary_path`
- `artifact_index_path`

同时还会把下面这段注入到 datamodule：

```text
local_cfg.datamodule.init_args.data_cfg.runtime
  -> output_dir
  -> artifacts
```

这就是为什么 datamodule 在 `setup()` 后可以自己写：

- `split_manifest.yaml`
- `dataset_summary.json`


### 3.9 对象是怎么被实例化的

位置：[`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)

`run_experiment()` 不直接 `new` 对象，而是走 builder：

- `build_model()`
- `build_datamodule()`
- `build_logger()`
- `build_callbacks()`
- `build_trainer()`

这里有两个关键点：

#### `build_model()`

内部先调 `_materialize_object_config()`，把：

```yaml
model:
  _target_: ...
  init_args:
    ...
```

变成 Hydra 可以直接 instantiate 的扁平结构。

#### `build_datamodule()`

会把 runtime 生成的：

- `split_indices`
- `split_provider`

注入 datamodule 构造函数。

所以 datamodule 的边界是：

> 它消费 split，不决定 split policy。


### 3.10 数据层调用是怎么走的

如果你读 tabular 示例，调用链是：

```text
run_experiment()
  -> build_datamodule()
     -> TabularClassificationDataModule(...)
  -> trainer.fit(...)
     -> Lightning 调用 datamodule.setup()
        -> resolve_split()
        -> _default_split() 或 split_file 或 split_provider
        -> 构造 train/val/test dataset
        -> emit_data_artifacts()
```

对应源码：

- datamodule 基类：[`src/datamodules/base.py`](/E:/projects/deeplab/src/datamodules/base.py)
- tabular datamodule：[`src/datamodules/tabular_dm.py`](/E:/projects/deeplab/src/datamodules/tabular_dm.py)
- dataset：[`src/datamodules/datasets/tabular_dataset.py`](/E:/projects/deeplab/src/datamodules/datasets/tabular_dataset.py)

你阅读时建议重点跟这几个函数：

- `resolve_split()`
- `_load_split_manifest()`
- `_default_split()`
- `setup()`
- `train_dataloader() / val_dataloader() / test_dataloader()`


### 3.11 模型层调用是怎么走的

如果你读 tabular 分类示例，调用链是：

```text
build_model()
  -> TabularClassificationModel(...)
     -> BaseModel.__init__()
     -> TabularMLPBackbone(...)
     -> ClassificationHead(...)

trainer.fit()
  -> training_step()
  -> on_train_epoch_end()
  -> validation_step()
  -> on_validation_epoch_end()

optional trainer.test()
  -> test_step()
  -> on_test_epoch_end()
```

对应源码：

- 任务模型基类：[`src/models/base_model.py`](/E:/projects/deeplab/src/models/base_model.py)
- 任务模型：[`src/models/tasks/tabular_classification.py`](/E:/projects/deeplab/src/models/tasks/tabular_classification.py)
- backbone：[`src/models/backbones/tabular_mlp.py`](/E:/projects/deeplab/src/models/backbones/tabular_mlp.py)
- head：[`src/models/heads/classification_head.py`](/E:/projects/deeplab/src/models/heads/classification_head.py)

要点是：

- `BaseModel` 管 step、metric、confusion matrix、optimizer/scheduler
- 具体 task model 只需要实现 `forward()`
- logger backend 细节不在 model 里分支


### 3.12 结果是怎么写出去的

训练结束后，`run_experiment()` 会做两次持久化：

#### 第一次：写 `run_summary.json`

来源对象：`RunSummary`

主要字段：

- `mode`
- `experiment_name`
- `run_name`
- `output_dir`
- `resolved_config_path`
- `summary_path`
- `artifact_index_path`
- `monitor`
- `val_score`
- `test_score`
- `best_ckpt_path`
- `fold`

#### 第二次：写 `artifacts.json`

来源对象：`ArtifactIndex`

主要内容：

- config 路径
- checkpoint 路径
- metrics
- data artifacts
- figures
- logger ids


## 4. `cv` 路径调用流程

你执行：

```bash
python main.py mode=cv mode.n_folds=5
```

调用链是：

```text
main.py:main
  -> validate_app_config()
  -> bootstrap_app()
  -> cv_loop(cfg)
     -> for fold in range(n_folds):
           run_experiment(cfg, fold=fold)
     -> 聚合每个 fold 的结果
     -> 写 workflow_summary.json
     -> 写 workflow_artifacts.json
```

对应源码：[`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)


### 4.1 和 `train` 最大的区别

`cv_loop()` 本身不训练，它只是：

1. 循环调用多次 `run_experiment()`
2. 收集每个 fold 的 `RunResult`
3. 聚合分数
4. 写 CV 级别的 summary

所以你可以把它理解成：

> `cv = 多次 train unit + 聚合`


### 4.2 fold-specific 输出目录怎么来的

关键在 `run_experiment(cfg, fold=fold)`。

当 `fold` 不为 `None` 时，`_build_run_context()` 会把输出目录变成：

```text
<parent_output_dir>/runs/<run_name>_fold{fold}
```

所以：

- fold run 各自有独立 summary
- fold run 各自有独立 artifacts
- `cv_loop()` 只负责把这些路径再聚合起来


### 4.3 CV 聚合产物是怎么写的

`cv_loop()` 会写两类聚合文件：

#### `workflow_summary.json`

来源对象：`CvSummary`

核心字段：

- `val_score`
- `test_score`
- `n_folds`
- `fold_summary_paths`
- `fold_artifact_paths`
- `fold_scores`
- `fold_test_scores`

#### `workflow_artifacts.json`

来源对象：`ArtifactIndex`

这里最关键的是：

- `children = fold_artifact_paths`

也就是：

> CV 聚合层不再猜目录，而是显式记录“我的子 run artifact 在哪”


## 5. workflow 路径调用流程

当前仓库里更高阶流程不通过新增 runtime mode 实现，而是走：

- [`src/workflows/flat_cv.py`](/E:/projects/deeplab/src/workflows/flat_cv.py)
- [`src/workflows/nested_cv.py`](/E:/projects/deeplab/src/workflows/nested_cv.py)
- [`src/workflows/hpo_refit.py`](/E:/projects/deeplab/src/workflows/hpo_refit.py)

这些 workflow 的共同套路都是：

```text
workflow main()
  -> 生成一组 Hydra overrides
  -> run_main(overrides)
     -> subprocess 调 main.py
        -> train 或 cv
  -> 读取子运行 summary/artifacts
  -> 聚合
  -> write_workflow_outputs()
```

公共辅助函数在：[`src/workflows/common.py`](/E:/projects/deeplab/src/workflows/common.py)


### 5.1 `run_main()` 的作用

`run_main()` 很关键。

它不是直接调 Python 函数，而是用 subprocess 再起一个：

```text
python main.py <overrides...>
```

这意味着 workflow 和 runtime 的关系是：

- workflow 是“上层调度者”
- runtime 是“被调度的最小执行单元”

这个边界很重要，因为它避免了 workflow 直接侵入 runtime 内部细节。


### 5.2 flat CV workflow

调用链：

```text
flat_cv.main()
  -> for fold in range(n_folds):
       run_main(["mode=train", ...])
       latest_json(run_summary.json)
       latest_path(artifacts.json)
  -> workflow_output_dir()
  -> write_workflow_outputs()
```

本质上是：

> workflow 级 flat CV = 多次独立 train + 聚合结果


### 5.3 nested CV workflow

调用链：

```text
nested_cv.main()
  -> 遍历 outer fold
     -> 遍历 candidate
        -> 遍历 inner split
           -> run_main(mode=train, split_file=inner_x.yaml, candidate overrides)
           -> 读取 inner run_summary.json
        -> 计算该 candidate 的 inner mean score
     -> 选择 best candidate
     -> run_main(mode=train, split_file=refit.yaml, best overrides)
     -> 记录 refit summary/artifacts
  -> write_workflow_outputs()
```

关键认知：

- inner/outer 并不是 runtime mode
- 它们只是 workflow 层的控制流
- runtime 仍然只认识 `train`


### 5.4 HPO + refit workflow

调用链：

```text
hpo_refit.main()
  -> 遍历 candidate
     -> run_main(mode=cv, candidate overrides)
     -> 读取 workflow_summary.json
  -> 选择 best candidate
  -> run_main(mode=train, best overrides)
  -> write_workflow_outputs()
```

这个流程很好地展示了“workflow 可以调 runtime 的 `cv`，而 `cv` 再调 runtime 的 `train unit`”这一层嵌套关系。


## 6. 配置是怎么流入代码的

阅读源码时，经常会搞不清“某个参数是在哪里进入对象的”，可以按下面这条线追：

```text
conf/*.yaml
  -> Hydra compose 成 cfg
  -> validate_app_config() 规范化
  -> _materialize_object_config()
  -> hydra.utils.instantiate()
  -> Python 对象构造函数 __init__()
```

比如：

```text
conf/model/cpath/tabular_classification.yaml
  -> cfg.model
  -> build_model(cfg.model)
  -> _materialize_object_config(cfg.model)
  -> instantiate(...)
  -> TabularClassificationModel.__init__(...)
```

数据也是一样：

```text
conf/datamodule/cpath/tabular_manifest.yaml
  -> cfg.datamodule
  -> build_datamodule(cfg.datamodule, split_indices, split_provider)
  -> instantiate(...)
  -> TabularClassificationDataModule.__init__(...)
```


## 7. 阅读源码时最容易混淆的几个点

### 7.1 `bootstrap` 和 `runner` 的区别

- `bootstrap_app()`：一次 Hydra 调用只执行一次
- `run_experiment()`：一次最小训练单元执行一次

如果是 `cv`，会出现：

- bootstrap 一次
- run_experiment 多次


### 7.2 `cv` 和 workflow 的区别

- `cv`：runtime 内建模式，知道 fold 循环
- workflow：更高阶的 Python 调度层，调用多个 `train/cv`

所以：

- `cv` 是 runtime 的一部分
- `flat_cv/nested_cv/hpo_refit` 是 runtime 外层的 orchestration


### 7.3 `run_summary.json` 和 `artifacts.json` 的区别

- `run_summary.json`：给人和脚本看的高频结果摘要
- `artifacts.json`：给下游系统和 workflow 用的完整 artifact 索引

聚合层也同理：

- `workflow_summary.json`
- `workflow_artifacts.json`


### 7.4 datamodule 为什么不该管理 fold

因为 fold 是 runtime/workflow 的执行语义，不是数据读取语义。

datamodule 只负责：

- 读取 manifest 或样本源
- 消费 split
- 构造 dataset / dataloader

所以读 datamodule 时，不要期待在里面看到：

- outer fold
- inner fold
- candidate selection

这些都不属于它的职责。


## 8. 一张总图

如果你想先抓住全局，可以先记住这张图：

```text
CLI / Script
  |
  v
Hydra compose config
  |
  v
main.py
  |
  +--> validate_app_config()
  |
  +--> bootstrap_app()
  |
  +--> train_loop()
  |      |
  |      +--> run_experiment()
  |             |
  |             +--> build_*()
  |             +--> trainer.fit()
  |             +--> optional trainer.test()
  |             +--> run_summary.json
  |             +--> artifacts.json
  |
  +--> cv_loop()
         |
         +--> run_experiment() x N folds
         +--> workflow_summary.json
         +--> workflow_artifacts.json


Python workflow
  |
  +--> run_main("python main.py ...")
  +--> read child summaries
  +--> aggregate
  +--> workflow_summary.json
  +--> workflow_artifacts.json
```


## 9. 最后建议

如果你现在要真正对照源码理解，建议你按下面方法读：

1. 先从 [`main.py`](/E:/projects/deeplab/main.py) 读到 [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)，只看 `train` 路径。
2. 然后再看 [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)，搞清楚 config 是怎么实例化成对象的。
3. 接着看一个 datamodule 示例和一个 model 示例。
4. 最后再回头看 [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py) 和 [`src/workflows/*.py`](/E:/projects/deeplab/src/workflows)，你就会发现它们本质上都只是“复用最小训练单元”。

  
