# Call Flow Guide

## Summary

这份文档回答三个问题：

1. `python main.py ...` 之后，代码按什么顺序执行
2. `train`、`cv`、workflow 三条路径分别经过哪些关键函数
3. `run_summary.json`、`artifacts.json`、workflow outputs 分别在哪里写出

## 1. 总体调用链

不论最终运行的是 `train`、`cv` 还是 workflow，底层都会回到同一个最小执行单元：

```text
Hydra CLI
  -> main.py
  -> validate_app_config()
  -> bootstrap_app()
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

## 2. train 路径

命令：

```bash
python main.py mode=train
```

调用顺序：

```text
main.py:main
  -> validate_app_config(cfg)
  -> bootstrap_app(cfg)
  -> train_loop(cfg)
  -> run_experiment(cfg)
```

关键文件：

- [`main.py`](/E:/projects/deeplab/main.py)
- [`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)
- [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)

### `main.py`

`main()` 只做：

1. 组合配置
2. 标准化配置
3. 执行 bootstrap
4. mode dispatch

### `run_experiment()`

`run_experiment()` 的核心流程是：

```text
run_experiment()
  -> _build_run_context()
  -> _with_runtime_context()
  -> _resolve_runtime_config()
  -> write_resolved_config()
  -> build_split_provider()
  -> build_model()
  -> build_datamodule()
  -> build_logger()
  -> build_callbacks()
  -> build_trainer()
  -> trainer.fit()
  -> optional trainer.test()
  -> build RunSummary
  -> save_json(run_summary.json)
  -> build ArtifactIndex
  -> save_json(artifacts.json)
```

这里的关键认知是：

- `run_experiment()` 才是真正的一次训练执行单元
- `train_loop()` 只是它的轻量包装

## 3. cv 路径

命令：

```bash
python main.py mode=cv mode.n_folds=5
```

调用顺序：

```text
main.py:main
  -> validate_app_config()
  -> bootstrap_app()
  -> cv_loop(cfg)
     -> for fold in range(n_folds):
          run_experiment(cfg, fold=fold)
     -> aggregate fold results
     -> write workflow_summary.json
     -> write workflow_artifacts.json
```

关键文件：

- [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)
- [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)

`cv` 本质上是：

> 多次 `run_experiment()` + 聚合

它不是一个全新的训练系统。

### fold-specific 输出路径

当 `run_experiment(cfg, fold=fold)` 被调用时，`RunContext` 会把这次 run 的 `output_dir` 变成 fold-specific 目录，因此每个 fold 都有自己的：

- `config.yaml`
- `run_summary.json`
- `artifacts.json`
- checkpoints

`cv_loop()` 再把这些子运行路径聚合起来，写 workflow 级别输出。

## 4. workflow 路径

当前复杂实验流程不通过新增 runtime mode 实现，而是通过：

- [`src/workflows/flat_cv.py`](/E:/projects/deeplab/src/workflows/flat_cv.py)
- [`src/workflows/nested_cv.py`](/E:/projects/deeplab/src/workflows/nested_cv.py)
- [`src/workflows/hpo_refit.py`](/E:/projects/deeplab/src/workflows/hpo_refit.py)

共同调用：

- [`src/workflows/common.py`](/E:/projects/deeplab/src/workflows/common.py)

workflow 层的标准套路是：

```text
workflow main()
  -> generate overrides
  -> run_main(overrides)
     -> subprocess: python main.py ...
  -> read child summaries/artifacts
  -> aggregate
  -> write_workflow_outputs()
```

这里的关键边界是：

- workflow 通过 subprocess 调用 runtime
- workflow 读取 child run artifacts
- workflow 不侵入 runtime 内部控制流

## 5. 结果是在哪里写出的

### 单次 run

[`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py) 会写：

- `config.yaml`
- `run_summary.json`
- `artifacts.json`

其中：

- `RunSummary` 是高频摘要
- `ArtifactIndex` 是完整 artifact 索引

### CV / Workflow 聚合层

[`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py) 和 [`src/workflows/common.py`](/E:/projects/deeplab/src/workflows/common.py) 会写：

- `workflow_summary.json`
- `workflow_artifacts.json`

聚合层不复制所有子运行内容，而是通过 child summary / child artifact path 做索引。

## 6. 建议的源码对照顺序

如果你想一边读文档一边看源码，建议顺序是：

1. [`main.py`](/E:/projects/deeplab/main.py)
2. [`src/config/schema.py`](/E:/projects/deeplab/src/config/schema.py)
3. [`src/core/bootstrap.py`](/E:/projects/deeplab/src/core/bootstrap.py)
4. [`src/core/train.py`](/E:/projects/deeplab/src/core/train.py)
5. [`src/core/cv.py`](/E:/projects/deeplab/src/core/cv.py)
6. [`src/core/runner.py`](/E:/projects/deeplab/src/core/runner.py)
7. [`src/utils/build.py`](/E:/projects/deeplab/src/utils/build.py)
