# DeepLab Training Framework

这是一套基于 Hydra + PyTorch Lightning 的训练框架。设计目标是把运行时收紧到两个最小执行单元：

- `train`：一次具体训练
- `cv`：按折重复执行的训练

更复杂的实验编排放到 `src/workflows/`，不要继续下沉到 `main.py`。

## 架构原则

- 配置驱动，不把实验分支写死在代码里
- 数据读取只在 datamodule 中发生
- split policy 由顶层 `split` 配置和 `split provider` 控制
- datamodule 不负责实现 CV / nested CV 的编排
- 运行时只保留真正被训练和聚合消费的 artifact

## split 约定

当前框架支持三类 split 语义：

- 非分层：`holdout`、`kfold`、`train_val_test_holdout`、`dev_test_kfold`
- group-aware：`group_*`
- stratified：`stratified_*`

group-aware 的意思是同一个 `group_id` 不能跨 split。  
stratified 的意思是各个 split 的标签分布尽量接近整体分布。  
`stratified_group_*` 同时要求：

- group 不泄漏
- 每个 split 必须包含全部类别
- 标签分布尽量接近整体分布

如果某个类别在候选数据里只覆盖很少的 group，框架会在 split 阶段直接失败，而不是把问题留到 metrics warning。

推荐配置示例：

```yaml
split:
  method: stratified_group_dev_test_kfold
  data_file: ${datamodule.init_args.data_cfg.data_file}
  group_id_column: patient_id
  label_column: label
  n_folds: ${mode.n_folds}
  test_ratio: 0.1
  seed: ${seed}
```

## 如何接入自己的项目

1. 准备 manifest / 样本表文件，至少包含样本路径、标签；如果需要 group-aware split，再加 group 列。
2. 实现 dataset，只负责单样本读取。
3. 实现 datamodule，只负责读取 `data_file`、消费 split、构建 dataloader。
4. 实现 task model，继承 `src/models/base_model.py` 的训练骨架。
5. 写 Hydra 配置：`model`、`datamodule`、`trainer`、`split`、`experiment`。
6. 先跑 `mode=train`。
7. 再跑 `mode=cv` 或 workflow。

推荐目录结构：

```text
src/models/
|-- backbones/
|-- heads/
`-- tasks/

src/datamodules/
|-- datasets/
`-- <your_datamodule>.py
```

## 运行示例

单次训练：

```bash
python main.py mode=train
```

使用实验模板：

```bash
python main.py experiment=tabular_baseline
```

CV：

```bash
python main.py mode=cv mode.n_folds=5
```

显式指定 split：

```bash
python main.py mode=train split.method=train_val_test_holdout split.test_ratio=0.2 split.val_ratio=0.2
```

## 产物

单次 run 会写出：

- `config.yaml`
- `run_summary.json`
- `artifacts.json`
- `split_manifest.yaml`

`cv` 和 workflow 还会写出：

- `workflow_summary.json`
- `workflow_artifacts.json`

## 推荐阅读顺序

1. `main.py`
2. `src/config/schema.py`
3. `src/core/runner.py`
4. `src/utils/build.py`
5. `src/datamodules/tabular_dm.py`
6. `src/models/tasks/tabular_classification.py`
7. `docs/CALL_FLOW.md`
8. `docs/MODULE.md`
