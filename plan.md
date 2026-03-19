# 重构执行计划 (Refactoring Execution Plan)

**生成日期**: 2026-03-16
**原则**: 每个任务高内聚、独立可验证，确保系统持续可用

---

## 执行策略总览

本计划将架构重构分解为 28 个小型任务，分为 6 个阶段。每个任务完成后，系统必须保持可运行状态。

```
Phase 1: 关键 Bug 修复          → 4 tasks (1-4)
Phase 2: 类型安全基础设施      → 6 tasks (5-10)
Phase 3: 测试基础设施           → 6 tasks (11-16)
Phase 4: 抽象基类架构           → 8 tasks (17-24)
Phase 5: 并行化交叉验证         → 5 tasks (25-29)
Phase 6: 包化结构与 CI/CD      → 4 tasks (30-33)
```

---

## Phase 1: 关键 Bug 修复 (快速胜利)

### Task 1: 修复 dummy_model.py 中的显式 Bug
**文件**: `src/models/dummy_model.py`
**风险**: 极低
**验证**: 运行一次测试看是否崩溃

**修改内容**:
```python
# Line 142: 将
self.test_confusion_matrix.reset()
# 改为
self.test_cm.reset()
```

**完成标准**:
- 代码语法正确
- 无 AttributeError

---

### Task 2: 更新 .gitignore 排除 IDE 配置
**文件**: `.gitignore`
**风险**: 无
**验证**: `git status` 确认 .idea/ 不再显示

**新增内容**:
```gitignore
# IDE
.vscode/
.idea/
*.swp
*.swo
*~
```

**完成标准**:
- .idea/ 目录在 git status 中不显示

---

### Task 3: 移除未使用的导入
**文件**: `src/models/dummy_model.py`
**风险**: 低
**验证**: 训练正常运行

**移除内容**:
```python
# 删除未使用的导入
import io          # 未使用
import tempfile    # 未使用
from PIL import Image  # 未使用
```

**保留内容**:
- `import os` - 用于 os.unlink
- `import matplotlib.pyplot as plt` - 用于绘图
- `import seaborn as sns` - 用于绘图
- `import mlflow` - 用于日志
- `import lightning as L` - 核心框架
- `import torch` / `torch.nn` / `torchmetrics` - 核心依赖
- `from hydra.utils import instantiate` - 配置实例化

**完成标准**:
- 导入整洁，无未使用模块
- 训练脚本正常运行

---

### Task 4: 消除通配符导入
**文件**: `src/core/train.py`, `src/core/cv.py`
**风险**: 低
**验证**: 训练和 CV 模式均正常

**修改 train.py**:
```python
# 将
from src.core.build import *
# 改为
from src.core.build import build_model, build_datamodule, build_logger, build_callbacks, build_trainer
```

**修改 cv.py**:
```python
# 将
from src.core.build import *
# 改为
from src.core.build import build_model, build_datamodule, build_logger, build_callbacks, build_trainer
```

**完成标准**:
- 明确导入所有使用的函数
- IDE 自动补全正常工作
- 训练和 CV 均正常运行

---

## Phase 2: 类型安全基础设施 (向前兼容)

### Task 5: 为 build.py 添加类型提示
**文件**: `src/core/build.py`
**风险**: 极低
**验证**: 代码语法正确，运行正常

**修改内容**:
```python
from typing import Any, List, Optional
from hydra.utils import instantiate
from omegaconf import DictConfig
from lightning import LightningModule, LightningDataModule, Trainer, Callback, Logger

def build_model(cfg: DictConfig) -> LightningModule:
    """构建模型

    Args:
        cfg: 模型配置字典，必须包含 _target_ 指向模型类

    Returns:
        LightningModule 实例
    """
    return instantiate(cfg)

def build_datamodule(
    cfg: DictConfig,
    fold: Optional[int] = None
) -> LightningDataModule:
    """构建数据模块

    Args:
        cfg: 数据配置字典
        fold: 交叉验证折数，None 表示不使用交叉验证

    Returns:
        LightningDataModule 实例
    """
    return instantiate(cfg, fold=fold)

def build_logger(cfg: DictConfig) -> Logger:
    """构建日志记录器

    Args:
        cfg: 日志配置，支持 mlflow 或 wandb

    Returns:
        Logger 实例
    """
    import mlflow
    mlflow.enable_system_metrics_logging()
    return instantiate(cfg)

def build_callbacks(cfg: DictConfig) -> List[Callback]:
    """构建回调函数列表

    Args:
        cfg: 回调配置字典

    Returns:
        Callback 对象列表
    """
    callbacks_list: List[Callback] = []
    for cb_name, cb_conf in cfg.items():
        if cb_conf is not None:
            cb: Callback = instantiate(cb_conf)
            callbacks_list.append(cb)
    return callbacks_list

def build_trainer(
    cfg: DictConfig,
    logger: Logger,
    callbacks: List[Callback]
) -> Trainer:
    """构建训练器

    Args:
        cfg: 训练器配置
        logger: 日志记录器
        callbacks: 回调函数列表

    Returns:
        Trainer 实例
    """
    return instantiate(cfg, logger=logger, callbacks=callbacks)
```

**完成标准**:
- 所有函数都有类型提示
- 代码可正常运行

---

### Task 6: 为 dummy_model.py 添加类型提示 (仅方法签名)
**文件**: `src/models/dummy_model.py`
**风险**: 极低
**验证**: 类型检查通过

**修改内容**:
```python
import torch
from typing import Any, Dict, Optional
import lightning as L

class DummyModel(L.LightningModule):
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # ... 现有代码保持不变

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... 现有代码保持不变

    def on_train_start(self) -> None:
        # ... 现有代码保持不变

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # ... 现有代码保持不变

    def on_train_epoch_end(self) -> None:
        # ... 现有代码保持不变

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        # ... 现有代码保持不变

    def on_validation_epoch_end(self) -> None:
        # ... 现有代码保持不变

    def test_step(self, batch: Any, batch_idx: int) -> None:
        # ... 现有代码保持不变

    def on_test_epoch_end(self) -> None:
        # ... 现有代码保持不变

    def configure_optimizers(self) -> Any:
        # ... 现有代码保持不变

    def log_confusion_matrix(self, cm: Any, epoch: int, stage: str = "val") -> None:
        # ... 现有代码保持不变
```

**完成标准**:
- 所有公共方法都有类型提示
- 训练正常运行

---

### Task 7: 为 dummy_dm.py 添加类型提示
**文件**: `src/datamodules/dummy_dm.py`
**风险**: 极低
**验证**: 数据加载正常

**修改内容**:
```python
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Tuple, Optional
import lightning as L
import numpy as np

class DummyDataset(Dataset):
    def __init__(self, num_samples: int = 20) -> None:
        # ... 现有代码保持不变

    def __len__(self) -> int:
        # ... 现有代码保持不变

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # ... 现有代码保持不变

class DummyDataModule(L.LightningDataModule):
    def __init__(self, data_cfg: Dict[str, Any], fold: Optional[int] = None, *args: Any, **kwargs: Any) -> None:
        # ... 现有代码保持不变

    def train_dataloader(self) -> DataLoader:
        # ... 现有代码保持不变

    def val_dataloader(self) -> DataLoader:
        # ... 现有代码保持不变

    def test_dataloader(self) -> DataLoader:
        # ... 现有代码保持不变
```

**完成标准**:
- 所有方法都有类型提示
- 训练正常运行

---

### Task 8: 为 train.py 和 cv.py 添加类型提示
**文件**: `src/core/train.py`, `src/core/cv.py`
**风险**: 极低
**验证**: 训练和 CV 正常运行

**修改内容**:
```python
# src/core/train.py
from typing import Any
from omegaconf import DictConfig

def train_loop(cfg: DictConfig) -> float:
    # ... 现有代码保持不变

# src/core/cv.py
from typing import Any
from omegaconf import DictConfig

def cv_loop(cfg: DictConfig) -> float:
    # ... 现有代码保持不变
```

**完成标准**:
- 函数签名有类型提示
- 训练和 CV 正常运行

---

### Task 9: 为 utils.py 添加类型提示
**文件**: `src/utils/utils.py`
**风险**: 极低
**验证**: 辅助功能正常

**修改内容**:
```python
from typing import Any, Dict
from omegaconf import DictConfig
from lightning.pytorch.loggers import Logger

def load_dotenv() -> None:
    # ... 现有代码保持不变

def set_seed(seed: int) -> None:
    # ... 现有代码保持不变

def snapshot_code(config_path: str, code_dir: str = "code") -> None:
    # ... 现有代码保持不变

def log_hyperparameters(logger: Logger, cfg: DictConfig) -> None:
    # ... 现有代码保持不变
```

**完成标准**:
- 所有函数都有类型提示
- 训练正常运行

---

### Task 10: 创建 pyproject.toml
**文件**: `pyproject.toml` (新建)
**风险**: 无
**验证**: `pip install -e .` 成功

**内容**:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "deeplab"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "hydra-core",
    "lightning",
    "omegaconf",
    "torch",
    "torchmetrics",
    "pyyaml>=6.0",
    "python-dotenv",
    "mlflow",
    "optuna",
    "matplotlib",
    "seaborn",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "pytest-mock",
    "mypy>=1.0",
    "ruff>=0.1.0",
    "pre-commit",
]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "W", "N"]
unfixable = ["F401"]

[tool.mypy]
python_version = "3.9"
strict = true
warn_return_any = true
warn_unused_configs = true
```

**完成标准**:
- 文件创建成功
- 不影响现有代码运行

---

## Phase 3: 测试基础设施

### Task 11: 创建测试目录结构
**新建目录**: `tests/`
**风险**: 无
**验证**: 目录创建成功

**目录结构**:
```
tests/
├── __init__.py
└── conftest.py
```

**完成标准**:
- 目录结构创建完成
- `tests/__init__.py` 为空文件

---

### Task 12: 创建 pytest fixtures (conftest.py)
**文件**: `tests/conftest.py` (新建)
**风险**: 无
**验证**: `pytest --collect-only` 能识别 fixtures

**内容**:
```python
import pytest
from omegaconf import DictConfig
import hydra
from hydra import compose, initialize

@pytest.fixture
def dummy_model_config() -> DictConfig:
    """提供模型配置 fixture"""
    return {
        "_target_": "src.models.dummy_model.DummyModel",
        "_recursive_": False,
        "model_cfg": {
            "model_name": "dummy_model",
            "num_classes": 2,
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 10,
            "eta_min": 1e-6,
        },
    }

@pytest.fixture
def dummy_datamodule_config() -> DictConfig:
    """提供数据模块配置 fixture"""
    return {
        "_target_": "src.datamodules.dummy_dm.DummyDataModule",
        "data_cfg": {
            "batch_size": 4,
            "num_workers": 0,
        },
    }

@pytest.fixture
def hydra_cfg():
    """提供完整的 Hydra 配置"""
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config", overrides=["model=cpath/dummy"])
        yield cfg
```

**完成标准**:
- fixtures 定义成功
- pytest 能识别这些 fixtures

---

### Task 13: 创建 builder 测试 (test_builders.py)
**文件**: `tests/test_builders.py` (新建)
**风险**: 无
**验证**: `pytest tests/test_builders.py -v` 通过

**内容**:
```python
import pytest
from src.core.build import build_model, build_datamodule

def test_build_model(dummy_model_config):
    """测试模型构建"""
    model = build_model(dummy_model_config)
    assert model is not None
    assert hasattr(model, 'forward')
    assert hasattr(model, 'training_step')

def test_build_datamodule(dummy_datamodule_config):
    """测试数据模块构建"""
    dm = build_datamodule(dummy_datamodule_config)
    assert dm is not None
    assert hasattr(dm, 'train_dataloader')
    assert hasattr(dm, 'val_dataloader')

def test_build_datamodule_with_fold(dummy_datamodule_config):
    """测试带 fold 参数的数据模块构建"""
    dm = build_datamodule(dummy_datamodule_config, fold=1)
    assert dm is not None
    assert dm.fold == 1
```

**完成标准**:
- 所有测试通过
- 不依赖外部服务 (如 MLFlow)

---

### Task 14: 创建模型测试 (test_models.py)
**文件**: `tests/test_models.py` (新建)
**风险**: 无
**验证**: `pytest tests/test_models.py -v` 通过

**内容**:
```python
import pytest
import torch
from src.models.dummy_model import DummyModel

def test_dummy_model_forward(dummy_model_config):
    """测试前向传播"""
    model = DummyModel(**dummy_model_config)
    x = torch.randn(2, 10, 512)  # batch=2, bag=10, feat=512
    logits = model(x)
    assert logits.shape == (2, 2)  # batch_size, num_classes

def test_dummy_model_training_step(dummy_model_config):
    """测试训练步骤"""
    model = DummyModel(**dummy_model_config)
    batch = (torch.randn(2, 10, 512), torch.randint(0, 2, (2,)))
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar loss

def test_dummy_model_metrics_update(dummy_model_config):
    """测试指标更新"""
    model = DummyModel(**dummy_model_config)
    logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    model.val_metrics.update(logits, labels)
    metrics = model.val_metrics.compute()
    assert 'val/acc' in metrics
```

**完成标准**:
- 所有测试通过
- 测试覆盖核心功能

---

### Task 15: 创建数据模块测试 (test_datamodules.py)
**文件**: `tests/test_datamodules.py` (新建)
**风险**: 无
**验证**: `pytest tests/test_datamodules.py -v` 通过

**内容**:
```python
import pytest
import torch
from src.datamodules.dummy_dm import DummyDataModule

def test_dummy_dataset():
    """测试 DummyDataset"""
    from src.datamodules.dummy_dm import DummyDataset
    dataset = DummyDataset(num_samples=10)
    assert len(dataset) == 10

    bag, label = dataset[0]
    assert bag.shape == (10, 512)
    assert label.shape == ()

def test_dummy_datamodule_train_dataloader(dummy_datamodule_config):
    """测试训练数据加载器"""
    dm = DummyDataModule(**dummy_datamodule_config)
    train_loader = dm.train_dataloader()

    batch = next(iter(train_loader))
    x, y = batch
    assert x.shape[0] == 4  # batch_size
    assert x.shape[1] == 10  # bag_size
    assert x.shape[2] == 512  # feature_dim

def test_dummy_datamodule_with_fold():
    """测试带 fold 参数的 DataModule"""
    config = {
        "data_cfg": {"batch_size": 4, "num_workers": 0},
    }
    dm = DummyDataModule(data_cfg=config, fold=2)
    assert dm.fold == 2
```

**完成标准**:
- 所有测试通过
- 数据加载逻辑验证正确

---

### Task 16: 验证所有测试通过
**任务**: 运行完整测试套件
**风险**: 无
**验证**: `pytest tests/ -v` 全部通过

**验证步骤**:
```bash
# 安装测试依赖
pip install pytest pytest-cov pytest-mock

# 运行所有测试
pytest tests/ -v

# 查看覆盖率
pytest tests/ --cov=src --cov-report=term-missing
```

**完成标准**:
- 所有测试通过
- 覆盖率报告生成成功
- 核心模块覆盖率 > 60%

---

## Phase 4: 抽象基类架构

### Task 17: 创建核心基础抽象类 (core/base.py)
**文件**: `src/core/base.py` (新建)
**风险**: 无
**验证**: 模块可导入

**内容**:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

def _resolve_metric_class(metric_name: str, **kwargs) -> Type:
    """解析指标类

    Args:
        metric_name: 指标名称
        **kwargs: 指标构造参数

    Returns:
        torchmetrics 指标类
    """
    import torchmetrics
    return getattr(torchmetrics, metric_name)

class BaseComponent(ABC):
    """所有可配置组件的基类"""
    pass
```

**完成标准**:
- 模块创建成功
- 不影响现有代码

---

### Task 18: 创建模型基类 (models/base_model.py)
**文件**: `src/models/base_model.py` (新建)
**风险**: 低
**验证**: 模块可导入

**内容**:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import lightning as L
import torch
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns


class BaseModel(L.LightningModule, ABC):
    """所有模型的基类，封装通用逻辑

    子类只需实现:
        - __init__(model_cfg, ...): 初始化模型结构
        - forward(x): 前向传播

    基类提供:
        - 指标管理
        - 通用训练/验证/测试流程
        - 最佳模型追踪
        - 混淆矩阵记录
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        metrics_cfg: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_cfg = model_cfg
        self.metrics_cfg = metrics_cfg or self._default_metrics()
        self.num_classes = model_cfg.get('num_classes', 2)

        # 设置指标
        self._setup_metrics()

        # 追踪最佳模型
        self.best_metric: float = float('inf') if self.monitor.startswith('loss') else float('-inf')
        self.best_epoch: int = -1
        self.monitor: str = "val/loss"  # 将在 on_train_start 中更新

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """子类必须实现的前向传播

        Args:
            x: 输入张量

        Returns:
            输出张量 (通常是 logits)
        """
        pass

    def _default_metrics(self) -> Dict[str, Any]:
        """可覆写的默认指标配置"""
        return {
            'accuracy': {'task': 'multiclass', 'num_classes': self.num_classes},
            'precision': {'task': 'multiclass', 'num_classes': self.num_classes},
            'recall': {'task': 'multiclass', 'num_classes': self.num_classes},
            'f1': {'task': 'multiclass', 'num_classes': self.num_classes},
            'auc': {'task': 'multiclass', 'num_classes': self.num_classes},
        }

    def _setup_metrics(self) -> None:
        """根据配置设置指标"""
        base_metrics = {}
        for name, kwargs in self.metrics_cfg.items():
            metric_class = getattr(torchmetrics, name.capitalize())
            base_metrics[name.lower()] = metric_class(**kwargs)

        metrics = torchmetrics.MetricCollection(base_metrics)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # 混淆矩阵
        self.val_cm = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=self.num_classes
        )
        self.test_cm = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=self.num_classes
        )

    def on_train_start(self) -> None:
        """训练开始时获取监控指标"""
        if self.trainer and self.trainer.checkpoint_callback:
            self.monitor = self.trainer.checkpoint_callback.monitor or self.monitor
            # 初始化最佳指标方向
            if 'loss' in self.monitor.lower():
                self.best_metric = float('inf')
            else:
                self.best_metric = float('-inf')

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """训练步骤 - 子类可覆写"""
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_metrics.update(logits, y)
        return loss

    def on_train_epoch_end(self) -> None:
        """训练 epoch 结束"""
        output = self.train_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """验证步骤"""
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.val_metrics.update(logits, y)
        self.val_cm.update(logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """验证 epoch 结束"""
        output = self.val_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

        # 获取当前监控指标
        current_metric = output.get(self.monitor, None)
        if current_metric is None:
            current_metric = self.trainer.callback_metrics.get(self.monitor)

        if current_metric is not None:
            current_value = current_metric.item() if hasattr(current_metric, 'item') else current_metric
            is_better = current_value < self.best_metric if 'loss' in self.monitor.lower() else current_value > self.best_metric

            if is_better:
                self.best_metric = current_value
                self.best_epoch = self.current_epoch

                # 记录混淆矩阵
                cm = self.val_cm.compute().cpu().numpy()
                self._log_confusion_matrix(cm, epoch=self.current_epoch, stage='val')

                self.log("best_metric", self.best_metric, on_step=False, on_epoch=True)
                self.log("best_epoch", self.best_epoch, on_step=False, on_epoch=True)

        self.val_cm.reset()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """测试步骤"""
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)

        self.test_metrics.update(logits, y)
        self.test_cm.update(logits, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """测试 epoch 结束"""
        output = self.test_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

        cm = self.test_cm.compute().cpu().numpy()
        self._log_confusion_matrix(cm, epoch=self.current_epoch, stage="test")

        self.test_cm.reset()

    def _log_confusion_matrix(self, cm: Any, epoch: int, stage: str = "val") -> None:
        """记录混淆矩阵图像到 logger

        Args:
            cm: 混淆矩阵数组
            epoch: 当前 epoch
            stage: 阶段名称 ('val' 或 'test')
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        if stage == "test":
            title = 'Confusion Matrix (Test Set)'
            file_name = "confusion_matrices/test_cm.png"
        else:
            title = f'Confusion Matrix (Val Set, Epoch {epoch})'
            file_name = f"confusion_matrices/val_cm.png"

        ax.set_title(title)

        if self.logger is None:
            plt.close(fig)
            return

        # WandB 记录
        if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            import wandb
            self.logger.experiment.log({
                f"{stage}_confusion_matrix": wandb.Image(fig, caption=title)
            }, step=epoch)

        # MLflow 记录
        elif isinstance(self.logger, L.pytorch.loggers.MLFlowLogger):
            import os, tempfile
            client = self.logger.experiment
            run_id = self.logger.run_id

            try:
                client.log_figure(run_id, fig, file_name)
            except Exception as e:
                print(f"MLflow log_figure failed: {e}. Using fallback...")
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    fig.savefig(tmp.name)
                    client.log_artifact(run_id, tmp.name, artifact_path="confusion_matrices")
                os.unlink(tmp.name)

        plt.close(fig)

    def configure_optimizers(self) -> Any:
        """配置优化器和学习率调度器 - 子类可覆写"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        return [optimizer], [scheduler]
```

**完成标准**:
- 模块创建成功
- 不影响现有代码

---

### Task 19: 创建日志处理器基类 (models/loggers/base.py)
**文件**: `src/models/loggers/base.py` (新建)
**风险**: 无
**验证**: 模块可导入

**内容**:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import lightning as L


class BaseLoggerHandler(ABC):
    """日志处理器基类

    使用策略模式处理不同类型的 logger
    """

    @abstractmethod
    def log(self, artifacts: Dict[str, Any], stage: str = "val") -> None:
        """记录工制品

        Args:
            artifacts: 工制品字典，包含图像、图表等
            stage: 阶段名称
        """
        pass

    @abstractmethod
    def log_figure(self, figure: Any, file_path: str, caption: str = "") -> None:
        """记录图像

        Args:
            figure: matplotlib 图像对象
            file_path: 文件路径
            caption: 图像标题
        """
        pass
```

**完成标准**:
- 模块创建成功
- 基类定义清晰

---

### Task 20: 创建 WandB 日志处理器 (models/loggers/wandb_handler.py)
**文件**: `src/models/loggers/wandb_handler.py` (新建)
**风险**: 无
**验证**: 模块可导入

**内容**:
```python
from typing import Any, Dict
import wandb
from .base import BaseLoggerHandler


class WandBLoggerHandler(BaseLoggerHandler):
    """WandB 日志处理器"""

    def __init__(self, logger):
        self.logger = logger

    def log(self, artifacts: Dict[str, Any], stage: str = "val") -> None:
        """记录工制品"""
        self.logger.experiment.log(artifacts)

    def log_figure(self, figure: Any, file_path: str, caption: str = "") -> None:
        """记录图像"""
        import wandb
        self.logger.experiment.log({
            file_path: wandb.Image(figure, caption=caption)
        })
```

**完成标准**:
- 模块创建成功
- 实现 BaseLoggerHandler 接口

---

### Task 21: 创建 MLFlow 日志处理器 (models/loggers/mlflow_handler.py)
**文件**: `src/models/loggers/mlflow_handler.py` (新建)
**风险**: 无
**验证**: 模块可导入

**内容**:
```python
import os
import tempfile
from typing import Any, Dict
from .base import BaseLoggerHandler


class MLFlowLoggerHandler(BaseLoggerHandler):
    """MLFlow 日志处理器"""

    def __init__(self, logger):
        self.logger = logger

    def log(self, artifacts: Dict[str, Any], stage: str = "val") -> None:
        """记录工制品"""
        client = self.logger.experiment
        run_id = self.logger.run_id

        for key, value in artifacts.items():
            if hasattr(value, 'savefig'):
                # 是 matplotlib 图像
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    value.savefig(tmp.name)
                    client.log_artifact(run_id, tmp.name, artifact_path=key)
                os.unlink(tmp.name)
            else:
                client.log_param(run_id, key, value)

    def log_figure(self, figure: Any, file_path: str, caption: str = "") -> None:
        """记录图像"""
        client = self.logger.experiment
        run_id = self.logger.run_id

        try:
            client.log_figure(run_id, figure, file_path)
        except Exception:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                figure.savefig(tmp.name)
                client.log_artifact(run_id, tmp.name, artifact_path="confusion_matrices")
            os.unlink(tmp.name)
```

**完成标准**:
- 模块创建成功
- 实现 BaseLoggerHandler 接口

---

### Task 22: 创建日志处理器工厂 (models/loggers/__init__.py)
**文件**: `src/models/loggers/__init__.py` (新建)
**风险**: 无
**验证**: 工厂模式工作正常

**内容**:
```python
from typing import Optional
import lightning as L
from .base import BaseLoggerHandler
from .wandb_handler import WandBLoggerHandler
from .mlflow_handler import MLFlowLoggerHandler


class LoggerFactory:
    """日志处理器工厂"""

    _handlers: dict = {
        L.pytorch.loggers.WandbLogger: WandBLoggerHandler,
        L.pytorch.loggers.MLFlowLogger: MLFlowLoggerHandler,
    }

    @classmethod
    def create(cls, logger: Any) -> Optional[BaseLoggerHandler]:
        """根据 logger 类型创建对应的处理器

        Args:
            logger: Lightning logger 实例

        Returns:
            BaseLoggerHandler 实例，如果类型不支持则返回 None
        """
        for logger_type, handler_class in cls._handlers.items():
            if isinstance(logger, logger_type):
                return handler_class(logger)
        return None

    @classmethod
    def register_handler(cls, logger_type: type, handler_class: type) -> None:
        """注册新的处理器类型

        Args:
            logger_type: logger 类型
            handler_class: 处理器类
        """
        cls._handlers[logger_type] = handler_class


__all__ = ['BaseLoggerHandler', 'WandBLoggerHandler', 'MLFlowLoggerHandler', 'LoggerFactory']
```

**完成标准**:
- 工厂类创建成功
- 支持注册新处理器类型

---

### Task 23: 重构 dummy_model.py 继承 BaseModel
**文件**: `src/models/dummy_model.py`
**风险**: 中
**验证**: 训练正常运行

**修改内容**:
```python
import torch
import torch.nn as nn
from typing import Any, Dict
from .base_model import BaseModel


class DummyModel(BaseModel):
    """模板模型类

    继承 BaseModel 后，只需实现:
        - __init__: 初始化模型结构
        - forward: 前向传播

    其余逻辑由基类处理
    """
    def __init__(self, model_cfg: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # 初始化分类器
        self.classifier = None  # 占位，将在 super().__init__ 后设置

        # 调用基类初始化 (设置指标、追踪逻辑等)
        super().__init__(model_cfg, *args, **kwargs)

        # 设置分类器
        self.classifier = nn.Linear(512, model_cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入张量，shape (batch_size, bag_size, feature_dim)

        Returns:
            输出 logits，shape (batch_size, num_classes)
        """
        bag_feature = torch.mean(x, dim=1)
        logits = self.classifier(bag_feature)
        return logits

    # 不需要覆写: training_step, on_train_epoch_end,
    #           validation_step, on_validation_epoch_end,
    #           test_step, on_test_epoch_end, _log_confusion_matrix
    # 这些方法由 BaseModel 提供

    def configure_optimizers(self):
        """配置优化器和调度器 (从配置文件读取)"""
        from hydra.utils import instantiate
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
        return [optimizer], [scheduler]
```

**完成标准**:
- 代码重构完成
- 训练正常运行
- 混淆矩阵正常记录

---

### Task 24: 验证重构后训练正常
**任务**: 运行完整训练和验证
**风险**: 低
**验证**: 训练和 CV 模式均正常

**验证步骤**:
```bash
# 训练模式
python main.py experiment_name=test_retrain run_name=test1

# CV 模式
python main.py mode=cv experiment_name=test_cv run_name=cv1

# 测试模式
python main.py test_after_train=true experiment_name=test_test run_name=test1
```

**完成标准**:
- 训练模式正常运行
- CV 模式正常运行
- 测试模式正常运行
- 混淆矩阵正确记录

---

## Phase 5: 并行化交叉验证

### Task 25: 创建 GPU 资源管理器 (core/resource_manager.py)
**文件**: `src/core/resource_manager.py` (新建)
**风险**: 低
**验证**: 模块可导入并正常工作

**内容**:
```python
from typing import List
import torch


class GPUResourceManager:
    """GPU 资源调度器

    避免多进程/多线程冲突，合理分配 GPU 资源
    """

    @staticmethod
    def get_available_gpus() -> int:
        """获取可用的 GPU 数量

        Returns:
            GPU 数量
        """
        return torch.cuda.device_count()

    @staticmethod
    def allocate_gpus(fold: int, total_folds: int, max_gpus: int = None) -> List[int]:
        """根据折数分配 GPU

        Args:
            fold: 当前折数
            total_folds: 总折数
            max_gpus: 最多使用的 GPU 数量，None 则使用所有可用 GPU

        Returns:
            分配的 GPU ID 列表
        """
        available_gpus = GPUResourceManager.get_available_gpus()
        if max_gpus is not None:
            available_gpus = min(available_gpus, max_gpus)

        if available_gpus == 0:
            return []  # CPU 模式

        if available_gpus >= total_folds:
            # 每折独占一个 GPU
            return [fold]

        # GPU 少于折数，轮询分配
        return [fold % available_gpus]

    @staticmethod
    def set_device(gpu_ids: List[int]) -> None:
        """设置当前进程使用的 GPU

        Args:
            gpu_ids: GPU ID 列表
        """
        if gpu_ids:
            torch.cuda.set_device(gpu_ids[0])
```

**完成标准**:
- 模块创建成功
- 测试 GPU 分配逻辑

---

### Task 26: 并行化 CV 循环 (cv.py)
**文件**: `src/core/cv.py`
**风险**: 中
**验证**: CV 结果与串行版本一致

**修改内容**:
```python
from typing import Dict, Any, Optional
import concurrent.futures
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.core.resource_manager import GPUResourceManager


def _train_single_fold(cfg: DictConfig, fold: int) -> Dict[str, Any]:
    """训练单折

    Args:
        cfg: 配置对象
        fold: 折数

    Returns:
        包含折数、分数、checkpoint 路径的字典
    """
    # 复制配置，避免修改原配置
    fold_cfg = cfg.copy()

    # 调整 checkpoint 路径
    if 'callbacks' in fold_cfg and 'model_checkpoint' in fold_cfg.callbacks:
        checkpoint_cfg = fold_cfg.callbacks.model_checkpoint
        if 'dirpath' in checkpoint_cfg:
            checkpoint_cfg.dirpath = f"{checkpoint_cfg.dirpath}/fold_{fold}"

    # 分配 GPU
    gpu_ids = GPUResourceManager.allocate_gpus(
        fold=fold,
        total_folds=cfg.mode.n_folds,
        max_gpus=cfg.mode.get('max_gpus', None)
    )
    if gpu_ids:
        GPUResourceManager.set_device(gpu_ids)
        # 修改 trainer 配置的 devices
        if 'trainer' in fold_cfg and 'devices' in fold_cfg.trainer:
            fold_cfg.trainer.devices = gpu_ids

    # 调整 logger run 名称
    if 'logger' in fold_cfg and 'run_name' in fold_cfg.logger:
        fold_cfg.logger.run_name = f"{fold_cfg.logger.run_name}_fold{fold}"

    # 构建组件
    model = build_model(fold_cfg.model)
    datamodule = build_datamodule(fold_cfg.datamodule, fold)
    logger = build_logger(fold_cfg.logger)
    callbacks = build_callbacks(fold_cfg.callbacks)
    trainer = build_trainer(fold_cfg.trainer, logger, callbacks)

    # 训练
    trainer.fit(model, datamodule)

    # 获取结果
    score = None
    if trainer.callback_metrics and cfg.monitor in trainer.callback_metrics:
        score = trainer.callback_metrics[cfg.monitor].item()

    checkpoint_path = ""
    if trainer.checkpoint_callback:
        checkpoint_path = trainer.checkpoint_callback.best_model_path

    return {
        'fold': fold,
        'score': score,
        'checkpoint_path': checkpoint_path,
    }


def cv_loop(cfg: DictConfig) -> float:
    """并行化 K 折交叉验证

    Args:
        cfg: 配置对象

    Returns:
        平均验证分数
    """
    n_splits = cfg.mode.get('n_folds', 5)
    max_workers = cfg.mode.get('max_workers', 1)

    # 如果 max_workers 为 1 或只有 1 个 GPU，使用串行执行
    if max_workers <= 1 or GPUResourceManager.get_available_gpus() <= 1:
        results = [_train_single_fold(cfg, fold) for fold in range(n_splits)]
    else:
        # 并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_train_single_fold, cfg, fold) for fold in range(n_splits)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # 收集结果
    val_scores = [r['score'] for r in results if r['score'] is not None]

    if not val_scores:
        return 0.0

    return float(np.mean(val_scores))
```

**完成标准**:
- 代码修改完成
- 并行 CV 正常运行

---

### Task 27: 更新 CV 配置文件
**文件**: `conf/mode/cv.yaml`
**风险**: 低
**验证**: 配置文件格式正确

**修改内容**:
```yaml
name: cv
n_folds: 5
max_workers: 2  # 并发折数，根据 GPU 数量调整
max_gpus: null  # 最多使用的 GPU 数量，null 表示使用所有可用 GPU
```

**完成标准**:
- 配置文件更新完成
- YAML 格式正确

---

### Task 28: 验证并行 CV 结果正确性
**任务**: 对比串行和并行 CV 结果
**风险**: 低
**验证**: 结果一致

**验证步骤**:
```bash
# 串行 CV (max_workers=1)
python main.py mode=cv mode.max_workers=1 experiment_name=serial_cv run_name=run1

# 并行 CV (max_workers=2)
python main.py mode=cv mode.max_workers=2 experiment_name=parallel_cv run_name=run1

# 比较两个结果的平均分数
```

**完成标准**:
- 并行和串行结果一致 (允许小范围浮点误差)
- 并行执行速度明显提升

---

### Task 29: 创建并行 CV 测试
**文件**: `tests/test_cv.py` (新建)
**风险**: 无
**验证**: 测试通过

**内容**:
```python
import pytest
from unittest.mock import Mock, patch
from src.core.cv import _train_single_fold
from src.core.resource_manager import GPUResourceManager


def test_gpu_resource_manager():
    """测试 GPU 资源管理器"""
    # 测试独占分配
    gpus = GPUResourceManager.allocate_gpus(fold=0, total_folds=3, max_gpus=4)
    assert gpus == [0]

    gpus = GPUResourceManager.allocate_gpus(fold=1, total_folds=3, max_gpus=4)
    assert gpus == [1]

    # 测试轮询分配
    gpus = GPUResourceManager.allocate_gpus(fold=0, total_folds=5, max_gpus=2)
    assert gpus == [0]

    gpus = GPUResourceManager.allocate_gpus(fold=2, total_folds=5, max_gpus=2)
    assert gpus == [0]

    gpus = GPUResourceManager.allocate_gpus(fold=3, total_folds=5, max_gpus=2)
    assert gpus == [1]

@patch('src.core.cv.build_trainer')
@patch('src.core.cv.build_callbacks')
@patch('src.core.cv.build_logger')
@patch('src.core.cv.build_datamodule')
@patch('src.core.cv.build_model')
def test_train_single_fold(mock_build_model, mock_build_dm, mock_build_logger,
                            mock_build_callbacks, mock_build_trainer, dummy_model_config):
    """测试单折训练"""
    # 设置 mock
    mock_model = Mock()
    mock_model.configure_optimizers.return_value = ([Mock()], [])
    mock_build_model.return_value = mock_model

    mock_dm = Mock()
    mock_build_dm.return_value = mock_dm

    mock_logger = Mock()
    mock_logger.run_id = "test_run"
    mock_build_logger.return_value = mock_logger

    mock_callbacks = []
    mock_build_callbacks.return_value = mock_callbacks

    mock_trainer = Mock()
    mock_trainer.callback_metrics = {"val/loss": 0.5}
    mock_trainer.checkpoint_callback = Mock()
    mock_trainer.checkpoint_callback.best_model_path = "/path/to/checkpoint.ckpt"
    mock_build_trainer.return_value = mock_trainer

    # 创建配置
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "model": dummy_model_config,
        "datamodule": {"data_cfg": {"batch_size": 4}},
        "logger": {"run_name": "test"},
        "callbacks": {"model_checkpoint": {"dirpath": "checkpoints"}},
        "trainer": {"max_epochs": 1},
        "monitor": "val/loss",
        "mode": {"n_folds": 5, "max_workers": 2}
    })

    # 执行
    result = _train_single_fold(cfg, fold=0)

    # 验证
    assert result['fold'] == 0
    assert result['score'] == 0.5
    assert 'fold_0' in mock_callbacks[0].dirpath
    assert 'fold0' in cfg.logger.run_name
```

**完成标准**:
- 测试通过
- 覆盖关键逻辑

---

## Phase 6: 包化结构与 CI/CD

### Task 30: 更新 .gitignore 添加更多排除项
**文件**: `.gitignore`
**风险**: 无
**验证**: git status 正常

**新增内容**:
```gitignore
# Testing
.pytest_cache/
.coverage
htmlcov/

# Type Checking
.mypy_cache/
.dmypy.json
dmypy.json
```

**完成标准**:
- .gitignore 完整
- 不应提交的文件都被排除

---

### Task 31: 创建 .pre-commit-config.yaml
**文件**: `.pre-commit-config.yaml` (新建)
**风险**: 无
**验证**: pre-commit hooks 正常

**内容**:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies:
          - types-PyYAML
          - types-requests
```

**完成标准**:
- 文件创建成功
- pre-commit hooks 配置正确

---

### Task 32: 创建 CI/CD 工作流 (可选)
**文件**: `.github/workflows/ci.yml` (新建)
**风险**: 无
**验证**: CI 配置正确

**内容**:
```yaml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]

      - name: Run ruff
        run: ruff check src/ tests/

      - name: Run mypy
        run: mypy src/
```

**完成标准**:
- CI 配置文件创建成功
- 配置结构清晰

---

### Task 33: 最终验证与文档更新
**任务**: 运行完整测试套件并更新文档
**风险**: 无
**验证**: 所有测试通过

**验证步骤**:
```bash
# 运行所有测试
pytest tests/ -v --cov=src --cov-report=html

# 运行代码检查
ruff check src/ tests/
mypy src/

# 运行完整训练流程
python main.py experiment_name=final_test run_name=run1

# 运行并行 CV
python main.py mode=cv mode.max_workers=2 experiment_name=final_cv run_name=run1
```

**更新文档**:
- 更新 CLAUDE.md 添加新架构说明
- 更新 README.md (如果需要)

**完成标准**:
- 所有测试通过
- 代码检查无错误
- 训练和 CV 正常运行
- 文档更新完成

---

## 完成检查清单

- [x] Phase 1: 关键 Bug 修复 (4 tasks) - ✅ 已完成 2026-03-16
- [x] Phase 2: 类型安全基础设施 (6 tasks) - ✅ 已完成 2026-03-16
- [ ] Phase 3: 测试基础设施 (6 tasks)
- [ ] Phase 4: 抽象基类架构 (8 tasks)
- [ ] Phase 5: 并行化交叉验证 (5 tasks)
- [ ] Phase 6: 包化结构与 CI/CD (4 tasks)

**总计**: 33 个任务

---

## 回滚策略

如果某个任务导致问题：
1. 使用 `git stash` 或 `git checkout` 回滚该任务的更改
2. 重新验证系统状态
3. 分析问题并调整任务内容
4. 重新执行任务

每个任务完成后，建议提交一个 commit，方便回滚：
```bash
git add .
git commit -m "Refactor: 完成 Task XX - <任务描述>"
```
