# 架构评估报告
# Architecture Assessment Report

生成日期: 2026-03-16
评估者: Chief Architect Review

---

## 1. 当前代码组织与架构模式评估

### 1.1 架构优势

| 维度 | 评估 | 说明 |
|------|------|------|
| **关注点分离** | ✅ 良好 | `src/` 按 core/models/datamodules/utils 清晰分层 |
| **配置驱动** | ✅ 优秀 | Hydra 配置系统与代码完全解耦，支持灵活覆盖 |
| **模块化设计** | ✅ 良好 | Builder 模式 (`src/core/build.py`) 实现组件解耦 |
| **可扩展性** | ⚠️ 中等 | 新增模型需要复制模板，缺乏基类抽象 |
| **测试覆盖** | ❌ 缺失 | 无测试基础设施 |

### 1.2 依赖关系分析

```
main.py
├── src.utils.utils (load_dotenv, snapshot_code)
├── src.core.train (train_loop)
│   ├── src.core.build (*)  ⚠️ 通配符导入
│   └── src.utils.utils (log_hyperparameters)
└── src.core.cv (cv_loop)
    ├── src.core.build (*)  ⚠️ 通配符导入
    └── src.utils.utils (log_hyperparameters)
```

**耦合度评估**: 中等耦合，无明显循环依赖

### 1.3 高内聚低耦合评估

- **高内聚**: ✅ 各模块职责相对清晰
- **低耦合**: ⚠️ 存在问题：
  - `build.py` 使用通配符导出，命名空间污染
  - 模型类混合了业务逻辑与日志/可视化逻辑

---

## 2. 设计反模式与技术债务

### 2.1 反模式 (Anti-Patterns)

| 反模式 | 位置 | 严重性 | 影响 |
|--------|------|--------|------|
| **Wildcard Import** | `src/core/train.py:1`, `cv.py:5` | 🔴 高 | 命名空间污染，IDE 无法自动补全，代码可读性差 |
| **Mixed Concerns** | `src/models/dummy_model.py` | 🟠 中 | 模型类包含日志、可视化逻辑，违反单一职责原则 |
| **Type Coercion** | `trainer.callback_metrics.get(monitor).item()` | 🟠 中 | 未检查 None 就调用 .item()，可能崩溃 |
| **Hardcoded Type Checks** | `isinstance(self.logger, WandbLogger)` | 🟠 中 | 违反开闭原则，新增 logger 需修改模型代码 |
| **Magic Strings** | `"val/loss"`, `"train/loss"` | 🟡 低 | 散布各处，重构时容易遗漏 |
| **Unused Imports** | `dummy_model.py:1-6` | 🟡 低 | `io`, `tempfile`, `PIL.Image` 未使用 |
| **Inconsistent Aliases** | `lightning as L` vs `pytorch_lightning as pl` | 🟡 低 | 代码风格不统一 |

### 2.2 技术债务 (Technical Debt)

| 债务项 | 类别 | 优先级 | 说明 |
|--------|------|--------|------|
| **无测试基础设施** | 质量 | P0 | 无法验证代码正确性，重构风险高 |
| **缺少类型提示** | 可维护性 | P1 | IDE 辅助功能受限，难以早期发现错误 |
| **配置与代码不匹配** | 结构 | P1 | `conf/model/cpath/` 按领域分类，代码却是通用的 `dummy` |
| **顺序式 CV 循环** | 性能 | P1 | K 折交叉验证串行执行，浪费 GPU 资源 |
| **缺少抽象基类** | 可扩展性 | P1 | 新模型需复制模板，维护成本高 |
| **.idea/ 未被忽略** | 卫生 | P2 | IDE 配置文件不应提交到版本控制 |
| **空文件夹** | 卫生 | P2 | `data/` 为空但存在于仓库中 |

### 2.3 潜在运行时风险

```python
# src/models/dummy_model.py:99-100
current_metric = self.trainer.callback_metrics.get(self.monitor)
if current_metric is not None:
    current_metric = current_metric.item()  # ⚠️ 如果是 tensor.item() 没问题
```

```python
# src/models/dummy_model.py:142
self.test_confusion_matrix.reset()  # ❌ Bug: 属性名是 test_cm 不是 test_confusion_matrix
```

---

## 3. 高优先级架构优化建议

### 建议 1: 建立抽象基类与接口分离

**当前问题:**
- 新增模型需要复制整个模板
- 指标逻辑、日志逻辑在每个模型中重复
- 监控逻辑与 logger 类型硬编码耦合

**优化方案:**

```
src/
├── core/
│   ├── base.py              # 新增: 基础抽象类
│   ├── build.py
│   ├── ...
├── models/
│   ├── base_model.py        # 新增: 模型基类
│   ├── metrics.py           # 新增: 指标配置独立化
│   ├── loggers/             # 新增: 日志处理器独立化
│   │   ├── base.py
│   │   ├── wandb.py
│   │   └── mlflow.py
│   └── dummy_model.py
```

**关键改动:**

```python
# src/models/base_model.py
from abc import ABC, abstractmethod
import lightning as L
import torch.nn as nn

class BaseModel(L.LightningModule, ABC):
    """所有模型的基类，封装通用逻辑"""

    def __init__(self, model_cfg, metrics_cfg=None, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.metrics_cfg = metrics_cfg or self._default_metrics()
        self._setup_metrics()

    @abstractmethod
    def forward(self, x):
        """子类必须实现的前向传播"""
        pass

    def _default_metrics(self):
        """可覆写的默认指标配置"""
        return {
            'accuracy': torchmetrics.Accuracy,
            'precision': torchmetrics.Precision,
            # ...
        }

    def log_artifacts(self, artifacts_dict, stage='val'):
        """统一的日志记录接口，支持多种 logger"""
        if self.logger is None:
            return
        # 使用策略模式处理不同 logger
        logger_handler = LoggerFactory.create(self.logger)
        logger_handler.log(artifacts_dict, stage)

# src/models/loggers/base.py
class BaseLoggerHandler(ABC):
    @abstractmethod
    def log(self, artifacts, stage):
        pass
```

**预期收益:**
- 新模型只需实现 `__init__` 和 `forward`
- 指标/日志逻辑可配置化，一处修改全局生效
- 新增 logger 类型无需修改模型代码

---

### 建议 2: 实现并行化交叉验证与资源调度

**当前问题:**
```python
# src/core/cv.py:15-49
for fold in range(n_splits):  # ⚠️ 串行执行，GPU 利用率低
    model = build_model(fold_cfg.model)
    datamodule = build_datamodule(fold_cfg.datamodule, fold)
    # ...
    trainer.fit(model, datamodule)
```

**优化方案:**

```python
# src/core/cv.py
import concurrent.futures
from typing import List

def cv_loop(cfg):
    """并行化 K 折交叉验证"""
    n_splits = cfg.mode.get('n_folds', 5)
    max_workers = cfg.mode.get('max_workers', 1)  # 可配置并发度

    def train_single_fold(fold: int) -> dict:
        """训练单折，返回结果字典"""
        fold_cfg = cfg.copy()
        # ... 构建组件
        trainer.fit(model, datamodule)
        return {
            'fold': fold,
            'score': trainer.callback_metrics.get(cfg.monitor),
            'checkpoint_path': trainer.checkpoint_callback.best_model_path
        }

    # 并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(train_single_fold, fold) for fold in range(n_splits)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # 收集结果
    val_scores = [r['score'] for r in results]
    return float(np.mean(val_scores))

# 配置文件新增
# conf/mode/cv.yaml
name: cv
n_folds: 5
max_workers: 2  # 并发折数，根据 GPU 数量调整
```

**资源调度考虑:**
```python
# src/core/resource_manager.py
class GPUResourceManager:
    """GPU 资源调度器，避免多进程冲突"""

    @staticmethod
    def allocate_gpus(fold: int, total_folds: int) -> List[int]:
        """根据折数分配 GPU"""
        if torch.cuda.device_count() >= total_folds:
            return [fold]
        # 默认策略：偶数折用 GPU 0，奇数折用 GPU 1
        return [fold % min(torch.cuda.device_count(), 2)]
```

**预期收益:**
- 2 GPU 环境: 2 倍速度提升
- 4 GPU 环境: 4 倍速度提升
- 支持灵活配置并发度

---

### 建议 3: 引入测试基础设施与类型安全

**当前问题:**
- 无单元测试
- 缺少类型提示
- 配置验证缺失

**优化方案:**

#### 3.1 项目结构改造

```
deeplab/
├── pyproject.toml              # 新增: 项目元数据与依赖
├── tests/                      # 新增: 测试目录
│   ├── __init__.py
│   ├── conftest.py            # pytest fixtures
│   ├── test_builders.py
│   ├── test_models.py
│   └── test_datamodules.py
├── src/
│   └── deeplab/               # 重命名: 包化结构
│       ├── __init__.py
│       ├── core/
│       ├── models/
│       └── ...
├── .gitignore                 # 更新: 忽略更多文件
└── .pre-commit-config.yaml    # 新增: 代码质量检查
```

#### 3.2 pyproject.toml

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

#### 3.3 示例测试文件

```python
# tests/test_builders.py
import pytest
from hydra import compose, initialize
from src.deeplab.core.build import build_model, build_datamodule

def test_build_model():
    """测试模型构建"""
    with initialize(config_path="../../conf"):
        cfg = compose(config_name="config", overrides=["model=cpath/dummy"])
        model = build_model(cfg.model)
        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'training_step')

def test_build_datamodule():
    """测试数据模块构建"""
    with initialize(config_path="../../conf"):
        cfg = compose(config_name="config", overrides=["datamodule=cpath/dummy"])
        dm = build_datamodule(cfg.datamodule)
        assert dm is not None
        assert hasattr(dm, 'train_dataloader')
        assert hasattr(dm, 'val_dataloader')

@pytest.mark.parametrize("fold", [0, 1, 2, 3, 4])
def test_cv_datamodule_folds(fold):
    """测试交叉验证数据模块的 fold 参数"""
    # ...
```

#### 3.4 类型提示示例

```python
# src/deeplab/core/build.py
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

#### 3.5 .gitignore 更新

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Project
.env
outputs/
data/
*.log
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/

# Type Checking
.mypy_cache/
.dmypy.json
dmypy.json
```

**预期收益:**
- CI/CD 中自动运行测试
- 类型检查提前发现错误
- 代码格式化统一
- 新人更容易理解代码结构

---

## 4. 实施优先级与路线图

| 阶段 | 任务 | 工作量 | 依赖 |
|------|------|--------|------|
| **Phase 1** | 修复显式 Bug (test_confusion_matrix) | 1h | - |
| **Phase 1** | 消除通配符导入，添加类型提示 | 4h | - |
| **Phase 1** | 创建测试基础设施和首批测试 | 8h | Phase 1 |
| **Phase 2** | 实现抽象基类与接口分离 | 12h | Phase 1 |
| **Phase 2** | 实现并行化交叉验证 | 6h | - |
| **Phase 3** | 重构为包化结构 | 4h | Phase 2 |
| **Phase 3** | CI/CD 集成 | 4h | Phase 3 |

---

## 5. 总结

本项目具有清晰的配置驱动架构，但在工程实践方面存在显著的技术债。三个高优先级优化建议分别针对**可扩展性**（基类抽象）、**性能**（并行化）和**质量**（测试与类型安全）。建议按 Phase 1 → Phase 2 → Phase 3 的顺序逐步实施，每个阶段完成后都有明确的交付价值。

**立即行动项:**
1. 修复 `dummy_model.py:142` 的 `test_confusion_matrix` Bug
2. 更新 `.gitignore` 排除 `.idea/` 目录
3. 移除通配符导入，明确导出接口
