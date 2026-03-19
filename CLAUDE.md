# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch Lightning-based deep learning framework designed for computational pathology (cpath) experiments. It uses Hydra for configuration management and supports MLFlow/WandB for experiment tracking.

## Common Commands

### Running Training
```bash
# Basic training with default config
python main.py

# Specify experiment and run name
python main.py experiment_name=my_exp run_name=my_run

# Override specific hyperparameters via command line
python main.py model.optimizer.lr=0.001 model.optimizer.weight_decay=0.0001
```

### Cross-Validation
```bash
# Run K-fold cross-validation
python main.py mode=cv

# Cross-validation with custom fold count
python main.py mode=cv mode.n_folds=10
```

### Hyperparameter Optimization (HPO)
```bash
# Run Optuna-based hyperparameter search
python main.py hpo=optuna

# HPO with cross-validation mode
python main.py mode=cv hpo=optuna
```

### Using Scripts
```bash
# Run hyperparameter optimization script
bash scripts/hpo.sh

# Run cross-validation with fixed hyperparameters (parallel execution)
bash scripts/run_cv.sh
```

## Architecture

### Configuration System (Hydra)
All configurations are stored in `conf/` and follow a modular structure:
- `conf/config.yaml` - Main entry point that uses Hydra defaults composition
- `conf/datamodule/{domain}/{name}.yaml` - Data loading configurations
- `conf/model/{domain}/{name}.yaml` - Model, optimizer, and scheduler configurations
- `conf/trainer/default.yaml` - PyTorch Lightning trainer settings
- `conf/logger/{mlflow,wandb}.yaml` - Experiment tracking configurations
- `conf/hpo/optuna.yaml` - Hyperparameter search space definitions

**Key Principle**: Configuration is completely decoupled from code. You can change models, dataloaders, optimizers, lr_schedulers, and search ranges by modifying only config files.

### Core Components (`src/core/`)

The framework uses a builder pattern via `src/core/build.py`:
- `build_model()` - Instantiates models via `hydra.utils.instantiate()`
- `build_datamodule()` - Creates LightningDataModule instances (accepts optional `fold` parameter for CV)
- `build_logger()` - Creates MLFlow or WandB loggers
- `build_callbacks()` - Instantiates callbacks from config
- `build_trainer()` - Creates PyTorch Lightning Trainer

### Training Loop (`src/core/train.py`)
Standard training flow: build components → log hyperparameters → `trainer.fit()` → optional `trainer.test()`

### Cross-Validation Loop (`src/core/cv.py`)
K-fold cross-validation that:
- Iterates through each fold
- Adjusts checkpoint paths per fold
- Appends fold number to logger run names
- Returns mean validation score

### Model Template (`src/models/dummy_model.py`)
Models inherit from `L.LightningModule`. New models only need to modify `__init__()` and `forward()`. The template provides:
- Built-in metrics (accuracy, precision, recall, F1, AUC) via torchmetrics
- Automatic confusion matrix logging for best validation epochs
- Optimizer and scheduler configuration from config

### DataModule Template (`src/datamodules/dummy_dm.py`)
DataModules inherit from `L.LightningDataModule`. The dummy module shows the expected structure for WSI (Whole Slide Image) data where each sample is a "bag" of image tiles.

## Environment Variables

- `MLFLOW_TRACKING_URI` - MLFlow server URI (required for MLFlow logging)
- `.env` file - Loaded via python-dotenv if available

## Output Structure

Each run creates:
- `config.yaml` - Full resolved configuration snapshot
- `code/` - Code snapshot for reproducibility
- Checkpoints and artifacts based on callbacks configuration
- `confusion_matrices/` - Saved confusion matrices (best val epoch, test set)

## Cross-Validation Strategy

The framework supports two CV approaches (documented in README.md):

1. **Flatten CV**: Fix one fold for hyperparameter search, then train K models with best params on all K folds
2. **Nested CV**: K outer folds for evaluation, K inner folds for hyperparameter tuning (higher compute cost)
