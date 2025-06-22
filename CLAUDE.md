# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RecBole experiment project for recommendation systems, specifically focused on item click prediction models. The project uses the RecBole framework to implement and compare various deep learning models for click-through rate (CTR) prediction and recommendation ranking.

## Key Commands

**Linting and Formatting:**
```bash
make lint          # Run ruff linter with auto-fix
make fmt           # Format code with ruff
make lint_fmt      # Run both lint and format
make mypy          # Run type checking with mypy
```

**Running the Experiment:**
```bash
# Default: ranking metrics comparison
python -m src.main

# Single model experiment
python -m src.main single_model --model DeepFM --eval-mode labeled

# Custom metrics comparison
python -m src.main custom_metrics --models LR --models FM --models DeepFM --metrics AUC --metrics LogLoss --eval-mode labeled

# Comprehensive comparison (all 32+ models)
python -m src.main comprehensive_comparison --metrics AUC --metrics LogLoss --metrics MAE --eval-mode labeled

# Quick comparison of main models
python -m src.main quick_comparison --eval-mode labeled

# Value metrics comparison (CTR prediction focus)
python -m src.main value_metrics

# Ranking metrics comparison (recommendation focus)  
python -m src.main ranking_metrics

# Click prediction example
python -m src.main predict_click
```

## CLI Interface

**Available Jobs:**
- `single_model`: Run single model experiment
- `quick_comparison`: Run quick comparison of main models
- `comprehensive_comparison`: Run comprehensive comparison of all models
- `value_metrics`: Run value metrics comparison
- `ranking_metrics`: Run ranking metrics comparison (default)
- `custom_metrics`: Run custom metrics comparison
- `predict_click`: Run click prediction example

**Command-Line Options:**
- `--job`: Select which job to run (required, with choices)
- `--model`: Specify model for single model experiment (default: "DeepFM")
- `--eval-mode`: Choose evaluation mode: "labeled" or "full" (default: "labeled")
- `--metrics`: Specify metrics (can be used multiple times)
- `--models`: Specify models for custom comparison (can be used multiple times)

## Architecture

**Core Components:**
- `-m src.main`: Well-architected experiment framework with clean class separation and CLI interface

**Class Architecture:**
- `DataGenerator`: Handles sample data creation and RecBole format conversion
- `Dataset`: Data container class for pre-generated datasets
- `MetricsManager`: Evaluation metrics management with ranking/value metric classification
- `ConfigManager`: Centralized configuration management for RecBole settings
- `ModelRegistry`: Model catalog with configurations and descriptions for 32+ models
- `ModelTrainer`: Training, evaluation, and comparison logic with error handling
- `ClickPredictionExperiment`: High-level experiment orchestration that accepts pre-generated datasets

**Data Pipeline:**
The project follows RecBole's atomic file format for data:
- `.inter` files: User-item interactions with `user_id:token`, `item_id:token`, `label:float`, `timestamp:float`
- `.user` files: User features with `user_id:token`, `age:float`, `gender:token`
- `.item` files: Item features with `item_id:token`, `price:float`, `rating:float`, `category:token`

**Dependency Injection Architecture:**
- Data generation is separated from experiment execution
- `ClickPredictionExperiment` accepts a `Dataset` object instead of creating data internally
- Promotes reusability and cleaner separation of concerns
- Pre-generated datasets can be reused across multiple experiments

**RecBole Integration:**
- Comprehensive recommendation model comparison framework with 32+ models across 3 major categories
- **Context-aware models** (18): LR, FM, FFM, FNN, DeepFM, NFM, AFM, PNN, WideDeep, DCN, DCNV2, xDeepFM, AutoInt, FwFM, FiGNN, DIN, DIEN, DSSM
- **General recommender models** (7): Pop, ItemKNN, BPR, NeuMF, LightGCN, NGCF, DGCF
- **Sequential recommender models** (5): GRU4Rec, SASRec, BERT4Rec, Caser, NARM
- **Flexible metrics system**: Choose between Value metrics (AUC, LogLoss, MAE, RMSE) and Ranking metrics (Recall, MRR, NDCG, Hit, Precision)
- **Automatic metric selection**: System intelligently chooses appropriate metrics based on model type and evaluation mode
- Multiple comparison modes with customizable metrics and evaluation settings
- Each model has optimized hyperparameters and category-specific configurations
- Implements binary classification for click prediction (0: PV, 1: Click)
- Data split: 80% train, 10% validation, 10% test
- Dynamic result display adapts to selected metrics with intelligent sorting
- Covers various recommendation paradigms: CTR prediction, collaborative filtering, sequence modeling

**Evaluation Options:**
- `run_value_metrics_comparison()`: CTR prediction focus with AUC, LogLoss, etc.
- `run_ranking_metrics_comparison()`: Recommendation focus with Recall, NDCG, etc.
- `run_custom_metrics_comparison()`: User-defined metrics and model selection
- Automatic eval mode adjustment (labeled vs full) based on metric type

**Output Structure:**
- `dataset/`: Generated RecBole-compatible data files
- `log/`: Training logs organized by model name
- `log_tensorboard/`: TensorBoard logs for training visualization
- `saved/`: Model checkpoints and saved states

## Development Environment

- Python 3.13.2+ required
- Uses uv for dependency management
- Key dependencies: recbole>=1.2.0, numpy<2, pandas>=2.3.0, click>=8.2.1
- Dev tools: ruff, mypy, isort configured via pyproject.toml

## Data Format Requirements

When working with RecBole data, ensure feature names follow `feat_name:feat_type` format:
- `token`: Discrete features (IDs, categories)
- `float`: Continuous features (ratings, timestamps, numerical attributes)
- Label field must be `float` type for binary classification tasks

## Known Fixes and Compatibility

**PyTorch 2.6 Compatibility:**
- The code includes a monkey patch for `torch.load` to handle the `weights_only` parameter
- Ensures compatibility with newer PyTorch versions

**Error Handling:**
- Comprehensive error handling for model compatibility issues
- Automatic metric selection prevents metric/model type mismatches
- Robust data format adaptation for different evaluation scenarios