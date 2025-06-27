# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RecBole experiment project for recommendation systems, specifically focused on item click prediction models. The project uses the RecBole framework to implement and compare various deep learning models for click-through rate (CTR) prediction and recommendation ranking.

## Key Commands

**Testing and Code Quality:**
```bash
# Testing
python -m pytest test/ -v    # Run all tests with verbose output
python -m pytest test/test_specific_file.py  # Run specific test file

# Linting and Formatting
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

**Project Structure:**
```
src/
├── main.py                      # CLI entry point and job orchestration
└── recbole_experiment/
    ├── config/
    │   └── manager.py          # ConfigManager: RecBole configuration management
    ├── data/
    │   ├── dataset.py          # Dataset: Data container class
    │   └── generator.py        # DataGenerator: Sample data creation
    ├── experiments/
    │   └── click_prediction.py # ClickPredictionExperiment: Main experiment class
    ├── models/
    │   └── registry.py         # ModelRegistry: Model catalog and configurations
    ├── training/
    │   ├── metrics.py          # MetricsManager: Evaluation metrics management
    │   └── trainer.py          # ModelTrainer: Training and comparison logic
    └── utils/
        └── torch_compat.py     # PyTorch compatibility utilities
```

**Core Components:**
- `src.main`: CLI entry point with well-architected job orchestration and clean class separation

**Class Architecture:**
- `ConfigManager` (`config/manager.py`): Centralized configuration management for RecBole settings
- `DataGenerator` (`data/generator.py`): Handles sample data creation and RecBole format conversion
- `Dataset` (`data/dataset.py`): Data container class for pre-generated datasets
- `ClickPredictionExperiment` (`experiments/click_prediction.py`): High-level experiment orchestration
- `ModelRegistry` (`models/registry.py`): Model catalog with configurations and descriptions for 33+ models
- `MetricsManager` (`training/metrics.py`): Evaluation metrics management with ranking/value metric classification
- `ModelTrainer` (`training/trainer.py`): Training, evaluation, and comparison logic with error handling
- `torch_compat` (`utils/torch_compat.py`): PyTorch 2.6+ compatibility patches

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
- Comprehensive recommendation model comparison framework with 33+ models across 3 major categories
- **Context-aware models** (19): LR, FM, FFM, FNN, DeepFM, NFM, AFM, PNN, WideDeep, DCN, DCNV2, xDeepFM, AutoInt, FwFM, FiGNN, DIN, DIEN, DSSM, LightGBM
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
- Testing: pytest>=8.4.1 with pytest-mock>=3.14.1 for mocking

## Testing Framework

**Test Structure:**
- All tests located in `test/` directory
- Uses pytest framework with pytest-mock for mocking
- 128+ comprehensive test cases covering all major components
- Test files follow `test_*.py` naming convention

**Test Coverage:**
- `test_click_prediction_experiment.py`: Tests for `experiments/click_prediction.py`
- `test_config_manager.py`: Tests for `config/manager.py`
- `test_data_generator.py`: Tests for `data/generator.py`
- `test_dataset.py`: Tests for `data/dataset.py`
- `test_metrics_manager.py`: Tests for `training/metrics.py`
- `test_model_registry.py`: Tests for `models/registry.py`
- `test_model_trainer.py`: Tests for `training/trainer.py`
- `test_torch_patch.py`: Tests for `utils/torch_compat.py`

**Mocking Pattern:**
- Uses pytest-mock's `mocker` fixture instead of unittest.mock
- Example: `mock_obj = mocker.patch("module.function")`
- All test methods requiring mocks accept `mocker` parameter
- Follows pytest best practices for test isolation

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

## Development Guidelines

**Code Quality:**
- All code must pass linting (`make lint_fmt`) before committing
- Type hints are required for public methods and classes
- Follow existing code patterns and architectural decisions
- Write tests for new functionality using pytest-mock patterns

**Testing Requirements:**
- New features must include corresponding test coverage
- Use `mocker` fixture for all mocking needs (not unittest.mock)
- Test method signatures: `def test_method(self, mocker):`
- Maintain test isolation and avoid side effects between tests

**When Contributing:**
- Run full test suite before submitting changes: `python -m pytest test/ -v`
- Ensure linting passes: `make lint_fmt`
- Follow existing naming conventions and project structure
- Update CLAUDE.md if adding new major features or architectural changes