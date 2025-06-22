# RecBole Experiment Framework

A comprehensive framework for experimenting with recommendation models for click prediction using [RecBole](https://recbole.io/). This repository provides tools for generating synthetic data, training various recommendation models, and evaluating their performance using different metrics.

## What is this repository?

This repository contains a framework built on top of RecBole that simplifies the process of:

1. Generating synthetic click prediction datasets
2. Training and evaluating various recommendation models
3. Comparing model performance using different metrics
4. Running comprehensive experiments with minimal setup

The framework supports various types of recommendation models:
- Context-aware models (e.g., DeepFM, DCN, AutoInt)
- General recommendation models (e.g., BPR, NeuMF, LightGCN)
- Sequential recommendation models (e.g., GRU4Rec, BERT4Rec)

## Requirements

- Python >= 3.13.2
- Dependencies:
  - recbole >= 1.2.0
  - click >= 8.2.1
  - numpy < 2
  - pandas >= 2.3.0
  - torch
  - scikit-learn >= 1.7.0
  - lightgbm >= 4.6.0
  - plotly >= 6.1.2
  - polars >= 1.31.0
  - pyarrow >= 20.0.0
  - pydantic >= 2.11.7
  - ray >= 2.47.1
  - kmeans-pytorch >= 0.3

## How to Use

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/shibuiwilliam/rec_bole_experiment.git
   cd rec-bole-experiment
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

### Running Experiments

The framework provides several CLI commands for running different types of experiments:

#### Single Model Experiment

```bash
python -m src.main run_single_model --model DeepFM --metrics auc precision recall --eval-mode labeled
```

#### Quick Comparison of Models

```bash
python -m src.main run_quick_comparison --metrics auc precision recall --eval-mode labeled
```

#### Comprehensive Comparison

```bash
python -m src.main run_comprehensive_comparison --metrics auc precision recall --eval-mode labeled
```

#### Value Metrics Comparison

```bash
python -m src.main value_metrics
```

#### Ranking Metrics Comparison

```bash
python -m src.main ranking_metrics
```

#### Custom Metrics and Models Comparison

```bash
python -m src.main custom_metrics --models DeepFM DCN AutoInt --metrics auc precision recall --eval-mode labeled
```

#### Click Prediction

```bash
python -m src.main predict_click
```

### Development

For development, you can use the following commands:

```bash
# Format code
make fmt

# Lint code
make lint

# Run type checking
make mypy

# Format and lint in one command
make lint_fmt

# Running tests
make test
```

## Disclaimer

This framework is provided for research and educational purposes only. The synthetic data generated does not represent real user behavior and should not be used for production systems without proper validation.

The performance of models may vary depending on the specific dataset and use case. Always validate models with your own data before deploying them in production environments.

This project is not officially affiliated with RecBole. For official documentation and support for RecBole, please visit [https://recbole.io/](https://recbole.io/).