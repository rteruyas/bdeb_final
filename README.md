# Predictive Maintenance ML Pipeline (AI4I dataset)

![Maintenance](https://img.shields.io/badge/Maintenance-Active-green.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-yellow?style=flat-square)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazonwebservices&logoColor=white)
![docker](https://img.shields.io/badge/docker-257bd6?style=for-the-badge&logo=docker&logoColor=white)


A complete end-to-end machine learning pipeline for predictive maintenance using the AI4I 2020 dataset.

## Overview

This project implements a production-ready ML pipeline for predicting machine failures using the AI4I 2020 Predictive Maintenance dataset. The pipeline handles everything from data preprocessing to model deployment with a focus on reproducibility, scalability, and continuous integration.

### Key Features

- **Data Version Control**: Full dataset and model versioning using DVC
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Cloud Infrastructure**: AWS S3 for storage and model serving
- **Interactive UI**: Streamlit dashboard for model exploration
- **API Service**: FastAPI endpoint for real-time predictions
- **MLOps Best Practices**: Reproducible experiments, model tracking, and monitoring

## Project Structure

```	
├── .github/                     			# GitHub Actions workflows
│   └── workflows/                          
│       ├── register_and_upload_model.yml	# Register best model and upload it to s3 model store
├── config/                      			# Configuration files
│   ├── general.yaml             			# General project configuration
│   ├── preprocessing.yaml       			# Preprocessing parameters
│   ├── split_train_test.yaml    			# Train-test split parameters
│   └── train_model.yaml         			# Model training parameters
├── data/                        			# Data directory (DVC-tracked)
│   ├── raw_data.csv             			# Original AI4I 2020 dataset
├── metrics/                     			# Model metrics (DVC-tracked)
│   ├── model_metrics.json       			# Performance metrics for all models
│   └── training_status.json     			# Training metadata
├── models/                      			# Trained models (DVC-tracked)
├── notebooks/                   			# Jupyter notebooks for exploration
├── src/                         			# Source code
│   ├── dashboard/               			# Streamlit dashboard
│   │   └── app.py               			# Dashboard implementation
│   ├── utils/                   			# Utility functions
│   │   └── path_utils.py        			# Path management
│   ├── preprocessing.py         			# Data preprocessing pipeline
│   ├── split_train_test.py      			# Data splitting logic
│   └── train_model.py           			# Model training and evaluation
├── .dvcignore                   			# Files to ignore in DVC
├── .gitignore                   			# Files to ignore in Git
├── dvc.yaml                     			# DVC pipeline definition
├── dvc.lock                     			# DVC pipeline lock file
├── requirements.txt             			# Python dependencies
└── README.md                    			# This file
```

## Dataset

The [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) represents synthetic data that mimics real predictive maintenance sensor data from industrial equipment. It includes:

- Process measurements (temperature, pressure, etc.)
- Machine settings
- Failure types and indicators
- Machine type/age information

The goal is to predict when machines will fail, enabling proactive maintenance to prevent costly downtime.

## Getting Started

### Prerequisites

- Python 3.9+ (recommended 3.12)
- Git
- AWS CLI configured with appropriate permissions
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/rteruyas/bdeb_final.git 
cd bdeb_final
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

4. Set up DVC with S3:

```bash
dvc remote add -d s3remote s3://your-bucket-name/dvc-storage
```

5. Pull data and models:

```bash
dvc pull
```

### Running the Pipeline

To execute the complete ML pipeline:

```bash
dvc repro
```

This will:
1. Preprocess the raw data
2. Split data into training and test sets
3. Train and evaluate multiple models
4. Track metrics and models into mlflow (provided by dagshub)

To run a specific stage:

```bash
dvc repro <stage_name>  # e.g., dvc repro train
```

## CI/CD Pipeline

This project uses GitHub Actions for CI/CD:

- **Continuous Integration**: Runs tests, linting, and generates metrics on each pull request
- **Continuous Deployment**: Automatically deploys the latest model and services when changes are merged to main

The workflows are defined in `.github/workflows/`.

## AWS Infrastructure

Key AWS components:

- **S3**: Stores DVC files, trained models, and deployment artifacts
- **ECR**: Hosts Docker images for services

## Development

### Adding New Models

1. Update `config/train_model.yaml` with your model's configuration
2. Add model implementation to `src/train_model.py`
3. Run `dvc repro train` to train and evaluate your model

### Experiment Tracking

To compare different experiments:

```bash
dvc metrics show
dvc metrics diff
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) creators
- [DVC](https://dvc.org/) for data version control
- [FastAPI](https://fastapi.tiangolo.com/) for API development
- [Streamlit](https://streamlit.io/) for dashboard creation
