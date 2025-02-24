import yaml
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.path_utils import get_project_root, get_config_path, get_data_path
from datetime import datetime
import dagshub
import mlflow
import pickle
from contextlib import contextmanager
import os
import platform
import psutil
import json
import subprocess
import sys
import pkg_resources
from itertools import product


class ModelTrainer:
    def __init__(self, config_file, general_config):
        """Initialize ModelTrainer with configuration."""
        self.logger = self._setup_logging()
        self.config = self._load_general_config(config_file) if config_file else None
        self.general_config = (
            self._load_config(general_config) if general_config else None
        )

        # Initialize model registry
        self.model_registry = {
            "logistic_regression": LogisticRegression,
            "decision_tree": DecisionTreeClassifier,
            "random_forest": RandomForestClassifier,
        }

        # Initialize data containers
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Results tracking
        self.best_models = {}
        self.best_params = {}
        self.experiments = {}

        # Initialize MLflow tracking
        self.current_experiment_name = None

        # Initialize MLflow tracking
        dagshub.init(
            repo_owner=self.general_config["dagshub_repo_owner"],
            repo_name=self.general_config["dagshub_uri"],
            mlflow=True,
        )

    def get_enabled_models(self):
        enabled_models = []
        
        # Get train_model section from config
        train_model_config = self.config.get('train_model', {})
        
        # Filter out non-model configuration keys
        skip_keys = ['input_test_file', 'input_train_file', 'target_column']
        
        # Iterate through model configurations
        for model_type, model_config in train_model_config.items():
            # Skip non-model configuration entries
            if model_type in skip_keys:
                continue
                
            # Check if model config is a dictionary and is enabled
            if isinstance(model_config, dict) and model_config.get('enabled', False):
                enabled_models.append(model_type)
                
        return enabled_models

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
        )
        return logging.getLogger(__name__)

    def _load_config(self, config_file):
        """Load configuration from a YAML file"""
        try:
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)
            self.logger.info(f"Successfully loaded config from {config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}")
            raise

    def _load_general_config(self, config_file):
        """Load general configuration from a YAML file"""
        try:
            with open(config_file, "r") as file:
                general = yaml.safe_load(file)
            self.logger.info(f"Successfully loaded general_config from {config_file}")
            return general
        except Exception as e:
            self.logger.error(f"Error loading general_config file: {e}")
            raise

    def load_data(self):
        """Load train and test data from CSV files"""
        self.train_data = pd.read_csv(
            f"{get_data_path()}\\{self.config['train_model']['input_train_file']}"
        )
        self.logger.info(
            f"Train data loaded successfully from {get_data_path()}\\{self.config['train_model']['input_train_file']}"
        )
        self.test_data = pd.read_csv(
            f"{get_data_path()}\\{self.config['train_model']['input_test_file']}"
        )
        self.logger.info(
            f"Test data loaded successfully from {get_data_path()}\\{self.config['train_model']['input_test_file']}"
        )
        return self

    def prepare_data(self):
        """Creates X_train, y_train, X_test, y_test dataframes"""
        self.X_train = self.train_data.drop(
            f"{self.config['train_model']['target_column']}", axis=1
        )
        self.y_train = self.train_data[f"{self.config['train_model']['target_column']}"]
        self.X_test = self.test_data.drop(
            f"{self.config['train_model']['target_column']}", axis=1
        )
        self.y_test = self.test_data[f"{self.config['train_model']['target_column']}"]
        self.logger.info("Data preparation completed successfully")
        return self

    def _setup_experiment(self, model_type):
        """Set up MLflow experiment for a specific model type"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config["train_model"][model_type]["model_name"]
        self.experiment_name = f"{model_name}_tuning_{timestamp}"

        mlflow.set_experiment(self.experiment_name)
        self.logger.info(f"Created experiment: {self.experiment_name}")

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            raise ValueError(f"Failed to create experiment: {self.experiment_name}")

        self.experiments[model_type] = experiment
        return experiment

    def _get_param_combinations(self, param_grid):
        """Generate all possible parameter combinations"""
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in product(*values):
            yield dict(zip(keys, instance))

    def _validate_param_grid(self, model_type, param_grid):
        """Validate parameter combinations based on model type"""
        if model_type == "logistic_regression":
            valid_combinations = []
            for params in self._get_param_combinations(param_grid):
                if params["penalty"] == "l1" and params["solver"] not in [
                    "liblinear",
                    "saga",
                ]:
                    continue
                if params["penalty"] == "elasticnet" and params["solver"] != "saga":
                    continue
                # Set l1_ratio to None if penalty is not elasticnet
                if params["penalty"] != "elasticnet":
                    params["l1_ratio"] = None
                if params["penalty"] == "none" and params["solver"] in ["liblinear"]:
                    continue
                valid_combinations.append(params)
            return valid_combinations
        else:
            return list(self._get_param_combinations(param_grid))

    def _save_system_info(self, experiment_id):
        """Save system information to MLflow"""
        system_info = {
            "OS": platform.system(),
            "OS Version": platform.version(),
            "OS Release": platform.release(),
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "CPU Cores": psutil.cpu_count(logical=False),
            "Logical CPUs": psutil.cpu_count(logical=True),
            "RAM (GB)": round(psutil.virtual_memory().total / (1024**3), 2),
            "Disk Space (GB)": round(psutil.disk_usage("/").total / (1024**3), 2),
            "Python Version": sys.version,
            "Environment Variables": {
                key: os.environ[key]
                for key in os.environ
                if key.startswith("PYTHON") or key.startswith("CUDA")
            },
        }

        try:
            gpu_info = subprocess.check_output(
                "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
                shell=True,
            )
            system_info["GPU"] = gpu_info.decode("utf-8").strip()
        except Exception:
            system_info["GPU"] = "No GPU detected"

        tmp_path = f"{get_project_root()}\\tmp"
        os.makedirs(tmp_path, exist_ok=True)

        with open(f"{tmp_path}\\machine_info.json", "w") as f:
            json.dump(system_info, f, indent=4)

        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        with open(f"{tmp_path}\\installed_packages.json", "w") as f:
            json.dump(installed_packages, f, indent=4)

        mlflow.log_artifact(f"{tmp_path}\\machine_info.json", "system_info")
        mlflow.log_artifact(f"{tmp_path}\\installed_packages.json", "system_info")
        self.logger.info("System information saved to MLflow")

    def train_model(self, model_type):
        """Train a specific model type with hyperparameter tuning"""
        if not self.config["train_model"][model_type]["enabled"]:
            self.logger.info(f"{model_type} is disabled in configuration. Skipping.")
            return self

        self.logger.info(f"\n{'='*50}\nStarting {model_type} training\n{'='*50}")

        # Setup experiment
        experiment = self._setup_experiment(model_type)

        # Get model configuration
        model_config = self.config["train_model"][model_type]
        model_class = self.model_registry[model_type]
        param_grid = model_config["param_grid"]

        # Validate parameters
        valid_param_combinations = self._validate_param_grid(model_type, param_grid)
        if not valid_param_combinations:
            self.logger.error(f"No valid parameter combinations found for {model_type}")
            return self

        self.logger.info(
            f"Starting hyperparameter tuning with {len(valid_param_combinations)} combinations"
        )

        best_score = 0
        best_params = None
        best_model = None

        # Try each parameter combination
        for idx, params in enumerate(valid_param_combinations):
            run_name = f"{self.experiment_name}_run_{idx+1:03d}"

            with mlflow.start_run(
                run_name=run_name, experiment_id=experiment.experiment_id
            ) as run:
                self.logger.info(f"Evaluating parameters: {params}")

                # Train and evaluate model
                current_model = model_class(**params)
                current_model.fit(self.X_train, self.y_train)
                y_pred = current_model.predict(self.X_test)

                # Calculate metrics
                metrics = {
                    "accuracy": accuracy_score(self.y_test, y_pred),
                    "precision": precision_score(self.y_test, y_pred, zero_division=0),
                    "recall": recall_score(self.y_test, y_pred, zero_division=0),
                    "f1_score": f1_score(self.y_test, y_pred),
                }

                # Log parameters and metrics
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Save system information to MLflow
                self._save_system_info(experiment.experiment_id)

                self.logger.info(
                    f"Run {run_name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}"
                )

                # Log artifacts
                for artifact in model_config["config_artifacts"]:
                    mlflow.log_artifact(f"{get_config_path()}\\{artifact}", "config")

                # Update best model if better
                if metrics["accuracy"] > best_score:
                    best_score = metrics["accuracy"]
                    best_params = params
                    best_model = current_model

        # Store best model and parameters
        if best_model:
            self.best_models[model_type] = best_model
            self.best_params[model_type] = best_params
            self._save_best_model(model_type)
            self._log_final_evaluation(model_type)

        return self

    def _save_best_model(self, model_type):
        """Save the best model for a specific model type"""
        if model_type not in self.best_models:
            return

        model_path = f"{get_project_root()}\\models\\{model_type}_{self.config['train_model'][model_type]['model_artifact']}"
        with open(model_path, "wb") as file:
            pickle.dump(self.best_models[model_type], file)
        self.logger.info(f"Best {model_type} model saved to {model_path}")

    def _log_final_evaluation(self, model_type):
        """Log final evaluation for best model of a specific type"""
        if model_type not in self.best_models or model_type not in self.experiments:
            return

        with mlflow.start_run(
            run_name=f"{self.experiment_name}_final_evaluation",
            experiment_id=self.experiments[model_type].experiment_id,
        ) as run:
            mlflow.set_tag("best_model", "True")

            # Log parameters
            for param_name, param_value in self.best_params[model_type].items():
                mlflow.log_param(param_name, param_value)

            # Evaluate and log metrics
            y_pred = self.best_models[model_type].predict(self.X_test)
            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred, zero_division=0),
                "recall": recall_score(self.y_test, y_pred, zero_division=0),
                "f1_score": f1_score(self.y_test, y_pred),
            }

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                self.logger.info(
                    f"{model_type} final {metric_name}: {metric_value:.4f}"
                )

            # Create model signature
            signature = mlflow.models.infer_signature(
                model_input=self.X_train,
                model_output=self.best_models[model_type].predict(self.X_train[:1]),
            )

            # Create input example (using first row of training data)
            input_example = self.X_train.iloc[[0]].copy()

            # Log model and system info
            mlflow.sklearn.log_model(
                sk_model=self.best_models[model_type],
                artifact_path="final_model",
                signature=signature,
                input_example=input_example,
            )
            self._save_system_info(self.experiments[model_type].experiment_id)

            # Log config artifacts
            model_config = self.config["train_model"][model_type]
            for artifact in model_config["config_artifacts"]:
                mlflow.log_artifact(f"{get_config_path()}\\{artifact}", "config")


if __name__ == "__main__":
    trainer = ModelTrainer(
        config_file=f"{get_config_path()}\\train_model.yaml",
        general_config=f"{get_config_path()}\\general.yaml",
    )

    try:
        trainer.logger.info(
            f"*** Starting train_model pipeline at {datetime.now()} ***"
        )

        # Load and prepare data
        trainer.load_data().prepare_data()

        # Train each enabled model
        available_models = trainer.get_enabled_models()
        for model_type in available_models:
            trainer.logger.info(f"Training enabled model: {model_type}")   
            trainer.train_model(model_type)

        # Log final results
        trainer.logger.info(f"\n{'='*50}\nTraining pipeline completed\n{'='*50}")
        for model_type, params in trainer.best_params.items():
            trainer.logger.info(f"\n{model_type} best parameters:")
            trainer.logger.info(f"{params}")

        trainer.logger.info(
            f"\n*** Finishing train_model pipeline at {datetime.now()} ***"
        )

    except Exception as e:
        trainer.logger.error(f"Error in model training pipeline: {str(e)}")
