import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime
from sklearn.model_selection import train_test_split
from utils.path_utils import get_project_root, get_config_path, get_data_path


class DataSplitter:
    def __init__(self, config_file=None):
        """Initialize DataSplitter with optional data."""
        self.logger = self._setup_logging()
        self.data = None
        self.train_data = None
        self.test_data = None
        self.config = self._load_config(config_file) if config_file else None

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
            self.logger.info(
                f"*** Starting split_train_test pipeline at {datetime.now()} ***"
            )
            self.logger.info(f"Successfully loaded config from {config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}")
            raise

    def load_data(self, filepath=None, file_type="csv"):
        """Load data from file."""
        if not filepath and self.config:
            filepath = (
                f"{get_data_path()}\\{self.config["split_train_test"]['input_file']}"
            )
        if not filepath:
            raise ValueError("No input file specified")

        try:
            if file_type == "csv":
                self.data = pd.read_csv(filepath)
            elif file_type == "excel":
                self.data = pd.read_excel(filepath)
            elif file_type == "json":
                self.data = pd.read_json(filepath)
            else:
                raise ValueError("Unsupported file type")

            self.logger.info(f"Loaded data from {filepath}")
            return self
        except Exception as e:
            self.logger.error(f"Data loading error: {e}")
            raise

    def split_data(
        self, target_column=None, test_size=0.2, random_state=23, stratify=False
    ):
        """Split data into training and testing sets."""
        if self.data is None:
            raise ValueError("No data loaded")

        try:
            # Determine stratification
            stratify_param = (
                self.data[target_column] if stratify and target_column else None
            )

            # Separate features and target if target column specified
            if target_column:
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify_param,
                )

                # Recombine features and target
                self.train_data = pd.concat([X_train, y_train], axis=1)
                self.test_data = pd.concat([X_test, y_test], axis=1)
            else:
                # Simple random split without target
                train_indices, test_indices = train_test_split(
                    self.data.index, test_size=test_size, random_state=random_state
                )

                self.train_data = self.data.loc[train_indices]
                self.test_data = self.data.loc[test_indices]

            self.logger.info(
                f"Split data: train size = {len(self.train_data)}, test size = {len(self.test_data)}, random state = {random_state}, stratify = {stratify}"
            )
            self.logger.info(
                f"Target distribution in train set: {self.train_data[target_column].value_counts(normalize=True)}"
            )
            self.logger.info(
                f"Target distribution in test set: {self.test_data[target_column].value_counts(normalize=True)}"
            )

            return self
        except Exception as e:
            self.logger.error(f"Data splitting error: {e}")
            raise

    def save_splits(self, train_path=None, test_path=None):
        """Save train and test datasets."""
        if self.train_data is None or self.test_data is None:
            raise ValueError("No split data available")

        if not train_path and self.config:
            train_path = f"{get_data_path()}\\{self.config["split_train_test"]['train_output_file']}"
        if not test_path and self.config:
            test_path = f"{get_data_path()}\\{self.config["split_train_test"]['test_output_file']}"
        if not train_path or not test_path:
            raise ValueError("No output file specified")

        try:
            self.train_data.to_csv(train_path, index=False)
            self.test_data.to_csv(test_path, index=False)
            self.logger.info(f"Saved train data to {train_path}")
            self.logger.info(f"Saved test data to {test_path}")
            self.logger.info(
                f"*** Finishing split_train_test pipeline at {datetime.now()} ***"
            )

        except Exception as e:
            self.logger.error(f"Saving split data error: {e}")
            raise


if __name__ == "__main__":
    splitter = DataSplitter(config_file=f"{get_config_path()}\\split_train_test.yaml")
    (
        splitter.load_data()
        .split_data(
            target_column=splitter.config["split_train_test"]["target_column"],
            test_size=splitter.config["split_train_test"]["test_size"],
            random_state=splitter.config["split_train_test"]["random_state"],
            stratify=splitter.config["split_train_test"]["stratify"],
        )
        .save_splits()
    )
