import pandas as pd
import numpy as np
import re
import logging
import yaml
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from utils.path_utils import get_project_root, get_config_path, get_data_path


class DataCleaner:
    def __init__(self, config_file=None):
        """Initialiser le DataCleaner"""
        self.logger = self._setup_logging()
        self.df = None
        self.config = self._load_config(config_file) if config_file else None
        self.dummies = None

    def _setup_logging(self):
        """Configurer la journalisation"""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    def _load_config(self, config_file):
        """Load configuration from a YAML file"""
        try:
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)
            self.logger.info(
                f"*** Starting preprocessing pipeline at {datetime.now()} ***"
            )
            self.logger.info(f"Successfully loaded config from {config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config file: {str(e)}")
            raise

    def load_data(self, file_path=None):
        """Charger les données à partir de différents formats"""
        if not file_path and self.config:
            file_path = (
                f"{get_data_path()}\\{self.config['preprocessing']['input_file']}"
            )
        if not file_path:
            raise ValueError("No input file specified")

        try:
            if file_path.endswith(".csv"):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx"):
                self.df = pd.read_excel(file_path)
            elif file_path.endswith(".json"):
                self.df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")

            self.logger.info(f"Successfully loaded data from {file_path}")
            return self
        except Exception as e:
            self.logger.error(f"Error loading file: {str(e)}")
            raise

    def remove_duplicates(self, subset=None):
        """Supprimer les lignes en double"""
        if self.df is None:
            raise ValueError("No data loaded")

        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset)
        removed_rows = initial_rows - len(self.df)

        self.logger.info(f"Removed {removed_rows} duplicate rows")
        return self

    def remove_columns(self, columns):
        """Supprimer les colonnes spécifiées"""
        if self.df is None:
            raise ValueError("No data loaded")

        self.df = self.df.drop(columns=columns)
        self.logger.info(f"Removed columns: {columns}")
        return self

    def handle_missing_values(self, strategy="drop", fill_value=None):
        """Handle missing values using various strategies."""
        if self.df is None:
            raise ValueError("No data loaded")

        if strategy == "drop":
            self.df = self.df.dropna()
        elif strategy == "fill":
            self.df = self.df.fillna(fill_value)
        elif strategy == "mean":
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_columns] = self.df[numeric_columns].fillna(
                self.df[numeric_columns].mean()
            )

        self.logger.info(f"Handled missing values using strategy: {strategy}")
        return self

    def handle_outliers(self, columns, method="iqr", threshold=1.5):
        """Handle outliers using various strategies."""
        if self.df is None:
            raise ValueError("No data loaded")

        if method == "iqr":
            for col in columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.df = self.df[
                    (self.df[col] > lower_bound) & (self.df[col] < upper_bound)
                ]
        elif method == "std_dev":
            for col in columns:
                upper_bound = self.df[col].mean() + threshold * self.df[col].std()
                lower_bound = self.df[col].mean() - threshold * self.df[col].std()
                self.df = self.df[
                    (self.df[col] > lower_bound) & (self.df[col] < upper_bound)
                ]
        else:
            raise ValueError("Unsupported method")

        self.logger.info(f"Handled outliers using method: {method}")
        return self

    def remove_correlated_features(self, threshold=0.85):
        """Remove highly correlated features. This applies only to numeric features"""
        if self.df is None:
            raise ValueError("No data loaded")

        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        self.df = self.df.drop(columns=to_drop)

        self.logger.info(
            f"Removed correlated features with threshold: {threshold}: {to_drop}"
        )
        return self

    def standardize_column_names(self):
        """Standardize column names to snake_case."""
        if self.df is None:
            raise ValueError("No data loaded")

        new_columns = []
        for col in self.df.columns:
            # Remplace blanks ou characters speciaux avec underscore
            column_name = re.sub(r"[^\w\s]", "", col)
            # Remplacer camel case avec snake case
            column_name = re.sub(r"([a-z])([A-Z])", r"\1_\2", column_name)
            column_name = column_name.replace(" ", "_")
            # Convertir a lowercase
            column_name = column_name.lower()
            new_columns.append(column_name)
        self.logger.info(
            f"Renamed columns from : {self.df.columns.tolist()} to : {new_columns}"
        )
        self.df.columns = new_columns
        return self

    def remove_special_characters(self, columns):
        """Remove special characters from specified columns."""
        if self.df is None:
            raise ValueError("No data loaded")

        for col in columns:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col].astype(str).apply(lambda x: re.sub(r"[^\w\s]", "", x))
                )

        self.logger.info(f"Removed special characters from columns: {columns}")
        return self

    def convert_dates(self, date_columns, format="%Y-%m-%d"):
        """Convert date strings to datetime objects."""
        if self.df is None:
            raise ValueError("No data loaded")

        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(
                    self.df[col], format=format, errors="coerce"
                )

        self.logger.info(f"Converted date columns: {date_columns}")
        return self

    def save_data(self, output_file=None):
        """Save the cleaned data to a file."""
        if self.df is None:
            raise ValueError("No data loaded")

        if not output_file and self.config:
            output_file = (
                f"{get_data_path()}\\{self.config['preprocessing']['output_file']}"
            )
        if not output_file:
            raise ValueError("No output file specified")

        try:
            if output_file.endswith(".csv"):
                self.df.to_csv(output_file, index=False)
            elif output_file.endswith(".xlsx"):
                self.df.to_excel(output_file, index=False)
            elif output_file.endswith(".json"):
                self.df.to_json(output_file)
            else:
                raise ValueError("Unsupported output format")

            self.logger.info(f"Successfully saved cleaned data to {output_file}")
            self.logger.info(
                f"*** Finishing preprocessing pipeline at {datetime.now()} ***"
            )
        except Exception as e:
            self.logger.error(f"Error saving file: {str(e)}")
            raise

    def encode_categorical_features(self, columns=None, drop_first=True):
        """One-hot encode categorical features."""
        if self.df is None:
            raise ValueError("No data loaded")
        if columns is None:
            columns = self.df.select_dtypes(include=["object", "category"]).columns

        # Encode specified columns
        for column in columns:
            # Create dummy variables
            dummies = pd.get_dummies(
                self.df[column], prefix=column, drop_first=drop_first
            ).astype(int)

        # Remove original categorical column
        self.df = self.df.drop(column, axis=1)
        # Add dummy columns to self.dummies
        self.dummies = dummies.columns

        # Add dummy columns to the DataFrame
        self.df = pd.concat([self.df, dummies], axis=1)
        self.logger.info(f"One-hot encoded columns: {columns}")
        return self

    def set_target_column(self, target_column):
        """Set the target column for machine learning."""
        if self.df is None:
            raise ValueError("No data loaded")

        if target_column not in self.df.columns:
            raise ValueError("Target column not found in DataFrame")

        self.df = self.df.rename(columns={target_column: "target"})
        self.logger.info(f"Set target column: {target_column} (renamed to 'target')")
        return self

    def prepare_dataset_for_famd(self, numeric_scaler="standard"):
        """
        Prepare a dataset for Factor Analysis of Mixed Data (FAMD)

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe to be prepared
        numeric_columns : list
            List of column names containing numeric variables
        numeric_scaler : str, optional (default='standard')
            Scaling method for numeric columns
            Options: 'standard', 'robust', 'minmax'
        Returns:
        --------
        pandas.DataFrame
            Prepared dataframe with transformed columns
        """
        # Handle numeric columns scaling
        numeric_columns = [
            col
            for col in self.df.columns
            if col not in self.dummies and col != "target"
        ]
        if numeric_columns:
            # Select scaling method
            if numeric_scaler.lower() == "robust":
                scaler = RobustScaler()
            elif numeric_scaler.lower() == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            # Apply scaling to numeric columns
            self.logger.info(f"Applying numeric scaler on columns : {numeric_columns}")
            self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])

        # Handle categorical columns
        if self.dummies is not None:
            for col in self.dummies:
                # Calculate the square root of (column sum / number of rows)
                col_sum = self.df[col].sum()
                normalizer = np.sqrt(col_sum / len(self.df))
                # Transform categorical column
                self.df[col] = self.df[col] / normalizer
            self.logger.info(f"Applying categorical scaler on columns : {self.dummies}")
        return self

    def move_target_column(self):
        # Move the target column to the end
        target = self.df.pop("target")
        self.df["target"] = target
        return self


if __name__ == "__main__":
    cleaner = DataCleaner(config_file=f"{get_config_path()}/preprocessing.yaml")
    try:
        (
            cleaner.load_data()
            .standardize_column_names()
            .remove_columns(cleaner.config["preprocessing"]["drop_columns"])
            .set_target_column(cleaner.config["preprocessing"]["target_column"])
            .remove_duplicates()
            .handle_missing_values(
                strategy=cleaner.config["preprocessing"]["missing_values_strategy"]
            )
            .remove_correlated_features(
                threshold=cleaner.config["preprocessing"]["correlation_threshold"]
            )
            .encode_categorical_features(
                drop_first=cleaner.config["preprocessing"][
                    "encode_categorical_drop_first"
                ]
            )
            .prepare_dataset_for_famd()
            .move_target_column()
            .save_data()
        )
    except Exception as e:
        cleaner.logger.error(f"Error in data cleaning pipeline: {str(e)}")
