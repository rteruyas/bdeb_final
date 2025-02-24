# path_utils.py
from pathlib import Path
import os

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()


def get_project_root() -> Path:
    """Returns project root folder."""
    return ROOT_DIR


def get_config_path() -> Path:
    """Returns the config folder path."""
    return ROOT_DIR / "config"


def get_data_path() -> Path:
    """Returns the data folder path."""
    return ROOT_DIR / "data"


# You can add more specific path getters as needed
def get_logs_path() -> Path:
    """Returns the logs folder path."""
    return ROOT_DIR / "logs"


# Example usage of environment variables for configuration
def get_env_config_path() -> Path:
    """
    Returns config path from environment variable if set,
    otherwise returns default config path.
    """
    env_config = os.getenv("PROJECT_CONFIG_PATH")
    if env_config:
        return Path(env_config)
    return get_config_path()
