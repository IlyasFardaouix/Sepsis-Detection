"""Configuration loader utility."""

import yaml
import os
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent

    # Construct full path
    if not os.path.isabs(config_path):
        config_path = project_root / config_path

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
