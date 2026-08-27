"""Configuration loader."""
import yaml
from pathlib import Path
from typing import Any
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


@lru_cache
def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_config() -> dict[str, Any]:
    """Get cached config."""
    return load_config()