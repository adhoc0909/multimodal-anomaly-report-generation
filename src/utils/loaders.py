import yaml
import json
import pandas as pd
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML: {e}")
        raise


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get('model', {})

def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get('training', {})

def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get('data', {})

def get_wandb_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get('wandb', {})



def load_json(json_path: str):
    with json_path.open("r", encoding="utf-8") as f:
        json_file = json.load(f)
    return json_file

def load_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    return df