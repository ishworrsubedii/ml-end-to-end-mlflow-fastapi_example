from box.exceptions import BoxValueError
import yaml
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from src.mlflow_implementation import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads and loads YAML content into a ConfigBox object

    :param path_to_yaml: Path to the YAML file
    :return: ConfigBox object with loaded YAML content
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file '{path_to_yaml}' loaded successfully")
            if content is None:
                raise ValueError("YAML file is empty")
            return ConfigBox(content)
    except BoxValueError as box_error:
        raise ValueError(f"BoxValueError: {box_error}")
    except FileNotFoundError as file_not_found_error:
        raise FileNotFoundError(f"File not found: {file_not_found_error}")
    except Exception as e:
        logger.exception(f"An error occurred while reading YAML file '{path_to_yaml}': {str(e)}")
        raise
