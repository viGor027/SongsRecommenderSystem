import json
from pathlib import Path


def read_json_to_dict(file_path: Path | str) -> dict:
    """
    Args:
      file_path (str): The path to the JSON file **(including file extension)**.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def write_dict_to_json(data: dict | list, file_path: Path | str):
    """
    Args:
      data (dict): The dictionary to write to the file.
      file_path (str): The path to the JSON file **(including file extension)**.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
