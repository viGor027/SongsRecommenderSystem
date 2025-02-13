import numpy
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from song_pipeline.constants import PROJECT_FOLDER_DIR
import os
import json


def read_json_to_dict(file_path: str):
    """
    Args:
      file_path (str): The path to the JSON file **(including file extension)**.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def write_dict_to_json(data: dict, file_path: str):
    """
    Args:
      data (dict): The dictionary to write to the file.
      file_path (str): The path to the JSON file **(including file extension)**.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
