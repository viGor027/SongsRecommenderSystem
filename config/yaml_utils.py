import yaml
import os


def save_dict_to_yaml(dct: dict, path: str, cfg_file_name: str):
    """
    Saves a dictionary to a YAML file.

    This function takes a dictionary and saves it as a YAML file at the specified
    path with the given file name. The file name should not include the file extension,
    as ".yaml" is automatically appended.

    Args:
        dct (dict): The dictionary to be saved to a YAML file.
        path (str): The directory where the YAML file will be saved.
        cfg_file_name (str): The name of the YAML file (without the file extension).
    """
    f = open(os.path.join(path, cfg_file_name + '.yaml'), 'w')
    yaml.dump(dct, f)
    f.close()


def yaml_to_dict(path: str, file_name: str) -> dict:
    """
    Loads a YAML file and converts its contents to a dictionary.

    This function reads a YAML file from the specified path and returns its contents
    as a Python dictionary. The file name should include the ".yaml" extension.

    Args:
        path (str): The directory path where the YAML file is located.
        file_name (str): The name of the YAML file (including the file extension).

    Returns:
        dict: The contents of the YAML file as a Python dictionary.
    """
    f = open(os.path.join(path, file_name), 'r')
    dct = yaml.load(f, Loader=yaml.SafeLoader)
    f.close()
    return dct
