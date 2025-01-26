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


def get_all_tags(tags_dir: str):
    """
    Args:
        tags_dir (str): Path to directory storing JSON files with song tags.

    Returns:
        Sorted list of all tags in the dataset.
    """
    tags = set()
    for song in os.listdir(tags_dir):
        data = read_json_to_dict(os.path.join(tags_dir, song))
        tags_to_add = []
        tags_to_add.extend(data["genres"])
        tags_to_add.extend(data["mood"])
        for t in tags_to_add:
            if ', ' in t:
                for nested_t in t.split(', '):
                    tags.add(nested_t)
            elif t == 'Dance-Pop':
                tags.add('Dance Pop')
            else:
                tags.add(t)
    return sorted(list(tags))


def tags_to_tags_indexes(y: list[list[str]], all_tags: list[str]) -> list[list[int]]:
    """
    Note: y need to be a batch of songs tags, even if the batch size is 1.

    Args:
        y (list[list[str]]): A batch of song tags, where each inner list contains tags (as strings)
            for a single song. Tags are represented as words, such as "Ambient", "Rock", etc.
        all_tags (list): A list containing all possible tags in the dataset.

    Returns:
        A batch of song tags, where each inner list contains the indices of the original tags
        based on their position in `all_tags`
    """
    for tags_list in y:
        for i, tag in enumerate(tags_list):
            tags_list[i] = all_tags.index(tag)
    return y


def multi_hot_batch(y: list[list[int]], n_classes: int) -> torch.Tensor:
    """
    Note: y need to be a batch of samples, even if the batch size is 1.

    Args:
        y (list[list[int]]): A batch of song tags, where each inner list contains tags (as ints)
            for a single song.
        n_classes (int): Number of all possible classes(how many different tags is there in the dataset).
    """
    multi_hotted = []
    for song_tags_indexes in y:
        one_hot = nn.functional.one_hot(torch.tensor(song_tags_indexes), n_classes)
        multi_hotted.append(torch.sum(one_hot, dim=0))

    multi_hotted = (torch.stack(multi_hotted, dim=0) != 0).float()
    #multi_hotted = torch.stack(multi_hotted, dim=0)
    return multi_hotted


def save_multi_hotted_labels(songs_titles: list[str], songs_labels: torch.Tensor, path: str):
    """
    Params:
        songs_titles: Titles of a songs to save
        songs_labels: Multi-hot encoded tags for songs in songs_titles
        path: The path to the JSON file **(including file extension)**.
    """
    labeled = dict()
    for i, title in enumerate(songs_titles):
        labeled[title] = [int(e) for e in songs_labels[i]]

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    write_dict_to_json(labeled, path)


def prepare_for_dataset(data: list[Tuple[str, np.ndarray, list[int]]],
                        shuffle: bool = False, return_titles: bool = False):
    """
   Prepares data for use in a PyTorch Dataset **(including normalizing values to [0,1] range)**.

   Args:
       data (list[Tuple[str, np.ndarray, list[int]]]): A list of tuples containing:
           - Song title (str)
           - Spectrogram as a NumPy array (np.ndarray)
           - Labels as a list of integers (list[int])
       shuffle (bool): Whether to shuffle the data. Defaults to False.
       return_titles (bool): Whether to return titles of songs in dataset.

   Returns:
       Depending on ``return_titles`:
            - If `return_titles=True`:
                Tuple[list[str], torch.Tensor, torch.Tensor]: Titles, Features (X) and labels (Y) as PyTorch tensors.
            - If `return_titles=False`:
                Tuple[torch.Tensor, torch.Tensor]: Features (X) and labels (Y) as PyTorch tensors.
    """
    if shuffle:
        np.random.shuffle(data)
    zipped = list(zip(*data))
    titles, X, Y = zipped[0], zipped[1], zipped[2]
    X = (np.stack(X, dtype=numpy.float16) + 80.) / 80.
    Y = np.stack(Y, dtype=numpy.float16)
    return list(titles), torch.from_numpy(X), torch.from_numpy(Y) if return_titles else (torch.from_numpy(X), torch.from_numpy(Y))


if __name__ == "__main__":
    # Usage example

    example_tags = ['Alternative Dance', 'Alternative Pop', 'Ambient',
                    'Angry', 'Anti-Pop', 'Bass', 'Bass House',
                    'Brazilian Phonk', 'Breakbeat', 'Chasing', 'Chill']

    example_songs = ['song_1', 'song_2', 'song_3']

    batch_of_tags = [
        ['Alternative Dance', 'Angry', 'Bass'],
        ['Alternative Dance', 'Alternative Pop', 'Ambient'],
        ['Alternative Pop', 'Angry', 'Bass', 'Ambient', 'Bass']
    ]

    batch_of_tags_idx = tags_to_tags_indexes(batch_of_tags, example_tags)
    # batch_of_tags_idx[i] contains tags for ith song of a batch,
    # element of batch_of_tags_idx[i][j] is an index of a tag in example_tags list
    print("tags indexes: ")
    print(batch_of_tags_idx)

    multi_hotted = multi_hot_batch(batch_of_tags_idx, len(example_tags))
    print(multi_hotted)
    print(multi_hotted.shape)

    example_path = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'labels')
    save_multi_hotted_labels(example_songs, multi_hotted, os.path.join(example_path, 'test.json'))

    loaded = read_json_to_dict(os.path.join(example_path, 'test.json'))
    print(loaded)
