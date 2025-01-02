import torch
import torch.nn as nn
from song_pipeline.constants import PROJECT_FOLDER_DIR
import os
import json


def read_json_to_dict(file_path: str):
    """
    Parameters:
      file_path (str): The path to the JSON file **(including file extension)**.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def write_dict_to_json(data: dict, file_path: str):
    """
    Parameters:
      data (dict): The dictionary to write to the file.
      file_path (str): The path to the JSON file **(including file extension)**.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def get_all_tags(tags_dir: str):
    """returns sorted list of all tags in the dataset
        Note: Method to be updated after using all songs
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


def tags_to_tags_indexes(y: list[list], all_tags: list):
    """
    Note: y need to be a batch of songs tags, even if the batch size is 1.
    """
    for tags_list in y:
        for i, tag in enumerate(tags_list):
            tags_list[i] = all_tags.index(tag)
    return y


def multi_hot_batch(y: list[list], n_classes: int):
    """
    Note: y need to be a batch of samples, even if the batch size is 1.
    """
    multi_hotted = []
    for song_tags_indexes in y:
        one_hot = nn.functional.one_hot(torch.tensor(song_tags_indexes), n_classes)
        multi_hotted.append(torch.sum(one_hot, dim=0))

    multi_hotted = torch.stack(multi_hotted, dim=0)
    return multi_hotted


def save_multi_hotted_labels(songs_titles: list[str], songs_labels: torch.tensor, path: str):
    """
    Params:
        songs_titles: Titles of a songs to save
        songs_labels: Multi-hot encoded tags for songs in songs_titles
        path: The path to the JSON file **(including file extension)**.
    """
    labeled = dict()
    for i, title in enumerate(songs_titles):
        labeled[title] = [int(e) for e in songs_labels[i]]

    write_dict_to_json(labeled, path)


if __name__ == "__main__":
    # Usage example
    example_tags = ['Alternative Dance', 'Alternative Pop', 'Ambient',
                    'Angry', 'Anti-Pop', 'Bass', 'Bass House',
                    'Brazilian Phonk', 'Breakbeat', 'Chasing', 'Chill']

    example_songs = ['song_1', 'song_2', 'song_3']

    batch_of_tags = [
        ['Alternative Dance', 'Angry', 'Bass'],
        ['Alternative Dance', 'Alternative Pop', 'Ambient'],
        ['Alternative Pop', 'Angry', 'Bass', 'Ambient']
    ]

    batch_of_tags_idx = tags_to_tags_indexes(batch_of_tags, example_tags)
    # batch_of_tags_idx[i] contains tags for ith song of a batch,
    # element of batch_of_tags_idx[i][j] is an index of a tag in example_tags list
    print("tags indexes: ")
    print(batch_of_tags_idx)

    multi_hotted = multi_hot_batch(batch_of_tags, len(example_tags))
    print(multi_hotted)
    print(multi_hotted.shape)

    example_path = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'labels')
    save_multi_hotted_labels(example_songs, list(multi_hotted), os.path.join(example_path, 'test.json'))

    loaded = read_json_to_dict(os.path.join(example_path, 'test.json'))
    print(loaded)
