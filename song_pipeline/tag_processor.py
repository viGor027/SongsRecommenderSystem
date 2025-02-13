import torch
import torch.nn as nn
from song_pipeline.utils import write_dict_to_json, read_json_to_dict
from song_pipeline.constants import TAGS_DIR, LABELS_DIR, PROJECT_FOLDER_DIR
import os


class TagProcessor:
    @staticmethod
    def _get_all_tags(tags_dir: str):
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

    @staticmethod
    def _tags_to_tags_indexes(y: list[list[str]], all_tags: list[str]) -> list[list[int]]:
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

    @staticmethod
    def _multi_hot_batch(y: list[list[int]], n_classes: int) -> torch.Tensor:
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
        return multi_hotted

    @staticmethod
    def _save_multi_hotted_labels(songs_titles: list[str], songs_labels: torch.Tensor, path: str):
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

    @staticmethod
    def multi_hot_tags_of_all_songs():
        """
        Encodes the tags of all songs in a `TAGS_DIR` directory into multi-hot format,
        saves the resulting data and labels mapping to `LABELS_DIR`.
        """
        all_tags = TagProcessor._get_all_tags(TAGS_DIR)
        song_titles = []
        song_tags_str = []
        for song in os.listdir(TAGS_DIR):
            current_song_tags_str = []
            song_title = song[:-5]
            tag_dct = read_json_to_dict(os.path.join(TAGS_DIR, song))
            for tag in tag_dct["genres"]:
                if tag == 'Dance-Pop':
                    current_song_tags_str.append('Dance Pop')
                elif ', ' in tag:
                    for nested_t in tag.split(', '):
                        current_song_tags_str.append(nested_t)
                else:
                    current_song_tags_str.append(tag)
            for tag in tag_dct["mood"]:
                current_song_tags_str.append(tag)
            song_titles.append(song_title)
            song_tags_str.append(current_song_tags_str)

        song_tags_indexes = TagProcessor._tags_to_tags_indexes(song_tags_str, all_tags)
        multi_hotted = TagProcessor._multi_hot_batch(song_tags_indexes, len(all_tags))
        TagProcessor._save_multi_hotted_labels(song_titles, multi_hotted, os.path.join(LABELS_DIR, 'labels.json'))
        write_dict_to_json(
            data={i: tag for i, tag in enumerate(all_tags)},
            file_path=os.path.join(LABELS_DIR, 'mapping.json')
        )


if __name__ == "__main__":
    # Usage example
    
    tp = TagProcessor()
    example_tags = ['Alternative Dance', 'Alternative Pop', 'Ambient',
                    'Angry', 'Anti-Pop', 'Bass', 'Bass House',
                    'Brazilian Phonk', 'Breakbeat', 'Chasing', 'Chill']

    example_songs = ['song_1', 'song_2', 'song_3']

    batch_of_tags = [
        ['Alternative Dance', 'Angry', 'Bass'],
        ['Alternative Dance', 'Alternative Pop', 'Ambient'],
        ['Alternative Pop', 'Angry', 'Bass', 'Ambient', 'Bass']
    ]

    batch_of_tags_idx = tp._tags_to_tags_indexes(batch_of_tags, example_tags)
    # batch_of_tags_idx[i] contains tags for ith song of a batch,
    # element of batch_of_tags_idx[i][j] is an index of a tag in example_tags list
    print("tags indexes: ")
    print(batch_of_tags_idx)

    multi_hotted = tp._multi_hot_batch(batch_of_tags_idx, len(example_tags))
    print(multi_hotted)
    print(multi_hotted.shape)

    example_path = os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'labels')
    tp._save_multi_hotted_labels(example_songs, multi_hotted, os.path.join(example_path, 'test.json'))

    loaded = read_json_to_dict(os.path.join(example_path, 'test.json'))
    print(loaded)