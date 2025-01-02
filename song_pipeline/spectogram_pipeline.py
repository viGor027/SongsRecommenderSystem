from song_pipeline.feature_extractor import FeatureExtractor
from song_pipeline.constants import N_MELS, N_SECONDS, SPEC_TYPE, PROJECT_FOLDER_DIR,\
    TAGS_DIR, LABELS_DIR, SONGS_DIR
from song_pipeline.dict_types import ConfigType, SongSpecDataDictType
from song_pipeline.utils import write_dict_to_json, read_json_to_dict, get_all_tags, \
    multi_hot_batch, tags_to_tags_indexes, save_multi_hotted_labels
from typing import Literal, List, Tuple
import numpy as np
import os


class SpectogramPipeline:
    """
    A pipeline for generating spectrograms and metadata from a collection of songs.

    This class processes audio files stored in a specified folder, splitting them into
    fragments, extracting spectrograms (e.g., mel spectrograms or standard spectrograms),
    and associating song metadata such as title and tags. It provides flexibility in
    configuration and output formats.

    Attributes:
        songs_path (str): The full path to the folder containing songs for processing.
        fe (FeatureExtractor): An instance of `FeatureExtractor` used for audio fragmenting
            and spectrogram generation.
        retrieve_specs_fn (dict): A mapping of spectrogram types ('mel', 'std') to their
            corresponding extraction methods in `FeatureExtractor`.
        n_mels (int): Number of mel bands for mel spectrogram extraction.
        n_seconds (int): Duration of each audio fragment in seconds.
        spec_type (Literal['mel', 'std']): The default spectrogram type for extraction.
        retrieve_counter (int): A counter used for cycling through example tags,
            to be replaced with real metadata retrieval logic.

    Args:
        songs_folder_path (str): The relative path to the folder containing the songs to process.

    Note: The `retrieve_tags` method currently returns simulated tags for demonstration purposes.
    """
    def __init__(self, songs_folder_path: str):
        self.songs_path = os.path.join(PROJECT_FOLDER_DIR, songs_folder_path)
        self.fe = FeatureExtractor(self.songs_path)
        self.retrieve_specs_fn = {
            'mel': self.fe.extract_mel_spec_from_fragments,
            'std': self.fe.extract_specs_from_fragments
        }

        self.n_mels, self.n_seconds, self.spec_type = [None for _ in range(3)]
        self.song_tags = None  # to be removed after implementing real tags retrieving

    def set_config(self, n_mels: int, n_seconds: int, spec_type: Literal['mel', 'std'], labels_path: str):
        """
        Sets the pipeline settings for spectrogram extraction.

        Args:
            n_mels (int): The number of mel bands for mel spectrogram computation.
            n_seconds (int): The duration of each audio fragment in seconds.
            spec_type (Literal['mel', 'std']): The type of spectrogram to use.
                - `'mel'`: Mel spectrogram.
                - `'std'`: Standard spectrogram.
            labels_path (str): Path **(including)** to a JSON file containing multi-hot encoded tags
        """
        self.n_mels = n_mels
        self.n_seconds = n_seconds
        self.spec_type = spec_type

        self.song_tags = read_json_to_dict(labels_path)

    def _check_if_config_is_set(self):
        """
        Ensures that the pipeline configuration has been properly set.

        Raises:
            Exception: If any of the configuration parameters are not set.
        """
        if not all([self.n_mels, self.n_seconds, self.spec_type, self.song_tags]):
            raise Exception('SpectogramPipeline._config_is_set(): Pipeline config must be set before usage.')

    def get_song_specs(
            self,
            song_path: str,
            song_title: str,
            song_tags: List[str],
            return_dict: bool = False
    ) -> SongSpecDataDictType | List[Tuple[str, np.ndarray, List[str]]] | None:
        """
        Extracts spectrograms for a single song and organizes the data.

        This method splits the song into fragments and computes the spectrograms
        based on the specified type (`mel` or `std`). The results are returned
        either as a dictionary or as a list of tuples, depending on the `return_dict` flag.

        Args:
            song_path (str): The path to the song file.
            song_title (str): The title of the song.
            song_tags (List[str]): Tags or labels associated with the song.
            return_dict (bool): If `True`, returns the data as a dictionary.
                If `False`, returns the data as a list of tuples.

        Returns:
            SongSpecDataDictType | List[Tuple[str, np.ndarray, List[str]]]:
                - If `return_dict=True`: A dictionary with the following structure:
                    {
                        'title': str,
                        'samples': List[np.ndarray],
                        'tags': List[str]
                    }
                - If `return_dict=False`: A list of tuples, where each tuple contains:
                    (song_title, spectrogram_fragment, song_tags).
        """
        self._check_if_config_is_set()

        fragments, sample_rate = self.fe.make_fragments(
            song_path,
            n_seconds=self.n_seconds
        )

        if fragments is None or sample_rate is None:
            return None
        song_specs = self.retrieve_specs_fn[self.spec_type](fragments, sr=sample_rate, n_mels=self.n_mels)
        n_specs = len(song_specs)
        if return_dict:
            return {
                'title': song_title,
                'samples': song_specs,
                'tags': song_tags
            }
        return list(
            zip(
                [song_title for _ in range(n_specs)],
                song_specs,
                [song_tags for _ in range(n_specs)]
            )
        )

    def get_data_from_songs(
            self,
            return_list_of_dct: bool = False
    ) -> List[SongSpecDataDictType] | List[Tuple[str, np.ndarray, List[str]]]:
        """
        Processes all songs in the specified folder and extracts their spectrogram data
        using 'get_song_specs' method.

        Args:
            return_list_of_dct (bool):
                - If `True`, returns a list of dictionaries, where each dictionary contains
                  the spectrograms, title, and tags for a song.
                - If `False`, returns a flat list of tuples, where each tuple contains:
                  (song_title, spectrogram_fragment, song_tags).

        Returns:
            List[SongSpecDataDictType] | List[Tuple[str, np.ndarray, List[str]]]:
                - If `return_list_of_dct=True`: A list of dictionaries with the following structure:
                    {
                        'title': str,
                        'samples': List[np.ndarray],
                        'tags': List[str]
                    }
                - If `return_list_of_dct=False`: A flat list of tuples, where each tuple contains:
                    (song_title, spectrogram_fragment, song_tags).
        """
        self._check_if_config_is_set()

        res = []
        for song in os.listdir(self.songs_path):
            song_path = os.path.join(self.songs_path, song)
            song_title = song[:-4]
            song_tags = self.retrieve_tags(song_title)
            song_data = self.get_song_specs(
                song_path=song_path,
                song_title=song_title,
                song_tags=song_tags,
                return_dict=return_list_of_dct
            )
            if song_data is None:
                continue
            else:
                print(f"Currently processing {song_title}")

            # remove after trials
            if len(res) > 1000:
                return res

            if return_list_of_dct:
                res.append(song_data)
            else:
                res.extend(song_data)
        return res

    def retrieve_tags(self, song_title) -> List[str]:
        """
        """
        return self.song_tags[song_title]

    def save_config(self, path: str) -> ConfigType:
        """
        Saves the current configuration settings of the pipeline to a YAML file.

        Args:
            path (str): The path of the configuration file **(including file and file extension)**.

        Returns:
            ConfigType: A dictionary containing the configuration settings with the following keys:
                - `n_mels` (int): The number of mel bands used for mel spectrogram computation.
                - `n_seconds` (int): The duration of each audio fragment in seconds.
                - `spec_type` (Literal['mel', 'std']): The type of spectrogram being used.
                    - `'mel'`: Mel spectrogram.
                    - `'std'`: Standard spectrogram.
        """
        self._check_if_config_is_set()

        cfg_dct = {
            'n_mels': self.n_mels,
            'n_seconds': self.n_seconds,
            'spec_type': self.spec_type
        }

        write_dict_to_json(cfg_dct, path)

        return cfg_dct

    @staticmethod
    def multi_hot_tags_of_all_songs():
        all_tags = get_all_tags(TAGS_DIR)
        print(all_tags)
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

        song_tags_indexes = tags_to_tags_indexes(song_tags_str, all_tags)
        multi_hotted = multi_hot_batch(song_tags_indexes, len(all_tags))
        save_multi_hotted_labels(song_titles, multi_hotted, os.path.join(LABELS_DIR, 'labels.json'))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import librosa
    ppl = SpectogramPipeline(SONGS_DIR)

    ppl.set_config(
        n_mels=N_MELS,
        n_seconds=N_SECONDS,
        spec_type=SPEC_TYPE,
        labels_path=os.path.join(LABELS_DIR, 'labels.json')
    )

    # Snippet 1
    # spec = ppl.get_song_specs(
    #     os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'music', 'About_To_Go_Down_-_Michael_White__Deflo.mp3'),
    #     'About_To_Go_Down_-_Michael_White__Deflo.mp3',
    #     ['dummy_tag'],
    #     return_dict=True
    # )["samples"][0]
    # print(spec.shape, type(spec))
    # fig, ax = plt.subplots(figsize=(14, 7))
    # img = librosa.display.specshow(spec, x_axis='time', y_axis='log', ax=ax)
    # fig.colorbar(img, ax=ax)
    # plt.show()

    # Snippet 2
    data = ppl.get_data_from_songs()
    for sample in data[40:50]:
        print(sample[0], sample[1].shape, sample[2])
        print()

    # Snippet 3
    # main_folder_path = os.path.join(PROJECT_FOLDER_DIR, 'test_config.json')
    # ppl.save_config(main_folder_path)

    # Snippet 4
    ppl.multi_hot_tags_of_all_songs()
