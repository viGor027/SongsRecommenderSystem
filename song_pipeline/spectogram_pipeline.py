import torch
from song_pipeline.feature_extractor import FeatureExtractor
from song_pipeline.constants import N_MELS, N_SECONDS, STEP, SPEC_TYPE, PROJECT_FOLDER_DIR, \
    TAGS_DIR, LABELS_DIR, SONGS_DIR, DATA_DIR
from song_pipeline.dict_types import ConfigType, SongSpecDataDictType
from song_pipeline.utils import write_dict_to_json, read_json_to_dict, get_all_tags, \
    multi_hot_batch, tags_to_tags_indexes, save_multi_hotted_labels, prepare_for_dataset
from typing import Literal
import numpy as np
import os


class SpectogramPipeline:
    """
    A pipeline for processing audio files into spectrogram data and preparing it for use in machine learning models.

    This class provides functionality to:
        - Configure spectrogram extraction parameters (`set_config`).
        - Extract spectrograms for individual songs or entire directories (`get_song_specs`, `get_data_from_songs`).
        - Convert processed data into ready-to-use PyTorch tensors (`get_dataset_ready_data`).
        - Save and retrieve pipeline configurations (`save_config`).
        - Generate and save multi-hot encoded labels for all songs (`multi_hot_tags_of_all_songs`).

    Attributes:
        songs_path (str): Path to the folder containing songs.
        fe (FeatureExtractor): Handles audio fragmenting and spectrogram generation.
        retrieve_specs_fn (dict): Maps spectrogram types ('mel', 'std') to corresponding extraction methods.
        n_mels (int): Number of mel bands for mel spectrograms.
        n_seconds (int): Duration of audio fragments in seconds.
        step (int | float): Defines the interval (in seconds) at which consecutive fragments start within the audio.
        spec_type (str): Type of spectrogram to extract ('mel' or 'std').
        song_tags (dict): Multi-hot encoded tags for the songs.
    """

    def __init__(self, songs_folder_path: str):
        self.songs_path = os.path.join(PROJECT_FOLDER_DIR, songs_folder_path)
        self.fe = FeatureExtractor(self.songs_path)
        self.retrieve_specs_fn = {
            'mel': self.fe.extract_mel_spec_from_fragments,
            'std': self.fe.extract_specs_from_fragments
        }

        self.n_mels, self.n_seconds, self.spec_type, self.step = [None for _ in range(4)]
        self.song_tags = None

    def set_config(
            self, n_mels: int, n_seconds: int, step: int | float,
            spec_type: Literal['mel', 'std'], labels_path: str
    ):
        """
        Sets the pipeline settings for spectrogram extraction.

        Args:
            n_mels (int): The number of mel bands for mel spectrogram computation.
            n_seconds (int): The duration of each audio fragment in seconds.
            spec_type (Literal['mel', 'std']): The type of spectrogram to use.
                - `'mel'`: Mel spectrogram.
                - `'std'`: Standard spectrogram.
            step (int | float): Defines the interval (in seconds) at which consecutive fragments start within the audio.
                        Set to n_seconds for fragments to be non-overlapping.
            labels_path (str): Path to a JSON file containing multi-hot encoded tags.
        """
        self.n_mels = n_mels
        self.n_seconds = n_seconds
        self.spec_type = spec_type
        self.step = step

        try:
            self.song_tags = read_json_to_dict(labels_path)
        except FileNotFoundError as e:
            print("SpectrogramPipeline.set_config() FileNotFoundError: Use "
                  "SpectrogramPipeline.multi_hot_tags_of_all_songs() or pass valid labels_path")
            raise Exception(e)

    def _check_if_config_is_set(self):
        """
        Ensures that the pipeline configuration has been properly set.

        Raises:
            Exception: If any of the configuration parameters are not set.
        """
        if not all([self.n_mels, self.n_seconds, self.spec_type, self.song_tags, self.step]):
            raise Exception('SpectrogramPipeline._config_is_set(): Pipeline config must be set before usage.')

    def get_song_specs(
            self,
            song_path: str,
            song_title: str,
            song_tags: list[int],
            return_dict: bool = False
    ) -> SongSpecDataDictType | list[tuple[str, np.ndarray, list[int]]] | None:
        """
        Extracts spectrograms for a single song and organizes the data.

        Note: If `return_list_of_dct=False` each tuple has the same song_title and song_tags
            **(returns single song in a wide format)**.

        Args:
            song_path (str): The path to the song file.
            song_title (str): The title of the song.
            song_tags (list[int]): Multi-hot encoded tags for a song.
            return_dict (bool): If `True`, returns the data as a dictionary.
                If `False`, returns the data as a list of tuples.

        Returns:
            Depending on `return_list_of_dct':
                - If `return_list_of_dct=True`: A dictionary with the following structure:
                {'title': str, 'samples': list[np.ndarray], 'tags': list[int]}

                - If `return_list_of_dct=False`: A flat list of tuples, where each tuple contains:
                (song_title, spectrogram_fragment, song_tags).
        """
        self._check_if_config_is_set()

        fragments, sample_rate = self.fe.make_fragments(
            song_path,
            n_seconds=self.n_seconds,
            step=self.step
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
    ) -> list[SongSpecDataDictType] | list[tuple[str, np.ndarray, list[int]]]:
        """
        Processes all songs in the `self.song_path` directory and extracts their spectrogram data
        using 'get_song_specs' method.

        Args:
            return_list_of_dct (bool): Specifies the format of returned data.

        Returns:
            Depending on `return_list_of_dct':
                - If `return_list_of_dct=True`: A list of dictionaries with the following structure:
                {'title': str, 'samples': list[np.ndarray], 'tags': list[int]}

                - If `return_list_of_dct=False`: A flat list of tuples, where each tuple contains:
                (song_title, spectrogram_fragment, song_tags).
        """
        self._check_if_config_is_set()

        res = []
        for song in os.listdir(self.songs_path):
            song_path = os.path.join(self.songs_path, song)
            song_title = song[:-4]
            song_tags = self._retrieve_tags(song_title)
            song_data = self.get_song_specs(
                song_path=song_path,
                song_title=song_title,
                song_tags=song_tags,
                return_dict=return_list_of_dct
            )
            if song_data is None:
                continue
            else:
                print(f"Processed {song_title}")

            if return_list_of_dct:
                res.append(song_data)
            else:
                res.extend(song_data)
        return res

    def _retrieve_tags(self, song_title: str) -> list[int]:
        """
        Args:
            song_title (str): The title of the song whose tags we want to receive.
        Returns:
            Multi-hot encoded tags for a song.
        """
        return self.song_tags[song_title]

    def get_dataset_ready_data(self, save_data: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processes all the downloaded songs to be ready to pass into torch Dataset.

        Args:
            save_data (bool): whether to save X, Y tensors to a file in `DATA_DIR`.
                              If set to True tensors are saved together with pipeline config into `DATA_DIR`.

        Returns:
            X and Y tensors of the songs.
        """
        data = self.get_data_from_songs()
        X, Y = prepare_for_dataset(data, shuffle=True)
        if save_data:
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
            torch.save(X, os.path.join(DATA_DIR, 'X.pt'))
            torch.save(Y, os.path.join(DATA_DIR, 'Y.pt'))
            self.save_config(os.path.join(DATA_DIR, 'pipeline_config.json'))

        return X, Y

    def save_config(self, path: str) -> ConfigType:
        """
        Saves the current configuration settings of the pipeline to a JSON file.

        Args:
            path (str): The path of the configuration file **(including file and file extension)**.

        Returns:
            ConfigType: A dictionary containing the configuration settings with the following keys:
                - `n_mels` (int)
                - `n_seconds` (int)
                - `spec_type` (Literal['mel', 'std'])
        """
        self._check_if_config_is_set()

        cfg_dct = {
            'n_mels': self.n_mels,
            'n_seconds': self.n_seconds,
            'step': self.step,
            'spec_type': self.spec_type
        }

        write_dict_to_json(cfg_dct, path)

        return cfg_dct

    @staticmethod
    def get_broken():
        broken_dct = dict()
        broken_dct['broken_songs'] = FeatureExtractor.logger
        write_dict_to_json(broken_dct, os.path.join(DATA_DIR, 'broken_songs.json'))

    @staticmethod
    def multi_hot_tags_of_all_songs():
        """
        Encodes the tags of all songs in a `TAGS_DIR` directory into multi-hot format,
        saves the resulting data and labels mapping to `LABELS_DIR`.
        """
        all_tags = get_all_tags(TAGS_DIR)
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
        write_dict_to_json(
            data={i: tag for i, tag in enumerate(all_tags)},
            file_path=os.path.join(LABELS_DIR, 'mapping.json')
        )


if __name__ == "__main__":
    # Run below snippet to prepare data after scraping it
    ppl = SpectogramPipeline(SONGS_DIR)
    ppl.multi_hot_tags_of_all_songs()
    ppl.set_config(
        n_mels=N_MELS,
        n_seconds=N_SECONDS,
        spec_type=SPEC_TYPE,
        step=STEP,
        labels_path=os.path.join(LABELS_DIR, 'labels.json')
    )
    ppl.get_dataset_ready_data(save_data=True)
    ppl.get_broken()
