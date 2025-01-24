import torch
from song_pipeline.feature_extractor import FeatureExtractor
from song_pipeline.constants import N_MELS, N_SECONDS, STEP, SPEC_TYPE, PROJECT_FOLDER_DIR, \
    TAGS_DIR, LABELS_DIR, SONGS_DIR, DATA_DIR
from song_pipeline.dict_types import ConfigType, SongSpecDataType
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
        validation_probability (float): Defines a chance of fragment being added to a validation set;
                        fragment gets to a training set with a chance equal to 1-validation_probability.
        song_tags (dict): Multi-hot encoded tags for the songs.
    """

    def __init__(self, songs_folder_path: str):
        self.songs_path = os.path.join(PROJECT_FOLDER_DIR, songs_folder_path)
        self.fe = FeatureExtractor(self.songs_path)
        self.retrieve_specs_fn = {
            'mel': self.fe.extract_mel_spec_from_fragments,
            'std': self.fe.extract_specs_from_fragments
        }

        self.n_mels, self.n_seconds, self.spec_type, self.step, self.validation_probability = [None for _ in range(5)]
        self.song_tags = None

    def set_config(
            self, n_mels: int, n_seconds: int, step: int | float, validation_probability: float,
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
            validation_probability (float): Defines a chance of fragment being added to a validation set;
                        fragment gets to a training set with a chance equal to 1-validation_probability.
            labels_path (str): Path to a JSON file containing multi-hot encoded tags.
        """
        self.n_mels = n_mels
        self.n_seconds = n_seconds
        self.spec_type = spec_type
        self.step = step
        self.validation_probability = validation_probability

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
        if not all(
                [self.n_mels, self.n_seconds, self.spec_type, self.song_tags, self.step, self.validation_probability]):
            raise Exception('SpectrogramPipeline._config_is_set(): Pipeline config must be set before usage.')

    def get_song_specs(
            self,
            song_path: str,
            song_title: str,
            song_tags: list[int],
            return_dict: bool = False
    ) -> SongSpecDataType | \
            tuple[list[tuple[str, np.ndarray, list[int]]], list[tuple[str, np.ndarray, list[int]]]] | \
            tuple[None, None]:
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
                {'title': str, 'validation_samples': list[np.ndarray], 'training_samples': list[np.ndarray], 'tags': list[int]}

                - If `return_list_of_dct=False`: Tuple containing two flat list of tuples, where each tuple contains:
                (song_title, spectrogram_fragment, song_tags); first element of a tuple contains training samples;
                second element of a tuple contains validation samples.
        """
        self._check_if_config_is_set()

        train_fragments, validation_fragments, sample_rate = self.fe.make_fragments(
            song_path,
            n_seconds=self.n_seconds,
            validation_probability=self.validation_probability,
            step=self.step
        )

        if train_fragments is None or validation_fragments is None or sample_rate is None:
            return None, None
        training_specs = self.retrieve_specs_fn[self.spec_type](train_fragments, sr=sample_rate, n_mels=self.n_mels)
        validation_specs = self.retrieve_specs_fn[self.spec_type](
            validation_fragments,
            sr=sample_rate, n_mels=self.n_mels)
        n_validation_specs = len(validation_specs)
        n_train_specs = len(training_specs)
        if return_dict:
            return {
                'title': song_title,
                'validation_samples': validation_specs,
                'training_samples': training_specs,
                'tags': song_tags
            }
        return list(
            zip(
                [song_title for _ in range(n_train_specs)],
                training_specs,
                [song_tags for _ in range(n_train_specs)]
            )
        ), list(
            zip(
                [song_title for _ in range(n_validation_specs)],
                validation_specs,
                [song_tags for _ in range(n_validation_specs)]
            )
        )

    def get_data_from_songs(
            self,
            return_list_of_dct: bool = False
    ) -> tuple[list[SongSpecDataType], list[SongSpecDataType]] | \
            tuple[list[tuple[str, np.ndarray, list[int]]], list[tuple[str, np.ndarray, list[int]]]]:
        """
        Processes all songs in the `self.song_path` directory and extracts their spectrogram data
        using 'get_song_specs' method.

        Args:
            return_list_of_dct (bool): Specifies the format of returned data.

        Returns:
            Depending on `return_list_of_dct':
                - If `return_list_of_dct=True`: A tuple containing two lists of dictionaries with the following structure:
                {'title': str, 'samples': list[np.ndarray], 'tags': list[int]};
                first tuple item is training data, second tuple item is validation data.

                - If `return_list_of_dct=False`: A tuple containing two flat list of tuples, where each tuple contains:
                (song_title, spectrogram_fragment, song_tags);
                first tuple item is training data, second tuple item is validation data.
        """
        self._check_if_config_is_set()

        train, valid = [], []
        for song in os.listdir(self.songs_path):
            song_path = os.path.join(self.songs_path, song)
            song_title = song[:-4]
            song_tags = self._retrieve_tags(song_title)
            training_data, validation_data = self.get_song_specs(
                song_path=song_path,
                song_title=song_title,
                song_tags=song_tags,
                return_dict=return_list_of_dct
            )
            if training_data is None or validation_data is None:
                continue
            else:
                print(f"Processed {song_title}")

            if return_list_of_dct:
                valid.append(validation_data)
                train.append(training_data)
            else:
                valid.extend(validation_data)
                train.extend(training_data)
        return train, valid

    def _retrieve_tags(self, song_title: str) -> list[int]:
        """
        Args:
            song_title (str): The title of the song whose tags we want to receive.
        Returns:
            Multi-hot encoded tags for a song.
        """
        return self.song_tags[song_title]

    def make_dataset_ready_data(self, set_num: int) -> None:
        """
        Processes all the downloaded songs to be ready to pass into torch Dataset.
        Saves tensors containing training and validation data to `DATA_DIR`.

        Returns:
            None
        """
        train_data, valid_data = self.get_data_from_songs()
        X_train, Y_train = prepare_for_dataset(train_data, shuffle=True)
        self._save_data(X_train, Y_train, set_label='train', set_num=set_num)
        X_valid, Y_valid = prepare_for_dataset(valid_data, shuffle=True)
        self._save_data(X_valid, Y_valid, set_label='valid', set_num=set_num)
        self.save_config(os.path.join(DATA_DIR, 'pipeline_config.json'))

    @staticmethod
    def _save_data(X: torch.Tensor, Y: torch.Tensor, set_label: Literal['train', 'valid'], set_num: int):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        torch.save(X, os.path.join(DATA_DIR, f'X_{set_num}_{set_label}.pt'))
        torch.save(Y, os.path.join(DATA_DIR, f'Y_{set_num}_{set_label}.pt'))
        print(f"dataset saved to {DATA_DIR}")

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
            'validation_probability': self.validation_probability,
            'spec_type': self.spec_type
        }

        write_dict_to_json(cfg_dct, path)
        print(f"SpectogramPipeline config was saved to{path}")

        return cfg_dct

    @staticmethod
    def get_broken():
        """Helper utility"""
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
    ppl = SpectogramPipeline(os.path.join(SONGS_DIR, 'music1'))
    ppl.multi_hot_tags_of_all_songs()
    ppl.set_config(
        n_mels=N_MELS,
        n_seconds=N_SECONDS,
        spec_type=SPEC_TYPE,
        step=STEP,
        validation_probability=0.08,
        labels_path=os.path.join(LABELS_DIR, 'labels.json')
    )
    ppl.make_dataset_ready_data(set_num=1)

    ppl = SpectogramPipeline(os.path.join(SONGS_DIR, 'music2'))
    ppl.set_config(
        n_mels=N_MELS,
        n_seconds=N_SECONDS,
        spec_type=SPEC_TYPE,
        step=STEP,
        validation_probability=0.08,
        labels_path=os.path.join(LABELS_DIR, 'labels.json')
    )
    ppl.make_dataset_ready_data(set_num=2)

    ppl = SpectogramPipeline(os.path.join(SONGS_DIR, 'music3'))
    ppl.set_config(
        n_mels=N_MELS,
        n_seconds=N_SECONDS,
        spec_type=SPEC_TYPE,
        step=STEP,
        validation_probability=0.08,
        labels_path=os.path.join(LABELS_DIR, 'labels.json')
    )
    ppl.make_dataset_ready_data(set_num=3)
