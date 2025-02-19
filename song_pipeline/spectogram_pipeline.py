import torch
from song_pipeline.tag_processor import TagProcessor
from song_pipeline.feature_extractor import FeatureExtractor
from song_pipeline.constants import PROJECT_FOLDER_DIR, SONGS_DIR, DATA_DIR, \
    N_MELS, LABELS_DIR, N_SECONDS, SPEC_TYPE, STEP, BATCH_SIZE
from song_pipeline.dict_types import ConfigType, SongSpecDataType
from song_pipeline.utils import write_dict_to_json, read_json_to_dict
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

        self.n_mels, self.n_seconds, self.spec_type, self.step, self.validation_probability, self.bath_size = [None for _ in range(6)]
        self.song_tags = None

    def set_config(
            self, n_mels: int, n_seconds: int, step: int | float,
            validation_probability: float, batch_size: int,
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
            batch_size (int): Batch size.
            labels_path (str): Path to a JSON file containing multi-hot encoded tags.
        """
        self.n_mels = n_mels
        self.n_seconds = n_seconds
        self.spec_type = spec_type
        self.step = step
        self.validation_probability = validation_probability
        self.bath_size = batch_size

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
                [self.n_mels, self.n_seconds, self.spec_type, self.song_tags, self.step, self.bath_size]) or \
                self.validation_probability is None:
            raise Exception('SpectrogramPipeline._config_is_set(): Pipeline config must be set before usage.')

    def get_song_specs(
            self,
            song_path: str,
            song_title: str,
            song_tags: list[int],
            return_dict: bool = False
    ) -> tuple[SongSpecDataType, SongSpecDataType] | \
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

        if not all([train_fragments, validation_fragments, sample_rate]):
            return None, None
        training_specs = self.retrieve_specs_fn[self.spec_type](train_fragments, sr=sample_rate, n_mels=self.n_mels)
        validation_specs = self.retrieve_specs_fn[self.spec_type](
            validation_fragments,
            sr=sample_rate, n_mels=self.n_mels)
        n_validation_specs = len(validation_specs)
        n_train_specs = len(training_specs)
        if return_dict:
            training_data: SongSpecDataType = {
                'title': song_title,
                'samples': training_specs,
                'tags': song_tags
            }
            validation_data: SongSpecDataType = {
                'title': song_title,
                'samples': validation_specs,
                'tags': song_tags
            }
            return training_data, validation_data
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

    def make_dataset_ready_data(self) -> None:
        """
        Processes all the downloaded songs to be ready to pass into torch Dataset.
        Saves batches as tensors to respective folders in `DATA_DIR`.
        """
        self._check_if_config_is_set()

        last_training_batch_idx = SpectogramPipeline.get_last_batch_index(set_label='train')
        last_valid_batch_idx = SpectogramPipeline.get_last_batch_index(set_label='valid')
        train_data, valid_data = self.get_data_from_songs()

        # training data
        X_train, Y_train = SpectogramPipeline._prepare_for_dataset(train_data, shuffle=True)
        self.batch_and_store_data(X_train, Y_train, batch_size=self.batch_size,
                                  set_label='train', last_batch_idx=last_training_batch_idx)

        # validation data
        X_valid, Y_valid = SpectogramPipeline._prepare_for_dataset(valid_data, shuffle=True)
        self.batch_and_store_data(X_valid, Y_valid, batch_size=self.batch_size,
                                  set_label='valid', last_batch_idx=last_valid_batch_idx)

        self.save_config(os.path.join(DATA_DIR, 'pipeline_config.json'))

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
            'batch_size': self.bath_size,
            'spec_type': self.spec_type
        }

        write_dict_to_json(cfg_dct, path)
        print(f"SpectogramPipeline config was saved to {path}")

        return cfg_dct

    @staticmethod
    def get_last_batch_index(set_label: Literal['train', 'valid']):
        """
        Args:
            set_label (Literal['train', 'valid']): Whether to return last batch index of training or validation set.
        """
        items = os.listdir(os.path.join(DATA_DIR, set_label))
        sorted_items = sorted(items)
        if len(sorted_items):
            last = sorted_items[-1]
            underscore_idx = last.index("_")
            dot_idx = last.index(".")
            return int(last[underscore_idx+1:dot_idx])
        else:
            print("SpectogramPipeline.get_last_batch_index: There is not a single batch in the directory.")
            return -1

    @staticmethod
    def _save_batch(X: torch.Tensor, Y: torch.Tensor, set_label: Literal['train', 'valid'], batch_index: int):
        """
        Utility for saving batch of data.

        Args:
            X (torch.Tensor): Input tensor containing the feature data.
            Y (torch.Tensor): Target tensor containing the labels.
            set_label (Literal['train', 'valid']): Whether batch belongs to training or validation set.
            batch_index (int): Number used for ordering or identifying batches.
        """
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        torch.save(X, os.path.join(DATA_DIR, set_label, f'X_{batch_index}.pt'))
        torch.save(Y, os.path.join(DATA_DIR, set_label, f'Y_{batch_index}.pt'))

    @staticmethod
    def batch_and_store_data(X: torch.Tensor, Y: torch.Tensor, batch_size: int,
                             set_label: Literal['train', 'valid'], last_batch_idx):
        """
        Args:
            X (torch.Tensor): Input tensor containing the feature data.
            Y (torch.Tensor): Target tensor containing the labels.
            batch_size (int): Number of samples per batch.
            set_label (Literal['train', 'valid']): Specifies whether the data belongs to the training or validation set.
            last_batch_idx: Index of the last batch that is currently stored in the respective data folder.
        """
        for batch_num in range(len(X) // batch_size):
            X_batch = X[batch_size * batch_num:batch_size * (batch_num + 1)].clone()
            Y_batch = Y[batch_size * batch_num:batch_size * (batch_num + 1)].clone()
            SpectogramPipeline._save_batch(X_batch, Y_batch,
                                           set_label=set_label,
                                           batch_index=batch_num + last_batch_idx+1)
        leftover_begin_idx = batch_size * (len(X) // batch_size)
        X_batch = X[leftover_begin_idx:].clone()
        Y_batch = Y[leftover_begin_idx:].clone()
        SpectogramPipeline._save_batch(X_batch, Y_batch,
                                       set_label=set_label,
                                       batch_index=len(X) // batch_size)

    @staticmethod
    def get_broken():
        """Helper utility"""
        broken_dct = dict()
        broken_dct['broken_songs'] = FeatureExtractor.logger
        write_dict_to_json(broken_dct, os.path.join(DATA_DIR, 'broken_songs.json'))

    @staticmethod
    def _prepare_for_dataset(data: list[tuple[str, np.ndarray, list[int]]],
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
        X = (np.stack(X, dtype=np.float16) + 80.) / 80.
        Y = np.stack(Y, dtype=np.float16)
        return (list(titles), torch.from_numpy(X), torch.from_numpy(Y)) if return_titles else (
                torch.from_numpy(X), torch.from_numpy(Y))


if __name__ == "__main__":
    # Run below snippet to prepare data after scraping it
    tp = TagProcessor()
    tp.multi_hot_tags_of_all_songs()
    for i in range(1, 3+1):
        ppl = SpectogramPipeline(os.path.join(SONGS_DIR, f'music{i}'))
        ppl.set_config(
            n_mels=N_MELS,
            n_seconds=N_SECONDS,
            spec_type=SPEC_TYPE,
            step=STEP,
            validation_probability=0.08,
            batch_size=BATCH_SIZE,
            labels_path=os.path.join(LABELS_DIR, 'labels.json')
        )
        ppl.make_dataset_ready_data()
