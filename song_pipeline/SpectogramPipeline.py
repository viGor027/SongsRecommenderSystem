from FeatureExtractor import FeatureExtractor
from constants import N_MELS, N_SECONDS, SPEC_TYPE, PROJECT_FOLDER_DIR
from typing import Literal, List, Tuple
from dict_types import ConfigType, SongSpecDataDictType
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

        self.n_mels = N_MELS
        self.n_seconds = N_SECONDS
        self.spec_type = SPEC_TYPE

        self.retrieve_counter = -1  # to be removed after implementing real tags retrieving

    def get_song_specs(
            self,
            song_path: str,
            song_title: str,
            song_tags: List[str],
            spec_type: Literal['mel', 'std'],
            return_dict: bool = False
    ) -> SongSpecDataDictType | List[Tuple[str, np.ndarray, List[str]]]:
        """
        Extracts spectrograms for a single song and organizes the data.

        This method splits the song into fragments and computes the spectrograms
        based on the specified type (`mel` or `std`). The results are returned
        either as a dictionary or as a list of tuples, depending on the `return_dict` flag.

        Args:
            song_path (str): The path to the song file.
            song_title (str): The title of the song.
            song_tags (List[str]): Tags or labels associated with the song.
            spec_type (Literal['mel', 'std']): The type of spectrogram to compute:
                - `'mel'`: Mel spectrogram.
                - `'std'`: Standard spectrogram.
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

        fragments, sample_rate = self.fe.make_fragments(
            song_path,
            n_seconds=self.n_seconds
        )
        song_specs = self.retrieve_specs_fn[spec_type](fragments, sr=sample_rate, n_mels=self.n_mels)
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
        Processes all songs in the specified folder and extracts their spectrogram data.

        This method iterates through the songs in the folder, splits each into fragments,
        computes spectrograms, and organizes the results. The returned data can either be
        a list of dictionaries (one per song) or a flat list of tuples, depending on the
        `return_list_of_dct` flag.

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
        res = []
        for song in os.listdir(self.songs_path):
            song_path = os.path.join(self.songs_path, song)
            song_title = song[:-4]
            song_tags = self.retrieve_tags()
            song_data = self.get_song_specs(
                song_path=song_path,
                song_title=song_title,
                song_tags=song_tags,
                spec_type=self.spec_type,
                return_dict=return_list_of_dct
            )

            if return_list_of_dct:
                res.append(song_data)
            else:
                res.extend(song_data)
        return res

    def retrieve_tags(self) -> List[str]:
        """
        Version of method for demonstration purposes.

        :return: dummy tags
        """
        tags = {
            0: ['tag1', 'tag2'],
            1: ['tag3', 'tag4'],
            2: ['tag5', 'tag6', 'tag7']
        }
        self.retrieve_counter += 1
        return tags[self.retrieve_counter % 3]

    def config(self) -> ConfigType:
        """
        Retrieves the current configuration settings of the pipeline.

        This method returns the configuration parameters used for spectrogram extraction,
        including the number of mel bands, fragment duration, and spectrogram type.

        Returns:
            ConfigType: A dictionary containing the configuration settings with the following keys:
                - `n_mels` (int): The number of mel bands used for mel spectrogram computation.
                - `n_seconds` (int): The duration of each audio fragment in seconds.
                - `spec_type` (Literal['mel', 'std']): The type of spectrogram being used.
                    - `'mel'`: Mel spectrogram.
                    - `'std'`: Standard spectrogram.
        """
        return {
            'n_mels': self.n_mels,
            'n_seconds': self.n_seconds,
            'spec_type': self.spec_type
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import librosa
    ppl = SpectogramPipeline(os.path.join(PROJECT_FOLDER_DIR, 'sample'))
    # Snippet 1
    spec = None
    for sample in ppl.get_data_from_songs():
        print(sample[0], sample[2])
        if 'tag6' in sample[2]:
            spec = sample[1]
    fig, ax = plt.subplots(figsize=(14, 7))
    img = librosa.display.specshow(spec, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax)
    plt.show()

    # Snippet 2
    # spec = None
    # for sample in ppl.get_data_from_songs(return_list_of_dct=True):
    #     print(sample['title'], len(sample['samples']), sample['tags'])
    #     if 'tag6' in sample['tags']:
    #         spec = sample['samples'][4]
    # fig, ax = plt.subplots(figsize=(14, 7))
    # img = librosa.display.specshow(spec, x_axis='time', y_axis='log', ax=ax)
    # fig.colorbar(img, ax=ax)
    # plt.show()
