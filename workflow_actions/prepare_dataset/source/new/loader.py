import librosa
from pathlib import Path
import numpy as np


def load_single_song(song_path: Path) -> tuple[np.ndarray, int]:
    song, sample_rate = librosa.load(song_path)
    return song, sample_rate


if __name__ == "__main__":
    path = Path("D:\\Nauka\\Projekty\\SongsRecommenderSystem\\data\\raw\\downloaded_songs\\ALEXYS,Strn_-So_Sweet.mp3")
    song, sample_rate = load_single_song(song_path=path)
    print(song.shape, type(song))
    print(sample_rate, type(sample_rate))
