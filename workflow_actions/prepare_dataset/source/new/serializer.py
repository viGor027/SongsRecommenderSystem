from pathlib import Path
import numpy as np
import torch
import torchaudio
from typing import Optional
import librosa


def save_numpy_fragment(fragment: np.ndarray, path: Path):
    np.save(str(path), fragment)


def load_numpy_fragment(path: Path) -> np.ndarray:
    return np.load(str(path))


def load_single_song_to_numpy(path: Path) -> tuple[np.ndarray, int]:
    song, sample_rate = librosa.load(path)
    return song, sample_rate


def load_single_song_to_torch(
        path: Path,
        new_sample_rate: Optional[int]
) -> tuple[torch.Tensor, int]:
    wave, sr = torchaudio.load(path)
    wave_mono = wave.mean(dim=0, keepdim=False)
    if new_sample_rate:
        wave_resampled = torchaudio.functional.resample(wave_mono, orig_freq=sr, new_freq=new_sample_rate)
        return wave_resampled, new_sample_rate
    return wave_mono, sr


# def save_tensor(path: Path, data: torch.Tensor) -> None:
#     torch.save(data, path)
#
# def load_tensor()


if __name__ == "__main__":
    from workflow_actions.paths import DOWNLOAD_DIR
    print(load_single_song_to_numpy(DOWNLOAD_DIR / "ALEXYS,Strn_-So_Sweet.mp3")[0].dtype)
