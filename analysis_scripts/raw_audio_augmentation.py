import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from workflow_actions.dataset_preprocessor.source import Serializer
from workflow_actions.dataset_preprocessor.source.raw_augment import RawAugment
from workflow_actions.dataset_preprocessor.source.spectrogram_extractor import (
    SpectrogramExtractor,
)

SONG_PATH = "D:\\Programowanie\\Python\\SongsRecommenderSystem\\data\\01_raw\\downloaded_songs\\1_K1-GODSLAYER.mp3"
SAMPLE_RATE = 22050
N_MELS = 80

serializer = Serializer(load_sample_rate=22050, serialize_song_wise=True)
song, sr = serializer.load_single_song_to_numpy(Path(SONG_PATH))

song = song[5 * SAMPLE_RATE : 10 * SAMPLE_RATE].copy()


extractor = SpectrogramExtractor(extraction_method="mel", sample_rate=sr, n_mels=N_MELS)
spec_orig = extractor([song])[0]

cmap = "viridis"

aug_defs = [
    ("PitchShift", {"min_semitones": -18, "max_semitones": -18, "p": 1.0}),
    # ("AddGaussianNoise", {"min_amplitude": 0.1, "max_amplitude": 0.1, "p": 1.0}),
    # ("Mp3Compression", {"min_bitrate": 16, "max_bitrate": 64, "p": 1.0}),
]

for aug_name, aug_params in aug_defs:
    augmenter = RawAugment([{"name": aug_name, "params": aug_params}])
    song_aug = augmenter([song], sample_rate=sr)[0]
    spec_aug = extractor([song_aug])[0]

    vmin = min(float(spec_orig.min()), float(spec_aug.min()))
    vmax = max(float(spec_orig.max()), float(spec_aug.max()))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axs[0].imshow(
        spec_orig, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axs[0].set_title("Original")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    im1 = axs[1].imshow(
        spec_aug, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axs[1].set_title("Augmented")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    fig.colorbar(im1, ax=axs, location="right", fraction=0.035, pad=0.02)
    fig.savefig(f"comparison_{aug_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
