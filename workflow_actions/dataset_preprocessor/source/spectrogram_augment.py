import torch
import torchaudio.transforms as T
import random


class SpectrogramAugment:
    """
    Spectrogram-level augmentation using torchaudio.transforms.

    To introduce new augmentation add entry in spectrogram_augment.augmentation list in
    prepare_dataset_config.json, this entry needs to contain `name` key which value
    corresponds to key in `_AUG_MAP`and `params` key that is dictionary containing every
    parameter that class from `_AUG_MAP` needs to be initialized with.
    """

    _AUG_MAP = {
        "TimeMasking": T.TimeMasking,
        "FrequencyMasking": T.FrequencyMasking,
    }

    def __init__(self, augmentations_p: float, augmentations: list[dict]):
        self.transforms = []
        self.p = augmentations_p

        for aug_cfg in augmentations:
            name = aug_cfg["name"]
            params = aug_cfg["params"]
            AugCls = self._AUG_MAP[name]
            transform = AugCls(**params)
            self.transforms.append(transform)

    def __call__(
        self,
        spectrograms: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        spectrograms (list[torch.Tensor]): spectrograms of shape [1, n_mels, len]
        """
        if not spectrograms:
            return []
        augmented_specs = []
        for spec in spectrograms:
            augmented_spec = spec
            for t in self.transforms:
                if random.random() < self.p:
                    augmented_spec = t(augmented_spec, mask_value=-80.0)
            augmented_specs.append(augmented_spec)
        return augmented_specs


if __name__ == "__main__":
    from workflow_actions.paths import MODEL_READY_DATA_DIR
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = MODEL_READY_DATA_DIR / "X_0.pt"
    X = torch.load(path)  # [1, 80, 216]

    augmentations = [
        {"name": "TimeMasking", "params": {"time_mask_param": 30, "p": 0.35}},
        {"name": "FrequencyMasking", "params": {"freq_mask_param": 15}},
    ]

    augmenter = SpectrogramAugment(augmentations)
    X_aug_list = augmenter([X])
    X_aug = X_aug_list[0]

    spec_orig = X[0].cpu()
    spec_aug = X_aug[0].cpu()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axs[0].imshow(spec_orig, origin="lower", aspect="auto")
    axs[0].set_title("Original")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(spec_aug, origin="lower", aspect="auto")
    axs[1].set_title("Augmented")
    fig.colorbar(im1, ax=axs[1])

    plt.tight_layout()
    out_path = "../raw_augmentations_overview/X_0_spectrogram_augment_2.png"
    fig.savefig(out_path)
