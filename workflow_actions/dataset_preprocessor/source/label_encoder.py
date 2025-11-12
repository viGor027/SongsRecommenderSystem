from workflow_actions.paths import LABEL_MAPPING_PATH, LABELS_PATH
from workflow_actions.json_handlers import read_json_to_dict, write_dict_to_json
import torch
from torch.nn.functional import one_hot
from collections import Counter


class LabelEncoder:
    def __init__(self):
        if not LABEL_MAPPING_PATH.exists():
            raise FileExistsError(
                (
                    "Labels file label_mapping.json doesn't exist. "
                    "Call label_encoder.create_label_mapping to create it or"
                    " put existing label mapping to 01_raw/labels"
                )
            )
        if not LABELS_PATH.exists():
            raise FileExistsError("Labels file labels.json doesn't exist.")

        self.song_to_labels = read_json_to_dict(LABELS_PATH)
        self.label_to_int = read_json_to_dict(LABEL_MAPPING_PATH)

        self.n_classes = len(self.label_to_int)

        self.EXCLUDED_TAGS = ["N/A"]

    def create_label_mapping(self) -> dict[int, str]:
        """Creates mapping tag -> number for every tag present in labels.json"""
        if not LABELS_PATH.exists():
            raise FileExistsError("Labels file labels.json doesn't exist.")

        all_tags = []
        for tags_list in self.song_to_labels.values():
            tags_list = [tag for tag in tags_list if tag not in self.EXCLUDED_TAGS]
            all_tags.extend(tags_list)
        tags = Counter(all_tags).keys()
        mapping = {label: idx for idx, label in enumerate(sorted(list(tags)))}
        write_dict_to_json(data=mapping, file_path=LABEL_MAPPING_PATH)
        return mapping

    def encode_song_labels_to_multi_hot_vector(self, song_title: str) -> torch.Tensor:
        """
        Encodes songs tags to multi-hot vector.
        **Pass song title without extension.**
        """
        song_int_tags = [
            self.label_to_int[label]
            for label in self.song_to_labels.get(song_title, [])
            if (label not in self.EXCLUDED_TAGS and label in self.label_to_int.keys())
        ]
        one_hot_tags = one_hot(torch.tensor(song_int_tags), num_classes=self.n_classes)
        multi_hot_tags = one_hot_tags.sum(dim=0).clamp(max=1).float()
        return multi_hot_tags
