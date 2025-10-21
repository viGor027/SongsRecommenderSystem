from workflow_actions.paths import LABEL_MAPPING_PATH, LABELS_PATH
from workflow_actions.json_handlers import read_json_to_dict, write_dict_to_json
import torch
from torch.nn.functional import one_hot
from collections import Counter


def create_label_mapping() -> dict[int, str]:
    """Creates mapping tag -> number for every tag present in labels.json"""
    if not LABELS_PATH.exists():
        raise FileExistsError("Labels file labels.json doesn't exist.")

    song_to_labels = read_json_to_dict(LABELS_PATH)
    all_tags = []
    for tags_list in song_to_labels.values():
        all_tags.extend(tags_list)
    tags = Counter(all_tags).keys()

    mapping = {label: idx for idx, label in enumerate(sorted(list(tags)))}
    write_dict_to_json(data=mapping, file_path=LABEL_MAPPING_PATH)
    return mapping


def encode_song_labels_to_multi_hot_vector(song_title: str) -> torch.Tensor:
    """
    Encodes songs tags to multi-hot vector.
    **Pass song title without extension.**
    """
    if not LABEL_MAPPING_PATH.exists():
        raise FileExistsError(
            (
                "Labels file label_mapping.json doesn't exist. "
                "Call label_encoder.create_label_mapping to create it."
            )
        )
    if not LABELS_PATH.exists():
        raise FileExistsError("Labels file labels.json doesn't exist.")
    labels = read_json_to_dict(LABELS_PATH)
    label_to_int = read_json_to_dict(LABEL_MAPPING_PATH)
    n_classes = len(label_to_int)
    song_int_tags = [label_to_int[label] for label in labels[song_title]]
    one_hot_tags = one_hot(torch.tensor(song_int_tags), num_classes=n_classes)
    multi_hot_tags = one_hot_tags.sum(dim=0).float()
    return multi_hot_tags
