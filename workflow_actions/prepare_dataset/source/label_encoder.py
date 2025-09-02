from workflow_actions.paths import LABEL_MAPPING_PATH, LABELS_PATH
from workflow_actions.json_handlers import read_json_to_dict
import torch
from torch.nn.functional import one_hot


def encode_song_labels_to_multi_hot_vector(song_title: str) -> torch.Tensor:
    """
    Encodes songs tags to multi-hot vector.
    **Pass song title without extension.**
    """
    if not LABEL_MAPPING_PATH.exists():
        raise FileExistsError("Labels file label_mapping.json doesn't exist.")
    if not LABELS_PATH.exists():
        raise FileExistsError("Labels file labels.json doesn't exist.")
    labels = read_json_to_dict(LABELS_PATH)
    label_to_int = read_json_to_dict(LABEL_MAPPING_PATH)
    n_classes = len(label_to_int)
    song_int_tags = [label_to_int[label] for label in labels[song_title]]
    one_hot_tags = one_hot(torch.tensor(song_int_tags), num_classes=n_classes)
    multi_hot_tags = one_hot_tags.sum(dim=0)
    return multi_hot_tags


if __name__ == "__main__":
    encode_song_labels_to_multi_hot_vector(song_title="Cartoon-Overheat")
