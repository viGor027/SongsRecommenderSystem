import os
import torch
from torcheval.metrics import MultilabelAccuracy
from torch.nn import BCELoss
from cloud.cloud_utils import get_ready_model_from_gcs_checkpoint
from song_pipeline.constants import DATA_DIR
from  song_pipeline.utils import write_dict_to_json


model = get_ready_model_from_gcs_checkpoint(bucket_name='fine_tuned', folder_name='model_1',
                                            checkpoint_name='model', cfg_file_name='cfg')
model.eval()


def load_concat_save_valid_set():
    X_valid = torch.load(os.path.join(DATA_DIR, 'X_1_valid.pt'))
    Y_valid = torch.load(os.path.join(DATA_DIR, 'Y_1_valid.pt'))

    for i in [2, 3]:
        X_valid_temp = torch.load(os.path.join(DATA_DIR, f"X_{i}_valid.pt"))
        Y_valid_temp = torch.load(os.path.join(DATA_DIR, f"Y_{i}_valid.pt"))

        X_valid = torch.concat([X_valid, X_valid_temp], dim=0)
        Y_valid = torch.concat([Y_valid, Y_valid_temp], dim=0)

    torch.save(X_valid, os.path.join(DATA_DIR, 'evaluation', 'X_valid.pt'))
    torch.save(Y_valid, os.path.join(DATA_DIR, 'evaluation', 'Y_valid.pt'))


def make_and_save_valid_preds():
    X_valid = torch.load(os.path.join(DATA_DIR, 'evaluation', 'X_valid.pt'))
    Y_valid = torch.load(os.path.join(DATA_DIR, 'evaluation', 'Y_valid.pt'))

    X_valid, Y_valid = X_valid.float(), Y_valid.float()
    print(X_valid.shape)
    print(Y_valid.shape)

    Y_pred = []
    batch_size = 64

    with torch.no_grad():
        for i in range(batch_size, len(X_valid), batch_size):
            Y_pred.append(model(X_valid[i - batch_size:i]))
        Y_pred.append(model(X_valid[(len(X_valid) // batch_size) * batch_size:]))

    Y_pred = torch.concat(Y_pred, dim=0)
    torch.save(Y_pred, os.path.join(DATA_DIR, 'evaluation', 'Y_pred.pt'))


def evaluate():
    Y_valid = torch.load(os.path.join(DATA_DIR, 'evaluation', 'Y_valid.pt'))
    Y_pred = torch.load(os.path.join(DATA_DIR, 'evaluation', 'Y_pred.pt'))
    Y_valid, Y_pred = Y_valid.float(), Y_pred.float()

    Y_valid_reduced = torch.cat((Y_valid[:, 1:61], Y_valid[:, 61 + 1:]), dim=1)
    Y_pred_reduced = torch.cat((Y_pred[:, 1:61], Y_pred[:, 61 + 1:]), dim=1)

    bce_loss = BCELoss()
    hamming_accuracy = MultilabelAccuracy(criteria="hamming")
    exact_match = MultilabelAccuracy(criteria='exact_match')

    hamming_accuracy.update(Y_pred_reduced, Y_valid_reduced)
    exact_match.update(Y_pred_reduced, Y_valid_reduced)

    metrics = {
        "BCELoss": bce_loss(Y_pred_reduced, Y_valid_reduced).item(),
        "HammingAccuracy": hamming_accuracy.compute().item(),
        "ExactMatch": exact_match.compute().item(),
    }

    write_dict_to_json(metrics, os.path.join(DATA_DIR, 'evaluation', 'metrics.json'))
    print(f"BCELoss: {metrics['BCELoss']}")
    print(f"HammingAccuracy: {metrics['HammingAccuracy']}")
    print(f"ExactMatch: {metrics['ExactMatch']}")


evaluate()
