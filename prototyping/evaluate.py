import os
import torch
from torcheval.metrics import MultilabelAccuracy
from torch.nn import BCELoss
from workflow_actions.dataset_preprocessor.source.constants import DATA_DIR
from workflow_actions.json_handlers import write_dict_to_json
import pandas as pd


def remove_redundant_tags(tensor: torch.Tensor):
    reduced = torch.cat((tensor[:, 1:61], tensor[:, 61 + 1 :]), dim=1)
    return reduced


def load_concat_save_valid_set():
    X_valid = torch.load(os.path.join(DATA_DIR, "X_1_valid.pt"))
    Y_valid = torch.load(os.path.join(DATA_DIR, "Y_1_valid.pt"))

    for i in [2, 3]:
        X_valid_temp = torch.load(os.path.join(DATA_DIR, f"X_{i}_valid.pt"))
        Y_valid_temp = torch.load(os.path.join(DATA_DIR, f"Y_{i}_valid.pt"))

        X_valid = torch.concat([X_valid, X_valid_temp], dim=0)
        Y_valid = torch.concat([Y_valid, Y_valid_temp], dim=0)

    torch.save(X_valid, os.path.join(DATA_DIR, "evaluation", "X_valid.pt"))
    torch.save(Y_valid, os.path.join(DATA_DIR, "evaluation", "Y_valid.pt"))


def make_and_save_valid_preds(model):
    X_valid = torch.load(os.path.join(DATA_DIR, "evaluation", "X_valid.pt"))
    Y_valid = torch.load(os.path.join(DATA_DIR, "evaluation", "Y_valid.pt"))

    X_valid, Y_valid = X_valid.float(), Y_valid.float()
    print(X_valid.shape)
    print(Y_valid.shape)

    Y_pred = []
    batch_size = 64

    with torch.no_grad():
        for i in range(batch_size, len(X_valid), batch_size):
            Y_pred.append(model(X_valid[i - batch_size : i]))
        Y_pred.append(model(X_valid[(len(X_valid) // batch_size) * batch_size :]))

    Y_pred = torch.concat(Y_pred, dim=0)
    torch.save(Y_pred, os.path.join(DATA_DIR, "evaluation", "Y_pred_2.pt"))


def evaluate():
    Y_valid = torch.load(os.path.join(DATA_DIR, "evaluation", "Y_valid.pt"))
    Y_pred = torch.load(os.path.join(DATA_DIR, "evaluation", "Y_pred_2.pt"))
    Y_valid, Y_pred = Y_valid.float(), Y_pred.float()

    Y_valid_reduced = remove_redundant_tags(Y_valid)
    Y_pred_reduced = remove_redundant_tags(Y_pred)

    bce_loss = BCELoss()
    hamming_accuracy = MultilabelAccuracy(criteria="hamming")
    exact_match = MultilabelAccuracy(criteria="exact_match")

    hamming_accuracy.update(Y_pred_reduced, Y_valid_reduced)
    exact_match.update(Y_pred_reduced, Y_valid_reduced)

    metrics = {
        "BCELoss": bce_loss(Y_pred_reduced, Y_valid_reduced).item(),
        "HammingAccuracy": hamming_accuracy.compute().item(),
        "ExactMatch": exact_match.compute().item(),
    }

    write_dict_to_json(metrics, os.path.join(DATA_DIR, "evaluation", "metrics.json"))
    print(f"BCELoss: {metrics['BCELoss']}")
    print(f"HammingAccuracy: {metrics['HammingAccuracy']}")
    print(f"ExactMatch: {metrics['ExactMatch']}")


def hamming_dist_at(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = remove_redundant_tags(y_true)
    y_pred = remove_redundant_tags(y_pred)

    diff = (y_true != (y_pred > 0.5).int()).int()
    distance_per_sample = torch.sum(diff, dim=1)
    unique_values, counts = torch.unique(distance_per_sample, return_counts=True)
    stacked = torch.stack([unique_values, counts], dim=1)
    df = pd.DataFrame(
        stacked.numpy(), columns=["number_of_incorrect_tags", "number_of_samples"]
    )
    print("N validation samples: ", y_true.shape[0])
    df["share_of_total"] = df["number_of_samples"] / y_true.shape[0]
    print(df)
    first_group = df.iloc[0, :]
    first_group["number_of_incorrect_tags"] = "0"

    second_group = df.iloc[1 : 6 + 1, :].sum()
    second_group["number_of_incorrect_tags"] = "1 to 6"

    third_group = df.iloc[7 : 10 + 1, :].sum()
    third_group["number_of_incorrect_tags"] = "7 to 10"

    fourth_group = df.iloc[11 : 14 + 1, :].sum()
    fourth_group["number_of_incorrect_tags"] = "11 to 14"

    fifth_group = df.iloc[15:, :].sum()
    fifth_group["number_of_incorrect_tags"] = "15 or more"

    print(df, end="\n\n")
    print(first_group, end="\n\n")
    print(second_group, end="\n\n")
    print(third_group, end="\n\n")
    print(fourth_group, end="\n\n")
    print(fifth_group, end="\n\n")


Y_valid = torch.load(os.path.join(DATA_DIR, "evaluation", "Y_valid.pt"))
Y_pred = torch.load(os.path.join(DATA_DIR, "evaluation", "Y_pred.pt"))
print(Y_pred.shape, Y_valid.shape)

hamming_dist_at(y_true=Y_valid, y_pred=Y_pred)
