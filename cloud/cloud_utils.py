import os
import torch
from google.cloud import storage
import json
from song_pipeline.constants import PROJECT_FOLDER_DIR
from io import BytesIO


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(PROJECT_FOLDER_DIR, 'cloud', 'key.json')

client = storage.Client()


def save_dict_to_gcs_as_json(data_dict, bucket_name, folder_name, file_name):
    """
    Saves a dictionary as a JSON file to a specified folder in the given GCS bucket.

    Args:
        data_dict (dict): The dictionary to save as a JSON file.
        bucket_name (str): The name of the GCS bucket (e.g., `my_bucket`).
        folder_name (str): The folder name where the JSON file will be stored.
        file_name (str): The name of the file (without extension).

    Returns:
        bool: True if upload is successful, False otherwise.
    """
    try:
        json_data = json.dumps(data_dict)

        client = storage.Client()

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{folder_name}/{file_name}.json")

        buffer = BytesIO(json_data.encode('utf-8'))

        blob.upload_from_file(buffer, content_type="application/json")

        print(f"JSON data saved to {folder_name}/{file_name}.json in {bucket_name}")
        return True
    except Exception as e:
        print(f"Error saving JSON data: {e}")
        return False


def save_tensor_to_gcs(tensor, bucket_name, folder_name, tensor_name):
    """
    Saves a PyTorch tensor to a specified folder in the `data_versions` bucket.

    Args:
        tensor (torch.Tensor): The tensor to save.
        bucket_name (str): The name of the bucket (e.g., `data_versions`).
        folder_name (str): The folder name where the tensor will be stored.
        tensor_name (str): The name of the tensor file **(without extension)**.

    Returns:
        bool: True if upload is successful, False otherwise.
    """
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{folder_name}/{tensor_name}.pt")

        buffer = BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type="application/octet-stream")

        print(f"Tensor {tensor_name} saved to {folder_name} in {bucket_name}")
        return True
    except Exception as e:
        print(f"Error saving tensor: {e}")
        return False


def load_tensor_from_gcs(bucket_name, folder_name, tensor_name):
    """
    Loads a PyTorch tensor from the `data_versions` bucket.

    Args:
        bucket_name (str): The name of the bucket (e.g., `data_versions`).
        folder_name (str): The folder name where the tensor is stored.
        tensor_name (str): The name of the tensor file **(without extension)**.

    Returns:
        torch.Tensor: The loaded tensor.
    """
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{folder_name}/{tensor_name}.pt")

        tensor_data = blob.download_as_bytes()
        buffer = BytesIO(tensor_data)
        tensor = torch.load(buffer)

        print(f"Tensor {tensor_name} loaded from {folder_name} in {bucket_name}")
        return tensor
    except Exception as e:
        print(f"Error loading tensor: {e}")
        return None


def save_model_to_gcs(model, bucket_name, folder_name, model_name):
    """
    Saves a PyTorch model to a specified folder in the `srs_models` bucket.

    Args:
        model (torch.nn.Module): The model to save.
        bucket_name (str): The name of the bucket (e.g., `srs_models`).
        folder_name (str): The folder name where the model will be stored.
        model_name (str): The name of the model file **(without extension)**.

    Returns:
        bool: True if upload is successful, False otherwise.
    """
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{folder_name}/{model_name}.pt")

        buffer = BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type="application/octet-stream")

        print(f"Model {model_name} saved to {folder_name} in {bucket_name}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def load_model_from_gcs(bucket_name, folder_name, model_name):
    """
    Loads a PyTorch model from the `srs_models` bucket.

    Args:
        bucket_name (str): The name of the bucket (e.g., `srs_models`).
        folder_name (str): The folder name where the model is stored.
        model_name (str): The name of the model file **(without extension)**.

    Returns:
        torch.nn.Module: The loaded model.
    """
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{folder_name}/{model_name}.pt")

        model_data = blob.download_as_bytes()
        buffer = BytesIO(model_data)
        model = torch.load(buffer)

        print(f"Model {model_name} loaded from {folder_name} in {bucket_name}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def save_checkpoint_to_gcs(checkpoint, bucket_name, folder_name, checkpoint_name):
    """
    Saves a PyTorch Lightning checkpoint to a specified folder in a GCS bucket.

    Args:
        checkpoint (dict): The checkpoint dictionary to save.
        bucket_name (str): The name of the GCS bucket.
        folder_name (str): The folder name where the checkpoint will be stored.
        checkpoint_name (str): The name of the checkpoint file (without extension).

    Returns:
        bool: True if the upload is successful, False otherwise.
    """
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{folder_name}/{checkpoint_name}.ckpt")

        buffer = BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type="application/octet-stream")

        print(f"Checkpoint {checkpoint_name} saved to {folder_name} in {bucket_name}")
        return True
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False


def load_checkpoint_from_gcs(bucket_name, folder_name, checkpoint_name):
    """
    Loads a PyTorch Lightning checkpoint from a specified folder in a GCS bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        folder_name (str): The folder name where the checkpoint is stored.
        checkpoint_name (str): The name of the checkpoint file (without extension).

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"{folder_name}/{checkpoint_name}.ckpt")

        checkpoint_data = blob.download_as_bytes()
        buffer = BytesIO(checkpoint_data)
        checkpoint = torch.load(buffer)

        print(f"Checkpoint {checkpoint_name} loaded from {folder_name} in {bucket_name}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def load_checkpoint_correctly(model, checkpoint):
    """Fixes wrong layer names"""
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('model.', '')
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    print("Model weights loaded successfully with corrected key names.")
    return model


if __name__ == "__main__":
    from song_pipeline.constants import DATA_DIR
    Y = torch.load(os.path.join(DATA_DIR, 'min_dataset', 'Y_min.pt'))
    print(Y.shape, Y.dtype)
    save_tensor_to_gcs(Y, 'data_versions', 'test', 'Y')
    y_load = load_tensor_from_gcs('data_versions', 'test', 'Y')
    print("LOADED")
    print(y_load.shape, y_load.dtype)
