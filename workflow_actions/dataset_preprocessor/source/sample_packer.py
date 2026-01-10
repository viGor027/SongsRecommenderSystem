import torch
from workflow_actions.paths import MODEL_READY_TRAIN_DIR, MODEL_READY_VALID_DIR
import random


class SamplePacker:
    """
    Packs samples into groups e.g. for group_size==3:
        X_<i>.pt, X_<i+1>.pt, X_<i+2>.pt, y_<i>.pt, y_<i+1>.pt, y_<i+2>.pt
        will be packed together.
    """

    def __init__(self, group_size: int | None, shuffle_before_pack: bool = False):
        """If group_size is None then nothing will happen when running methods."""
        if group_size is not None and group_size < 1:
            raise ValueError("group_size must be >= 1")
        self.group_size = group_size
        self.shuffle_before_pack = shuffle_before_pack

    def pack(self):
        """
        Notes:
         - Intended for offline data preparation only; keep group_size=None for any online/training usage.
         - Expects matching X_<id>.pt and y_<id>.pt files for every packed id.
         - Writes shard_<n>.pt and deletes the X_*/y_* files that the given shard was created from.
         - Does not remove pre-existing shard_*.pt; clean the target dirs or unpack existing shards with SamplePacker.unpack before packing to avoid stale shards.
        """
        if self.group_size is None:
            return

        for folder in [MODEL_READY_TRAIN_DIR, MODEL_READY_VALID_DIR]:
            if not folder.exists():
                raise FileNotFoundError(f"Data directory does not exist: {folder}")

            ids = sorted(int(p.stem.split("_", 1)[1]) for p in folder.glob("X_*.pt"))
            if self.shuffle_before_pack:
                random.shuffle(ids)

            shard_num = 0
            for i in range(0, len(ids), self.group_size):
                chunk = ids[i : i + self.group_size]
                xs = [
                    torch.load(folder / f"X_{j}.pt", map_location="cpu") for j in chunk
                ]
                ys = [
                    torch.load(folder / f"y_{j}.pt", map_location="cpu") for j in chunk
                ]
                X = torch.cat(xs, dim=0)
                y = torch.stack(ys, dim=0)
                out_path = folder / f"shard_{shard_num}.pt"
                torch.save({"X": X, "y": y}, out_path)
                for j in chunk:
                    (folder / f"X_{j}.pt").unlink()
                    (folder / f"y_{j}.pt").unlink()
                shard_num += 1

    def unpack(self):
        """
        Notes:
         - Offline/debug helper: expands shard_*.pt back into X_*/y_* files and deletes shard file given X_*/y_*'s were created from.
         - if shuffle_before_pack was True original ids/order are NOT preserved.
        """
        if self.group_size is None:
            return

        for folder in [MODEL_READY_TRAIN_DIR, MODEL_READY_VALID_DIR]:
            if not folder.exists():
                raise FileNotFoundError(f"Data directory does not exist: {folder}")

            shard_files = sorted(
                folder.glob("shard_*.pt"),
                key=lambda p: int(p.stem.split("_", 1)[1]),
            )

            for shard_path in shard_files:
                sample_idx = int(shard_path.stem.split("_", 1)[1])

                d = torch.load(shard_path, map_location="cpu")
                X = d["X"]
                y = d["y"]

                start_id = sample_idx * self.group_size

                n_in_sample = X.shape[0]
                for offset in range(n_in_sample):
                    sample_id = start_id + offset
                    x_i = X[offset : offset + 1].clone()
                    y_i = y[offset].clone()
                    torch.save(x_i, folder / f"X_{sample_id}.pt")
                    torch.save(y_i, folder / f"y_{sample_id}.pt")

                shard_path.unlink()
