import torch
from workflow_actions.paths import MODEL_READY_TRAIN_DIR, MODEL_READY_VALID_DIR
import random


class SamplePacker:
    """
    Packs samples into groups e.g. for group_size==3:
        X_<i>.pt, X_<i+1>.pt, X_<i+2>.pt, y_<i>.pt, y_<i+1>.pt, y_<i+2>.pt
        will be packed together.
    """

    def __init__(self, group_size: int, shuffle_before_pack: bool = False):
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        self.group_size = group_size
        self.shuffle_before_pack = shuffle_before_pack

    def pack(self):
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


if __name__ == "__main__":
    sp = SamplePacker(group_size=64, shuffle_before_pack=True)
    sp.pack()
