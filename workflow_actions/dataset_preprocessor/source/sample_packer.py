import torch
from workflow_actions.paths import MODEL_READY_TRAIN_DIR, MODEL_READY_VALID_DIR


class SamplePacker:
    """
    Packs samples into groups e.g. for group_size==3:
        X_<i>.pt, X_<i+1>.pt, X_<i+2>.pt, y_<i>.pt, y_<i+1>.pt, y_<i+2>.pt
        will be packed together.
    """

    def __init__(self, group_size: int):
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        self.group_size = group_size

    def pack(self, *args, **kwargs):
        for folder in [MODEL_READY_TRAIN_DIR, MODEL_READY_VALID_DIR]:
            if not folder.exists():
                raise FileNotFoundError(f"Data directory does not exist: {folder}")

            ids = sorted(int(p.stem.split("_", 1)[1]) for p in folder.glob("X_*.pt"))

            sample_num = 0
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
                out_path = folder / f"sample_{sample_num}.pt"
                torch.save({"ids": chunk, "X": X, "y": y}, out_path)
                for j in chunk:
                    (folder / f"X_{j}.pt").unlink()
                    (folder / f"y_{j}.pt").unlink()
                sample_num += 1
