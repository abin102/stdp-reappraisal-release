"""Shared MedMNIST loader for the conv-STDP / SG-SNN MC-dropout pipeline.

All MedMNIST-2D sets are 28x28 (same as MNIST). RGB sets (blood/path) are
converted to grayscale (channel mean) so the existing 1-channel conv pipeline
applies unchanged. Returns torch TensorDatasets:
  flatten=True  -> images (N, 784) float[0,1]   (for conv-STDP scripts)
  flatten=False -> images (N, 1, 28, 28) float[0,1]  (for sg_snn_eval)
Labels are squeezed to (N,) long.
"""
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset

MEDMNIST_CLASSES = {
    "organamnist": 11, "tissuemnist": 8, "bloodmnist": 8, "pathmnist": 9,
}


def is_medmnist(name: str) -> bool:
    return (name or "").lower() in MEDMNIST_CLASSES


def n_classes(name: str) -> int:
    return MEDMNIST_CLASSES[(name or "").lower()]


def _ensure_npz(ds: str, data_dir: str) -> str:
    path = os.path.join(data_dir, f"{ds}.npz")
    if os.path.exists(path):
        return path
    # auto-download via medmnist into data_dir
    import medmnist
    from medmnist import INFO
    cls = getattr(medmnist, INFO[ds]["python_class"])
    cls(split="train", download=True, root=data_dir)
    if not os.path.exists(path):
        # medmnist may save under its own name; locate it
        alt = os.path.join(data_dir, INFO[ds]["python_class"].lower() + ".npz")
        if os.path.exists(alt):
            return alt
    return path


def load_medmnist(ds: str, flatten: bool = True, data_dir: str = "./data", rgb: bool = False):
    """rgb=False -> 1-channel (RGB sets are channel-averaged to grayscale).
    rgb=True   -> keep native 3 channels for RGB sets (blood/path); grayscale
                  sets stay 1-channel."""
    ds = ds.lower()
    path = _ensure_npz(ds, data_dir)
    d = np.load(path)

    def prep(imgs):
        x = torch.from_numpy(imgs.astype(np.float32) / 255.0)
        if x.ndim == 4:                       # (N,28,28,3) RGB source
            if rgb:
                x = x.permute(0, 3, 1, 2).contiguous()      # (N,3,28,28)
            else:
                x = x.mean(dim=3).unsqueeze(1)              # (N,1,28,28) grayscale
        else:                                 # (N,28,28) grayscale source
            x = x.unsqueeze(1)                              # (N,1,28,28)
        return x.reshape(x.shape[0], -1) if flatten else x

    Xtr, Xte = prep(d["train_images"]), prep(d["test_images"])
    ytr = torch.from_numpy(d["train_labels"].squeeze().astype(np.int64))
    yte = torch.from_numpy(d["test_labels"].squeeze().astype(np.int64))
    return TensorDataset(Xtr, ytr), TensorDataset(Xte, yte)
