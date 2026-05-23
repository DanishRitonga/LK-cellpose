import numpy as np
import torch
from pathlib import Path


def labels_to_flows(labels: np.ndarray, device=None) -> np.ndarray:
    """
    Convert integer label masks to Cellpose flow fields + cell probability.

    Args:
        labels: (H, W) integer array, 0=background, 1,2,...=instance IDs

    Returns:
        flows: (3, H, W) float32 array [Y-flow, X-flow, cellprob]
    """
    from cellpose import dynamics
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    result = dynamics.labels_to_flows([labels], device=device)
    raw = result[0]
    if isinstance(raw, torch.Tensor):
        raw = raw.cpu().numpy()
    raw = raw.astype(np.float32)
    if raw.shape[0] == 4:
        y_flow = raw[2]
        x_flow = raw[3]
        cellprob = raw[1]
    elif raw.shape[0] == 3:
        y_flow = raw[0]
        x_flow = raw[1]
        cellprob = (labels > 0).astype(np.float32)
    else:
        raise ValueError(f"Unexpected flow shape: {raw.shape}")
    flows = np.stack([y_flow, x_flow, cellprob], axis=0)
    return flows


def compute_class_map(labels: np.ndarray, categories: list[int], n_classes: int = 5) -> np.ndarray:
    """
    Convert instance labels + per-instance category list to per-pixel class map.

    Args:
        labels: (H, W) int array, 0=background, 1,2,...=instance IDs
        categories: list of int, category for each instance (0-indexed, len = max(labels))
        n_classes: number of foreground classes

    Returns:
        class_map: (H, W) int8 array, 255=background (ignore), 0..4=nucleus class
    """
    class_map = np.full(labels.shape, 255, dtype=np.int16)
    for inst_id in range(1, len(categories) + 1):
        cat = categories[inst_id - 1]
        if 0 <= cat < n_classes:
            class_map[labels == inst_id] = cat
    return class_map


def normalize_img(img: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Normalize image so that lower percentile -> 0, upper percentile -> 1."""
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    img = img.astype(np.float32)
    for c in range(img.shape[2]):
        lo = np.percentile(img[:, :, c], lower)
        hi = np.percentile(img[:, :, c], upper)
        if hi - lo > 1e-6:
            img[:, :, c] = (img[:, :, c] - lo) / (hi - lo)
        else:
            img[:, :, c] = 0.0
    return np.clip(img, 0.0, 1.0)


def cache_flows(cache_dir: Path, image_idx: int, fold: int, labels: np.ndarray,
                categories: list[int], device=None) -> dict:
    """Compute and cache flows + class map to .npz file.

    Returns dict with keys: flows, class_map
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"fold{fold}_{image_idx:06d}.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        return {"flows": data["flows"], "class_map": data["class_map"]}

    flows = labels_to_flows(labels, device=device)
    class_map = compute_class_map(labels, categories)

    np.savez(cache_path, flows=flows, class_map=class_map)
    return {"flows": flows, "class_map": class_map}
