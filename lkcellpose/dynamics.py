import numpy as np


def compute_masks(flows, cellprob, cellprob_threshold=0.0,
                  flow_threshold=0.4, min_size=15, niter=200):
    """Compute instance masks from Cellpose flow fields using cellpose dynamics.

    Args:
        flows: (2, H, W) array [Y-flow, X-flow]
        cellprob: (H, W) array, cell probability
        cellprob_threshold: threshold on cellprob for foreground
        flow_threshold: max flow error to keep a mask
        min_size: minimum mask size in pixels
        niter: number of Euler integration steps

    Returns:
        masks: (H, W) integer array, 0=background, 1,2,...=instances
    """
    from cellpose import dynamics
    result = dynamics.compute_masks(
        flows,
        cellprob,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
        niter=niter,
    )
    if isinstance(result, tuple):
        mask = result[0]
    else:
        mask = result
    return mask


def labels_to_flows(labels, device=None):
    """Convert integer label masks to Cellpose flow fields + cell probability.

    Args:
        labels: (H, W) integer array, 0=background, 1,2,...=instance IDs

    Returns:
        flows: (4, H, W) float32 array [Y-flow, X-flow, cellprob, label_mask]
    """
    import torch
    from cellpose import dynamics
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    result = dynamics.labels_to_flows([labels], device=device)
    flows = result[0]
    if isinstance(flows, torch.Tensor):
        flows = flows.cpu().numpy()
    if flows.ndim == 3 and flows.shape[0] == 3:
        cellprob = (labels > 0).astype(np.float32)
        flows = np.concatenate([flows, cellprob[np.newaxis]], axis=0)
    return flows.astype(np.float32)
