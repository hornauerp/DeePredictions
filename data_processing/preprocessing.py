import os

import numpy as np


def load_sorting_results(root_path, segment_idx=0):
    """Loads phy-formatted spike sortings

    Args:
        root_path (string): Either path where the sorting is located (if segment_idx is
        empty) or root path for split concatenated sortings
        segment_idx (int, optional): Index of concatenation if applicable. If empty,
        only root path is used. Defaults to 0.

    Returns:
        spike_times (float32): Spike times in (s)
        spike_templates (int64): Unit IDs corresponding to spike_times
    """
    if not len(segment_idx) == 0:
        sorting_path = os.path.join(root_path, "segment" + segment_idx)
    else:
        sorting_path = root_path

    spike_times = (
        np.load(os.path.join(sorting_path, "spike_times.npy")).astype("float32") / 10000
    )
    spike_templates = np.load(os.path.join(sorting_path, "spike_templates.npy"))
    return spike_times, spike_templates
