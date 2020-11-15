import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

def render_size(unique_mask: np.ndarray) -> None:

    image = np.zeros(unique_mask.shape, dtype=np.uint8)

    cells = np.unique(unique_mask)

    for u in cells:
        if u == 0:
            continue
        num = np.sum(unique_mask==u)
        if num < 5000:
            image[unique_mask == u] = 2
        elif num > 15000:
            image[unique_mask == u] = 3
        else:
            image[unique_mask == u] = 1

    skimage.io.imsave('size_validation.tif', image[0, 0, :, :, :].transpose((2, 1, 0)))
    return None