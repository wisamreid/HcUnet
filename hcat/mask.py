import numpy as np
import torch


class Part:
    def __init__(self, mask, segmented_mask, loc):
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        self.shape = mask.shape
        self.is_nul = not mask.sum() > 0
        self.loc = np.array(loc)  # This should be the top left corner ind
        self.mask = mask
        self.dtype = mask.dtype
        self.segmented_mask = segmented_mask

        # unique_segmentation_mask.astype(np.uint16)
    @property
    def mask(self):
        if self.__mask is not None:
            return self.__mask
        else:
            return np.zeros(self._shape_mask)

    @mask.setter
    def mask(self, mask):
        self._shape_mask = mask.shape
        if np.sum(mask) == 0:
            self.__mask = None
        else:
            self.__mask = mask

    @property
    def segmented_mask(self):
        if self.__segmented_mask is not None:
            return self.__segmented_mask
        else:
            return np.zeros(self._shape_segmented)

    @segmented_mask.setter
    def segmented_mask(self, mask):
        self._shape_segmented = mask.shape
        if mask.sum() == 0:
            self.__segmented_mask = None
        else:
            self.__segmented_mask = mask
