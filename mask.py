import numpy as np

class Part:

    def __init__(self, mask, loc):
        self.shape = mask
        self.is_nul = not mask.sum() > 0
        self.loc = loc  # This should be the top left corner ind
        self.mask = mask

    @property
    def mask(self):
        if self.__mask is not None:
            return self.__mask
        else:
            return np.zeros(self._shape)

    @mask.setter
    def mask(self, mask):
        self._shape = mask.shape
        if np.sum(mask) == 0:
            self.__mask = None
        else:
            self.__mask = mask
