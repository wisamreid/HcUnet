import dataloader as dataloader
import loss
import transforms as t
import mask
import os
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
import numpy as np
import matplotlib.pyplot as plt
import pickle
import skimage.io as io
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
import skimage.filters
from scipy import interpolate
from skimage.morphology import skeletonize
import scipy.ndimage
import scipy.stats
import pickle
import time

class HairCell:
    def __init__(self, image_coords, center, image, mask, id):
        self.image_coords = image_coords # [x1, y1, z1, x2, y2, z2]
        self.center = center# [x,y,z] with respect to the slice maybe?
        # self.mask = mask # numpy array
        # self.frequency = []
        self.distance_from_apex = []
        self.unique_id = id
        self.is_bad = False
        self.signal_stats = {}

        for i, channel in enumerate(['dapi', 'gfp', 'myo7a', 'actin']):
            if mask.sum() > 1:
                self.signal_stats[channel] = self._calculate_gfp_statistics(image.float().numpy(), mask, i)  # green channel flattened array
            else:
                self.unique_mask = torch.zeros(10)
                self.is_bad = True
                self.signal_stats[channel] = {'mean': np.NaN, 'std': np.NaN, 'median': np.NaN}

        if mask.sum() > 1:
            self.gfp_stats = self._calculate_gfp_statistics(image.float().numpy(), mask) # green channel flattened array
        else:
            self.unique_mask = torch.zeros(10)
            self.is_bad = True
            self.gfp_stats  = {'mean': np.NaN, 'std': np.NaN, 'median': np.NaN}

    @property
    def frequency(self):
        raise NotImplementedError

    @frequency.setter
    def frequency(self, location, cochlea_curveature):
        """
        Somehow take location of cell, and the spline fit to estimate cochelar frequency.

        :param location:
        :param cochlea_curveature:
        :return:
        """

        raise NotImplementedError

    @staticmethod
    def _calculate_gfp_statistics(image, mask, channel=1):
        """
        calculates mean, std, and median gfp intensity for the cell and stores the values in a dict.

        :param image: numpy image
        :param mask:  numpy array of same size of image, type bool, used as index's for the image
        :return: dict{'mean', 'median', 'std'}
        """

        # channel = 1
        # 0:DAPI
        # 1:GFP
        # 2:MYO7a
        # 3:Actin

        mask = torch.tensor(mask > 0)

        if image.min() < 0:
            gfp = (image[0, channel, :, :, :][mask] * 0.5) + 0.5
        else:
            gfp = image[0, channel, :, :, :][mask].float()

        return {'mean': gfp.mean(), 'std': gfp.std(), 'median': np.median(gfp), 'num_samples': gfp.shape}

















