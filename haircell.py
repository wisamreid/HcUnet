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
        self.mask = np.round(mask) # numpy array
        self.unique_mask = []
        # self.frequency = []
        self.distance_from_apex = []
        self.unique_id = id
        self.is_bad = False

        # if self.mask.sum() > 1:
        #     self.watershed()
        #     self.gfp_stats = self._calculate_gfp_statistics(image, self.unique_mask) # green channel flattened array
        # else:
        #     self.unique_mask = torch.zeros(10)
        #     self.is_bad = True
        #     self.gfp_stats  = {'mean': np.NaN, 'std': np.NaN, 'median': np.NaN}

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


    def _calculate_gfp_statistics(self, image, mask):
        """
        calculates mean, std, and median gfp intensity for the cell and stores the values in a dict.

        :param image: numpy image
        :param mask:  numpy array of same size of image, type bool, used as index's for the image
        :return: dict{'mean', 'median', 'std'}
        """
        mask = torch.tensor(mask > 0)
        gfp = (image[0, 1, :, :, :][mask].float() * 0.5) + 0.5
        # 0:DAPI
        # 1:GFP
        # 2:MYO7a
        # 3:Actin

        if int(gfp.shape[0]) < 1:
            gfp = torch.zeros(10)
        # print(image[0,1,:,:,:].float().mean()*0.5 + 0.5, gfp.mean())
        return {'mean': gfp.mean(), 'std': gfp.std(), 'median': gfp.median(), 'num_samples': gfp.shape}

    # def watershed(self):
    #     self.seed = np.zeros(self.mask.shape)
    #     try:
    #         self.seed[0, 0, self.center[0]-1, self.center[1]-1, self.center[2]-1] = 1
    #     except IndexError:
    #         print(self.seed.shape, self.center)
    #
    #     self.seed = scipy.ndimage.label(self.seed)[0]
    #
    #     distance = np.zeros(self.mask.shape)
    #
    #     for i in range(self.mask.shape[-1]):
    #         distance[0,0,:,:,i] = cv2.distanceTransform(self.mask[0, 0, :, :, i].astype(np.uint8), cv2.DIST_L2, 5)
    #
    #     labels = skimage.segmentation.watershed(-1*self.mask[0,0,:,:,:], self.seed[0,0,:,:,:], mask=self.mask[0,0,:,:,:])
    #     self.unique_mask = labels
    #
    #     # plt.imshow(labels[:,:,17])
    #     # plt.show()
    #     # raise ValueError
    #     return None















