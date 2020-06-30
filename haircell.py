from unet import unet_constructor as GUnet
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
import pickle
import time

class HairCell:
    def __init__(self, ):
        self.image_coords = [] # [x,y,z,w,l,h]
        self.center = [] # [x,y]
        self.mask = [] # numpy array
        self.gfp_stats = {'mean:': [], 'std': [], 'median': []} # green channel flattened array

        # self.frequency = []
        self.distance_from_apex = []
        self.unique_id = []

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
        return None

    def watershed(self, mask):

        distance = np.zeros(mask.shape)
        for i in range(mask.shape[-1]):
            distance[0,0,:,:,i] = cv2.distanceTransform(mask[0, 0, :, :, i].astype(np.uint8), cv2.DIST_L2, 5)













