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
    def __init__(self):
        self.image_coords = [] # [x,y,z,w,l,h]
        self.center = [] # [x,y]
        self.mask = [] # numpy array
        self.gfp_stats = {'mean:': [], 'std': [], 'median': []} # green channel flattened array

        self.frequency = []
        self.distance_from_apex = []
        self.unique_id = []






