from unet import unet_constructor as GUnet
import dataloader as dataloader
import loss
import transforms as t

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
import skimage.exposure
import ray
from unet import unet_constructor as GUnet
from loss import dice_loss, cross_entropy_loss, random_cross_entropy
import torch
import pickle
import os
import ray
import glob
import train_mask_utils

basedir = '/home/chris/Desktop/ColorImages/*.labels.tif'

ray.init()

mm = train_mask_utils.makeMask(erosion=True)
mpwl = train_mask_utils.makePWL()

images = glob.glob(basedir)
print(images)
results = []
#
#    NOTES SO YOU ONLY HAVE TO DO THIS ONCE
#    Please save the amira files as rgb tif's or else it wont work.
#
@ray.remote
def make_mask(image_path):
    image = mm(image_path)
    basename = os.path.splitext(image_path)[0]
    basename = basename
    io.imsave(basename+'.mask.tif', image)
    pwl = mpwl(basename+'.mask.tif')
    print(f'PWL MAX ray: {pwl.max()}')
    io.imsave(basename+'.pwl.tif', pwl)
    image = train_mask_utils.colormask_to_mask(image)
    io.imsave(basename+'.mask.tif', image)
    print(basename, ' DONE')
    return image

for i in images:
    print(i)
    results.append(make_mask.remote(i))

ray.get(results)

