from hcat.unet import Unet_Constructor as GUnet
from hcat import mask, utils, rcnn, transforms as t, segment
from hcat.loss import dice, cross_entropy
import hcat
import hcat.dataloader as dataloader

import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

import os
import os.path
import time
import pickle

data = dataloader.Stack(path='../Data/train',
                        joint_transforms=[t.to_float(),
                                          t.reshape(),
                                          t.nul_crop(rate=1),
                                          t.random_crop([128, 128, 24]),
                                          t.elastic_deform(grid_shape=(4, 4, 3), scale=2),

                                          #t.random_affine()
                                          ],
                        image_transforms=[
                                          t.random_gamma((.7, 1.3)),
                                          # t.random_intensity(),
                                          t.drop_channel(.8),
                                          t.spekle(0.00001),
                                          t.clean_image(),
                                          t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                          ]
                        )

val_data = dataloader.Stack(path='../Data/train',
                        joint_transforms=[t.to_float(),
                                          t.reshape(),
                                          t.random_crop([512, 512, 30]),
                                          t.random_rotate(90),
                                          #t.random_affine
                                          ],
                        image_transforms=[
                                          #t.random_gamma((.8,1.2)),
                                          #t.spekle(),
                                          t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                          ]
                        )

image, mask, pwl = data[0]

for i in range(image.shape[-1]):
    plt.figure(figsize=(9, 9))
    # plt.imshow(image[0,[0,2,1],:,:,i].float().transpose(0,1).transpose(1,2)*.5 + .5)
    plt.imshow(image[0, [1, 2, 3], :, :, i].float().transpose(0, 1).transpose(1, 2) * .5 + .5)
    # plt.imshow(mask[0,0,:,:,i].float(), cmap = 'Greens')

    plt.show()
    print(i, end = ' ')