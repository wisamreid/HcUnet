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
                                          t.elastic_deform(grid_shape=(4, 4, 3), scale=5),

                                          #t.random_affine()
                                          ],
                        image_transforms=[
                                          t.random_gamma((.7, 1.3)),
                                          t.random_intensity(range=(-15, 15)),
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

# image, mask, pwl = data[0]
#
# for i in range(image.shape[-1]):
#     plt.figure(figsize=(9, 9))
#     # plt.imshow(image[0,[0,2,1],:,:,i].float().transpose(0,1).transpose(1,2)*.5 + .5)
#     plt.imshow(image[0, [1, 2, 3], :, :, i].float().transpose(0, 1).transpose(1, 2) * .5 + .5)
#     # plt.imshow(mask[0,0,:,:,i].float(), cmap = 'Greens')
#
#     plt.show()
#     print(i, end = ' ')

norm = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
data = dataloader.Section(path='../Data/FasterRCNN_trainData/Top/',
                          simple_class=True,
                          image_transforms=[t.to_float(),
                                            t.random_gamma((.85, 1.15)),
                                            t.random_intensity(range=(-15, 15)),
                                            t.spekle(0.00001),
                                            t.remove_channel(remaining_channel_index=[0, 2, 3]),
                                            t.elastic_deform((30,30), 2),
                                            t.normalize(**norm),
                                            ],
                          joint_transforms=[
                                            t.random_x_flip(),
                                            t.random_y_flip(),
                                            t.add_junk_image(path='../Data/FasterRCNN_junkData/',
                                                             junk_image_size=(100, 100),
                                                             normalize=norm),
                                            t.add_junk_image(path='../Data/FasterRCNN_junkData/',
                                                             junk_image_size=(100, 100),
                                                             normalize=norm)
                                            # t.random_resize(scale=(.3, 4)),
                                            ]
                          )


images, _ = data[10]
faster_rcnn = hcat.rcnn(path='/media/DataStorage/Dropbox (Partners HealthCare)/HcUnet/fasterrcnn_Aug20_13:49.pth')
faster_rcnn.to('cuda')
faster_rcnn.eval()
faster_rcnn.eval()
with torch.no_grad():
    a = faster_rcnn(images.to('cuda').float())

utils.show_box_pred(images.squeeze().float(), a, .3)
plt.show()