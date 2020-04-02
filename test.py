import torch
from models import Unet3D as Unet
from GenericUnet import GenericUnet as GUnet
import numpy as np
import torch.nn as nn
import dataloader
import loss as loss
import transforms as t
import matplotlib.pyplot as plt

def get_min(image):
    for i in image:
        print(i.min())
    return image
data = dataloader.stack(path='./Data',
                        joint_transforms=[t.to_float,
                                          t.reshape,
                                          t.random_crop([300, 300, 19]),
                                          t.random_rotate,
                                          ],
                        image_transforms=[
                                          t.random_gamma,
                                          t.spekle,
                                          t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                          ],
                        out_transforms=[t.to_tensor]
                        )

test = GUnet(conv_functions=(nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d),
             in_channels=4,
             out_channels=2,
             feature_sizes=[16,32,64,128,256,512],
             kernel={'conv1': (3, 3, 2), 'conv2': (3, 3, 1)},
             upsample_kernel=(7, 7, 2),
             max_pool_kernel=(2, 2, 1),
             upsample_stride=(2, 2, 1),
             dilation=1,
             groups=2).to('cpu')

test = test.type(torch.float)

image, mask, pwl = data[0]

#out = test.forward(image.float())

#print(loss.loss(out, mask,pwl))

plt.imshow(image[0,2,:,:,4])
plt.show()
plt.imshow(mask[0,0,:,:,4])
plt.show()
plt.imshow(pwl[0,0,:,:,4])
plt.show()