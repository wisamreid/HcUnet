from unet import unet_constructor as GUnet
import dataloader as dataloader
import loss
import transforms as t
import skimage.feature

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


data = dataloader.stack(path='./Data/train',
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

device = 'cuda:0'

test = GUnet(image_dimensions=3,
             in_channels=4,
             out_channels=1,
             feature_sizes=[16,32,64,128],
             kernel={'conv1': (3, 3, 2), 'conv2': (3, 3, 1)},
             upsample_kernel=(8, 8, 2),
             max_pool_kernel=(2, 2, 1),
             upsample_stride=(2, 2, 1),
             dilation=1,
             groups=2).to(device)

test.load('May14_chris-MS-7C37_2.unet')
test.cuda()
test.train()
print('Yeet')


image, mask, pwl = data[8]
with torch.no_grad():
    out = test(image.float().cuda())

z = 12
# real = np.copy(mask.float().numpy())
# print(mask.max())
# mask=mask.int()
# print(mask.max())

# for x in range(mask[0,0,:,:,z].shape[0]):
#     for y in range(mask[0, 0, :, :, z].shape[1]):
#         if x == 0 or x == mask[0,0,:,:,z].shape[0]-1:
#             continue
#         if y == 0 or y == mask[0,0,:,:,z].shape[1]-1:
#             continue
#
#         s = mask[0,0,x-1,y-1,z] + mask[0,0,x,y-1,z] + mask[0,0,x+1,y-1,z] + \
#             mask[0, 0,x+1,y,z] + mask[0,0,x+1,y+1,z] + mask[0,0,x,y+1,z]  + \
#             mask[0,0,x-1, y+1,z] + mask[0,0,x-1,y,z]
#         if int(s) == 8:
#             real[0,0,x,y,z] = 0

# real = real[0,0,:,:,z]
real = skimage.feature.canny(mask[0,0,:,:,z].cpu().float().numpy(), sigma=0)
real = np.ma.masked_where(real < 0.9, real).transpose((1,0))
plt.figure(figsize=(20,20))

plt.imshow((image[0,[3,2,0],:,:,z].transpose(2,0).float().cpu().numpy() * .5 + .5) * 1.4)
plt.imshow(real, cmap='Greys')

pred = F.sigmoid(out) > .5
pred = pred[0,0,:,:,z].transpose(1,0).cpu().detach().numpy()
yeet = np.zeros(real.shape)
yeet[0:pred.shape[0], 0:pred.shape[1]] = pred
pred = yeet
pred = np.ma.masked_where(pred < 0.9, pred)
plt.imshow(pred, cmap='Oranges_r', alpha=.5)
plt.savefig('figure_yay.tif')
plt.show()

