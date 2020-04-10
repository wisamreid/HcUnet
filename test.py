import torch
from GenericUnet import GenericUnet as GUnet
import numpy as np
import torch.nn as nn
import dataloader
import loss as loss
import transforms as t
import matplotlib.pyplot as plt
import LossMasks
import skimage.io as io
import os
import pickle
#
# def get_min(image):
#     for i in image:
#         print(i.min())
#     return image
# data = dataloader.stack(path='./Data',
#                         joint_transforms=[t.to_float,
#                                           t.reshape,
#                                           #t.random_crop([250, 250, 19]),
#                                           t.random_rotate,
#                                           ],
#                         image_transforms=[
#                                           #t.random_gamma,
#                                           #t.spekle,
#                                           #t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
#                                           ],
#                         out_transforms=[t.to_tensor]
#                         )
#
# test = GUnet(conv_functions=(nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d),
#              in_channels=4,
#              out_channels=2,
#              feature_sizes=[128,256,512],
#              kernel={'conv1': (3, 3, 2), 'conv2': (3, 3, 1)},
#              upsample_kernel=(3, 3, 2),
#              max_pool_kernel=(2, 2, 1),
#              upsample_stride=(2, 2, 1),
#              dilation=1,
#              groups=2).to('cpu')
#
# test = test.type(torch.float)
#
# image, mask, pwl = data[0]
#
# for i in range(19):
#     plt.imshow(mask[0,0,:,:,i], cmap="Greys")
#     plt.show()
#
# test.save()
# test.load('model.unet')

file = '/Users/chrisbuswinka/Desktop/ToMask/C2-Mar-2-AAV2-PHP.B-CMV-Eric-m7a.lif---m7.labels.tif'
filename = os.path.splitext(file)[0]
filename = os.path.splitext(filename)[0]

makemask = LossMasks.makeMask()
makepwl = LossMasks.makePWL()

colormask = makemask(file)
io.imsave(filename+'.colormask.tif', colormask)

pwl = makepwl(filename+'.colormask.tif')
pickle.dump(pwl, open(filename+'pwl.pkl','wb'))

mask = np.copy(colormask)
bw_mask = LossMasks.colormask_to_mask(colormask)

io.imsave(filename+'.mask.tif', bw_mask)


