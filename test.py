import torch
from GenericUnet import GenericUnet as GUnet
import numpy as np
import torch.nn as nn
import dataloader
import loss as loss
import transforms as t
import utils
import matplotlib.pyplot as plt
import mask
import skimage.io as io
import os
import pickle

def get_min(image):
    for i in image:
        print(i.min())
    return image
data = dataloader.stack(path='Data/Originals/C2Mar2',
                        joint_transforms=[t.to_float,
                                          t.reshape,
                                          t.random_crop([400, 400, 19]),
                                          # t.random_affine,
                                          # t.random_rotate,
                                          ],
                        image_transforms=[
                                          t.random_gamma(gamma_range=(.5, 1.5)),
                                          t.spekle,
                                          t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                          ],
                        out_transforms=[t.to_tensor]
                        )

test = GUnet(conv_functions=(nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d, nn.BatchNorm3d),
             in_channels=4,
             out_channels=1,
             feature_sizes=[8,16,32,64,128],
             kernel={'conv1': (3, 3, 2), 'conv2': (3, 3, 1)},
             upsample_kernel=(2, 2, 2),
             max_pool_kernel=(2, 2, 1),
             upsample_stride=(2, 2, 1),
             dilation=1,
             groups=1).to('cpu')


# test.save()
# test.load('model.unet')
image, mask, pwl = data[0]

out = test.forward(image.float())
#test.evaluate(image)

# for i in range(19):
#
#     plt.imshow(mask[0,0,:,:,i] * .5 + .5 , cmap="Greys")
#     plt.show()

#test.save()
#test.load('model.unet')

# file = '/Users/chrisbuswinka/Desktop/ToMask/C2-Mar-2-AAV2-PHP.B-CMV-Eric-m7a.lif---m7.labels.tif'
# filename = os.path.splitext(file)[0]
# filename = os.path.splitext(filename)[0]
#
# makemask = LossMasks.makeMask()
# makepwl = LossMasks.makePWL()
#
# colormask = makemask(file)
# io.imsave(filename+'.colormask.tif', colormask)
#
# pwl = makepwl(filename+'.colormask.tif')
# pickle.dump(pwl, open(filename+'pwl.pkl','wb'))
#
# mask = np.copy(colormask)
# bw_mask = LossMasks.colormask_to_mask(colormask)
#
# io.imsave(filename+'.mask.tif', bw_mask)

