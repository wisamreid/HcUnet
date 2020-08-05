from unet import unet_constructor as GUnet
import dataloader as dataloader
import loss
import transforms as t
import mask
import os
import segment
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
import skimage.exposure
import skimage.filters
from scipy import interpolate
from skimage.morphology import skeletonize
import scipy.ndimage
import ray
import pickle
import time
import scipy.stats
from torchvision import datasets, models
import glob
import re
import pymc3 as pm


path = '/media/chris/Padlock_3/ToAnalyze/'

folders = glob.glob(path+'*/')

image_name = []
gfp_list = []
myo_list = []
for f in folders:
    print(f)
# 8 and 10 for cmv8 m3
# 9, 11 for cmv8 m4

folders=[folders[8],folders[10]]
keep_mask = False
for f in folders:

    plt.figure()
    plt.yscale('log')
    plt.xlabel('GFP Intensity')
    plt.ylabel('Num Pixels (Log)')

    name = os.path.basename(f[0:-1:1])
    promoter = re.search('(Control)|(CMV\d?\d?)', name)[0]
    animal = re.search('m\d',name)[0]
    gain = re.search('Gain\d?\d?\d?',name)

    if re.search('Eric', name) is not None:
        promoter = promoter + ' Full'

    if gain is not None:
        gain = gain[0]
    else:
        gain = ''
    id = promoter + ' ' + animal + ' ' + gain
    print(f'{id}', end=' | ')
    image_name.append(id)

    print(f'Loading Image...', end='')
    n =len('Loading Image...')
    image = io.imread(f[0:-1:1] + '.tif')[:,:,:,1] / 2**16
    print('\b \b'*n, end='')

    print('Loading Mask...', end='')
    mask = io.imread(os.path.dirname(f) + '/test_mask.tif')
    mask = mask > 0

    # YEEEET
    if not keep_mask:
        keep_mask = np.copy(mask)


    n =len('Loading Mask...')
    print('\b'*n, end='')
    print(' '*n, end='')
    print('\b'*n, end='')

    print('Indexing...',end='')
    gfp = image[keep_mask]
    gfp_list.append(gfp)
    pickle.dump(gfp, open(f'{id}.pkl', 'wb'))
    plt.hist(gfp, bins=200, alpha=.5, range=[0,1])
    del gfp
    n =len('Indexing...')
    print('\b'*n, end='')
    print(' '*n, end='')
    print('\b'*n, end='')

    # print('Smoothing Image...', end='')
    # smooth_image = scipy.ndimage.gaussian_filter(image, 3)
    # gfp_smooth = image[mask]
    # plt.hist(gfp_smooth, bins=200, alpha=.5, range=[0,1])
    # del gfp_smooth
    # n =len('Smoothing Image...')
    # print('\b'*n, end='')
    # print(' '*n, end='')
    # print('\b'*n, end='')
    # smooth_image=0

    print('Dilating Mask...',end='')
    gfp_dilated = image[scipy.ndimage.binary_dilation(mask, iterations=5)]
    plt.hist(gfp_dilated, bins=200, alpha=.5, range=[0,1])
    del gfp_dilated
    n =len('Dilating Mask...')
    print('\b'*n, end='')
    print(' '*n, end='')
    print('\b'*n, end='')

    print('Eroding Mask...',end='')
    gfp_eroded = image[scipy.ndimage.binary_erosion(mask, iterations=5)]
    plt.hist(gfp_eroded, bins=200, alpha=.5, range=[0,1])
    del gfp_eroded
    n =len('Eroding Mask...')
    print('\b'*n, end='')
    print(' '*n, end='')
    print('\b'*n, end='')

    plt.title(id)
    plt.legend(['GFP','GFP Dilated', 'GFP Eroded'])
    plt.savefig(f'{id}.png')
    plt.show()

    del image
    del mask
    # del gfp
    print('Done')

plt.figure()
plt.boxplot(gfp_list)
ax = plt.gca()
plt.xticks(np.arange(1, len(gfp_list)+1, 1), image_name, rotation=45)
plt.title('GFP (with zeros)')
plt.show()


