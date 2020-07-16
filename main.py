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
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
import skimage.filters
from scipy import interpolate
from skimage.morphology import skeletonize
import scipy.ndimage
import ray
import pickle
import time
from torchvision import datasets, models, transforms

path = '/home/chris/Dropbox (Partners HealthCare)/HcUnet/Data/Jan 27 AAV2-PHPB m5.lif - TileScan m5 Merged.tif'
ray.init()

transforms = [
              t.to_float(),
              t.reshape(),
              t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
              t.to_tensor(),
              ]

print('Loading Image:  ',end='')
image = io.imread(path)
print('Done')

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

print('Initalizing Unet:  ',end='')
unet= GUnet(image_dimensions=3,
            in_channels=4,
            out_channels=1,
            feature_sizes=[16,32,64,128],
            kernel={'conv1': (3, 3, 2), 'conv2': (3, 3, 1)},
            upsample_kernel=(8, 8, 2),
            max_pool_kernel=(2, 2, 1),
            upsample_stride=(2, 2, 1),
            dilation=1,
            groups=2).to(device)

# unet.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/Jun7_chris-MS-7C37_1.unet')
unet.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/Jun25_chris-MS-7C37_1.unet')
test_image_path = 'Feb 6 AAV2-PHP.B PSCC m1.lif - PSCC m1 Merged.tif'
unet.to(device)
unet.eval()
print('Done')

print('Initalizing FasterRCNN:  ', end='')
faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                       progress=True,
                                                       num_classes=5,
                                                       pretrained_backbone=True,
                                                       box_detections_per_img=500)

faster_rcnn.load_state_dict(torch.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/fasterrcnn_Jul15_15:36.pth'))
faster_rcnn.to(device)
faster_rcnn.eval()
print('Done')
print(f'Starting Analysis: \n')

num_chunks = 3

y_ind = np.linspace(0, image.shape[1], num_chunks).astype(np.int16)
x_ind = np.linspace(0, image.shape[2], num_chunks).astype(np.int16)

base = './maskfiles/'
newfolder = time.strftime('%y%m%d%H%M')
os.mkdir(base+newfolder)

for i, y in enumerate(y_ind):
    if i == 0: continue
    for j, x in enumerate(x_ind):
        if j == 0: continue

        # We take the chunk from the original image.
        image_slice = image[:, y_ind[i-1]:y, x_ind[j-1]:x, :]

        # Apply only necessary transforms needed to turn it into a suitable image for pytorch.
        for tr in transforms:
            image_slice = tr(image_slice)

        # Convert to a 3 channel image for faster rcnn.
        image_slice_frcnn = image_slice[:,[0,2,3],:,:,:]

        # We want this to generate a list of all the cells in the chunk.
        # These cells will have centers that can be filled in with watershed later.
        print(f'Generating list of cell candidates for chunk [{x_ind[j-1]}:{x} , {y_ind[i-1]}:{y}]')
        predicted_cell_candidate_list = segment.predict_cell_candidates(image_slice_frcnn.float().to(device), model=faster_rcnn, initial_coords=(x_ind[j-1], y_ind[i-1]))
        print(f'Done. Predicted {len(predicted_cell_candidate_list["scores"])} cells.')

        # We now want to predict the semantic segmentation mask for the chunk.
        print(f'Predicting segmentation mask for [{x_ind[j-1]}:{x} , {y_ind[i-1]}:{y}]')
        predicted_semantic_mask = segment.predict_segmentation_mask(unet, image_slice, device, use_probability_map=True)
        print('Finished predicting segmentation mask.')

        # Now take the segmentation mask, and list of cell candidates and uniquely segment the cells.
        print(f'Assigning cell labels for [{x_ind[j-1]}:{x} , {y_ind[i-1]}:{y}]')
        unique_mask, seed = segment.generate_unique_segmentation_mask_from_probability(predicted_semantic_mask.numpy(), predicted_cell_candidate_list, image_slice, rejection_probability_threshold=.8)
        print('Finished assigning cell labels.\n')

        if len(predicted_cell_candidate_list['scores']) > 0:
            plt.figure(figsize=(20,20))
            utils.show_box_pred(predicted_semantic_mask[0,:,:,:,5], [predicted_cell_candidate_list], .8)
            plt.savefig(f'chunk{i}_{j}.tif')
            plt.show()

        plt.figure(figsize=(20,20))
        plt.imshow(unique_mask[0,0,:,:,8])
        plt.show()

        plt.figure(figsize=(20,20))
        plt.imshow(predicted_semantic_mask.numpy()[0,0,:,:,8],)
        plt.show()

        io.imsave(f'unique_mask_{i}_{j}.tif', unique_mask[0,0,:,:,:].transpose((2, 1, 0)))
        io.imsave(f'predicted_prob_map_{i}_{j}.tif', predicted_semantic_mask.numpy()[0,0,:,:,:].transpose((2, 1, 0)))


#         a = mask.Part(predicted_semantic_mask.numpy(), torch.tensor([]), (x_ind[j-1], y_ind[i-1]))
#
#         pickle.dump(a, open(base+newfolder+'/'+time.strftime("%y:%m:%d_%H:%M_") + str(time.monotonic_ns())+'.maskpart','wb'))
#         a = a.mask.astype(np.uint8)[0,0,:,:,:].transpose(2,1,0)
#         #
#         #
#         # io.imsave(f'test_unique_cell_x{i}_y{j}.tif', unique_segmentation_mask[0, 0, :, :, :].transpose((2, 1, 0)))
#
# image = 0
# mask = utils.reconstruct_mask('/home/chris/Dropbox (Partners HealthCare)/HcUnet/maskfiles/' + newfolder)
#
# print('Done!')
# print('Saving Image...', end='')
# io.imsave('test_mask.tif', mask[0,0,:,:,:].transpose((2, 1, 0)))
# print('Done!')

# mask = utils.reconstruct_segmented('/home/chris/Dropbox (Partners HealthCare)/HcUnet/maskfiles/' + newfolder)
#
# print('Saving Segment Image...', end='')
# io.imsave('test_segment.tif', mask[0,0,:,:,:].transpose((2, 1, 0)))
# print('Done!')
#
# #
# for i in range(distance.shape[-1]):
#     fig, ax = plt.subplots(1,2)
#     fig.set_size_inches(18.5, 10.5)
#     ax[0].imshow(np.array(distance[0,0,:,:,i]/distance.max(),dtype=np.float))
#     ax[1].imshow(np.array(unique_mask[0,0,:,:,i]/distance.max(),dtype=np.float))
#
#     plt.show()








