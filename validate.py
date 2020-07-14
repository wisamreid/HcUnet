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


data = dataloader.stack(path='./Data/train',
                        joint_transforms=[t.to_float(),
                                          t.reshape(),
                                          t.random_crop([512, 512, 30]),
                                          t.random_rotate(90),
                                          ],
                        image_transforms=[
                                          t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                          ]
                        )
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


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
unet.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/Jun25_chris-MS-7C37_1.unet')
unet.to(device)
unet.eval()

faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                       progress=True,
                                                       num_classes=5,
                                                       pretrained_backbone=True,
                                                       box_detections_per_img=500)

faster_rcnn.load_state_dict(torch.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/fasterrcnn_Jun26_18:12.pth'))
faster_rcnn.to(device)
faster_rcnn.eval()

for image, mask, pwl in data:

    image_frcnn = image[:, [0, 2, 3], :, :, :]

    predicted_cell_candidate_list = segment.predict_cell_candidates(image_frcnn.float().to(device), model=faster_rcnn,
                                                                    initial_coords=(0, 0))

    predicted_semantic_mask = segment.predict_segmentation_mask(unet, image, device)

    unique_cells_original = segment.generate_unique_segmentation_mask(mask.numpy(),
                                                             predicted_cell_candidate_list, image)


    unique_cells = segment.generate_unique_segmentation_mask(predicted_semantic_mask.numpy(),
                                                             predicted_cell_candidate_list, image)

    # out = utils.construct_instance_mask(unique_cells_original, mask)
    #
    # plt.figure()
    # plt.imshow(out[0,0,:,:,23])
    # plt.show()
    #
    #
    # break


    plt.figure()
    a_all = []
    b_all = []
    for a, b in zip(unique_cells, unique_cells_original):
        # print(a.gfp_stats['mean'], b.gfp_stats['mean'])
        if a.gfp_stats['mean'] == 0 or  b.gfp_stats['mean'] == 0:
            continue
        if np.isnan(a.gfp_stats['mean']) or np.isnan(b.gfp_stats['mean']):
            continue


    #     plt.plot(0, a.gfp_stats['mean'], 'k.', alpha=0.1)
    #     plt.plot(1, b.gfp_stats['mean'], 'k.', alpha=0.1)
    #     plt.plot([0, 1], [a.gfp_stats['mean'], b.gfp_stats['mean']], 'r-', alpha=0.1)
    #
        a_all.append(a.gfp_stats['mean'])
        b_all.append(b.gfp_stats['mean'])

    diff1 = np.array(b_all) - np.array(a_all)

    plt.hist(diff1, bins=30, alpha=0.3)
    plt.axvline(diff1.mean(), c='C0')
    plt.axvline(diff1.mean()-diff1.std(), linestyle='--', c='C0' )
    plt.axvline(diff1.mean()+diff1.std(), linestyle='--', c='C0' )

    b_all = np.array(b_all)
    a_all = np.array(a_all)

    np.random.shuffle(b_all)
    np.random.shuffle(a_all)

    diff2 = np.array(b_all) - np.array(a_all)

    plt.axvline(diff2.mean(), c='C1')
    plt.hist(diff2, bins=30, alpha=0.3)
    plt.axvline(diff2.mean()-diff2.std(), linestyle='--', c='C1' )
    plt.axvline(diff2.mean()+diff2.std(), linestyle='--', c='C1' )
    plt.show()

    #
    # plt.plot(0, np.mean(np.array(a_all)), 'ko')
    # plt.plot(1, np.mean(np.array(b_all)), 'ko')
    # plt.plot([0, 1], [np.mean(np.array(a_all)), np.mean(np.array(b_all))], 'b-')
    #
    #
    # ax = plt.gca()
    # ax.set_xlim(-0.25, 1.25)
    # plt.xticks([0,1], ['Automatic Segmentation', 'Manual Segmentation'])
    # plt.ylabel('Mean GFP Pixel Intensity')
    # print(scipy.stats.ttest_rel(a_all, b_all))
    # plt.title(f'Paired T-Test pval: {scipy.stats.ttest_rel(a_all, b_all)[1]}')
    # plt.show()


    # if len(predicted_cell_candidate_list['scores']) > 0:
    #     plt.figure(figsize=(20,20))
    #     utils.show_box_pred(predicted_semantic_mask[0,:,:,:,5], [predicted_cell_candidate_list], .95)
    #     plt.show()
    #
    #     plt.figure(figsize=(20,20))
    #     utils.show_box_pred(mask[0,:,:,:,5].float(), [predicted_cell_candidate_list], .95)
    #     plt.show()

    # plt.imshow(image[0,0,:,:,20].cpu().float().numpy())
    # plt.show()
    # plt.imshow(image[0,1,:,:,20].cpu().float().numpy())
    # plt.show()
    # plt.imshow(image[0,2,:,:,20].cpu().float().numpy())
    # plt.show()
    # plt.imshow(image[0,3,:,:,20].cpu().float().numpy())
    # plt.show()




