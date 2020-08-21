from hcat.unet import Unet_Constructor as GUnet
import transforms as t
from hcat import mask as m, utils, segment
import skimage.io as io
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import ray
import pickle
import time
import logging
from torchvision import models

# path = '/home/chris/Dropbox (Partners HealthCare)/HcUnet/Data/Feb 6 AAV2-PHP.B PSCC m1.lif - PSCC m1 Merged-test_part.tif'
path = '/media/chris/Padlock_3/ToAnalyze/Mar 6 AAV2-PHP.B-CMV11 m3.lif - m3 Stat Merged.tif'
ray.init(logging_level=logging.CRITICAL)



def analyze(path=path):

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

    faster_rcnn.load_state_dict(torch.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/fasterrcnn_Jul16_13:30.pth'))
    faster_rcnn.to(device)
    faster_rcnn.eval()
    print('Done')
    print(f'Starting Analysis...')

    num_chunks = 4

    y_ind = np.linspace(0, image.shape[1], num_chunks).astype(np.int16)
    x_ind = np.linspace(0, image.shape[2], num_chunks).astype(np.int16)

    base = './maskfiles/'
    newfolder = time.strftime('%y%m%d%H%M')
    os.mkdir(base+newfolder)
    all_cells = []

    for i, y in enumerate(y_ind):
        if i == 0: continue
        for j, x in enumerate(x_ind):
            if j == 0: continue

            image_slice = image[:, y_ind[i-1]:y, x_ind[j-1]:x, :]

            for tr in transforms:
                image_slice = tr(image_slice)

            image_slice_frcnn = image_slice[:,[0,2,3],:,:,:]

            print(f'\tGenerating list of cell candidates for chunk [{x_ind[j-1]}:{x}, {y_ind[i-1]}:{y}]: ', end='')
            predicted_cell_candidate_list = segment.predict_cell_candidates(image_slice_frcnn.float().to(device), model=faster_rcnn, initial_coords=(x_ind[j - 1], y_ind[i - 1]))
            print(f'Done [Predicted {len(predicted_cell_candidate_list["scores"])} cells]')

            print(f'\tPredicting segmentation mask for [{x_ind[j-1]}:{x}, {y_ind[i-1]}:{y}]:',end=' ')
            predicted_semantic_mask = segment.predict_segmentation_mask(unet, image_slice, device, use_probability_map=False)
            print('Done')

            print(f'\tAssigning cell labels for [{x_ind[j-1]}:{x}, {y_ind[i-1]}:{y}]:', end=' ')
            unique_mask, seed = segment.generate_unique_segmentation_mask_from_probability(predicted_semantic_mask.numpy(), predicted_cell_candidate_list, image_slice,
                                                                                           cell_prob_threshold=.5)
            print('Done')

            print(f'\tAssigning cell objects:', end=' ')
            cell_list = segment.generate_cell_objects(image_slice, unique_mask)
            all_cells = all_cells + cell_list
            print('Done')

            print(f'\tDisplaying sample z slices: ',end='')
            if len(predicted_cell_candidate_list['scores']) > 0:
                plt.figure(figsize=(20,20))
                utils.show_box_pred(predicted_semantic_mask[0, :, :, :, 5], [predicted_cell_candidate_list], .5)
                plt.savefig(f'chunk{i}_{j}.tif')
                plt.show()

            plt.figure(figsize=(20,20))
            plt.imshow(unique_mask[0,0,:,:,8])
            plt.show()

            plt.figure(figsize=(20,20))
            plt.imshow(predicted_semantic_mask.numpy()[0,0,:,:,8],)
            plt.show()
            print('Done')

            print(f'\tSaving chunk images: ',end='')
            io.imsave(f'unique_mask_{i}_{j}.tif', unique_mask[0,0,:,:,:].transpose((2, 1, 0)))
            io.imsave(f'predicted_prob_map_{i}_{j}.tif', predicted_semantic_mask.numpy()[0,0,:,:,:].transpose((2, 1, 0)))

            a = m.Part(predicted_semantic_mask.numpy(), torch.tensor([]), (x_ind[j-1], y_ind[i-1]))
            pickle.dump(a, open(base+newfolder+'/'+time.strftime("%y_%m_%d_%H_%M_") + str(time.monotonic_ns())+'.maskpart','wb'))
            a = a.mask.astype(np.uint8)[0,0,:,:,:].transpose(2,1,0)
            print('done')

    print('Reconstructing Mask...', end='')
    mask = utils.reconstruct_mask('./maskfiles/' + newfolder)
    print('Done!')

    print('Saving Mask Image...', end='')
    io.imsave('test_mask.tif', mask[0,0,:,:,:].transpose((2, 1, 0)))
    print('Done')

    print('Generating Graphs: ', end='')
    pickle.dump(all_cells, open('all_cells.pkl', 'wb'))

    gfp = []
    myo = []
    dapi = []
    actin = []
    for cell in all_cells:
        if not np.isnan(cell.gfp_stats['mean']):
            gfp.append(cell.gfp_stats['mean'])
            myo.append(cell.signal_stats['myo7a']['mean'])
            dapi.append(cell.signal_stats['dapi']['mean'])
            actin.append(cell.signal_stats['actin']['mean'])

    gfp = np.array(gfp).flatten()
    myo = np.array(myo).flatten()
    dapi = np.array(dapi).flatten()
    actin = np.array(actin).flatten()

    plt.figure()
    plt.hist(gfp, bins=50)
    plt.axvline(gfp.mean(), c='red', linestyle='-')
    plt.xlabel('GFP Intensity')
    plt.ylabel('Occurrence (cells)')
    plt.title(path, fontdict={'fontsize': 8})
    plt.savefig('hist0_gfp.png')
    plt.show()

    plt.figure()
    plt.hist(gfp, color='green', bins=50, alpha=0.6)
    plt.hist(myo, color='yellow', bins=50, alpha=0.6)
    plt.hist(dapi, color='blue', bins=50, alpha=0.6)
    plt.hist(actin, color='red', bins=50, alpha=0.6)
    plt.axvline(gfp.mean(), c='green', linestyle='-')
    plt.axvline(myo.mean(), c='yellow', linestyle='-')
    plt.axvline(dapi.mean(), c='blue', linestyle='-')
    plt.axvline(actin.mean(), c='red', linestyle='-')
    plt.xlabel('Signal Intensity')
    plt.ylabel('Occurrence (cells)')
    plt.title(path, fontdict={'fontsize': 8})
    plt.savefig('hist0_all_colors.png')
    plt.show()

    mask = mask[0,0,:,:,:].transpose((2, 1, 0))
    gfp = image[mask>0]
    gfp = np.array(gfp[:, 1]) / 2**16

    plt.figure()
    plt.hist(gfp, bins=100, range=[0.00000001, 1])
    plt.axvline(gfp.mean(),c='r', linestyle='-')
    plt.xlabel('GFP Intensity (excludes 0)')
    plt.ylabel('Occurrence (px)')
    plt.title(path, fontdict={'fontsize': 8})
    plt.savefig('hist1.png')
    plt.show()

    plt.figure()
    plt.hist(gfp, bins=100, range=[0, 1])
    plt.axvline(gfp.mean(),c='r', linestyle='-')
    plt.xlabel('GFP Intensity')
    plt.ylabel('Occurrence (px)')
    plt.title(path, fontdict={'fontsize': 8})
    plt.savefig('hist2.png')
    plt.show()

    print('Done')

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








