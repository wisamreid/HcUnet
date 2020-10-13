import hcat
from hcat import utils, transforms as t
import hcat.mask
import skimage.io as io
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import glob
import skimage.filters
import skimage.morphology
import matplotlib.colors
import skimage.exposure
import tifffile
import plotarchive

def analyze(path=None, numchunks=3, save_plots=False, show_plots=False, path_chunk_storage=None):

    if path_chunk_storage is None:
        raise NotADirectoryError(f'Specify a path to chunk storage.')

    if path is None:
        path = '../Data/Feb 6 AAV2-PHP.B PSCC m1.lif - PSCC m1 Merged-test_part.tif'
        # path = '/media/chris/Padlock_3/ToAnalyze/Jul 18 Control m1.lif - TileScan 1 Merged.tif'

    transforms = [
        t.to_float(),
        t.reshape(),
        t.normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
        t.to_tensor(),
    ]

    print('Loading Image:  ', end='', flush=True)
    image = io.imread(path)
    print('Done')

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print('Initalizing Unet:  ', end='', flush=True)
    unet = hcat.unet(image_dimensions=3,
                     in_channels=4,
                     out_channels=1,
                     feature_sizes=[16, 32, 64, 128],
                     kernel={'conv1': (3, 3, 2), 'conv2': (3, 3, 1)},
                     upsample_kernel=(8, 8, 2),
                     max_pool_kernel=(2, 2, 1),
                     upsample_stride=(2, 2, 1),
                     dilation=1,
                     groups=2).to(device)

    # unet.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/TrainedModels/May28_chris-MS-7C37_2.unet')
    unet.load('/media/DataStorage/Dropbox (Partners HealthCare)/HcUnet/Aug21_chris-MS-7C37_1.unet')
    # unet.load('/home/chris/Dropbox (Partners HealthCare)/HcUnet/Sep8_DISTANCE_chris-MS-7C37_2.unet')
    # test_image_path = '/home/chris/Dropbox (Partners HealthCare)/HcUnet/Data/Feb 6 AAV2-PHP.B PSCC m1.lif - PSCC m1 Merged-test_part.tif'
    unet.to(device)
    unet.eval()
    print('Done', flush=True)

    print('Initalizing FasterRCNN:  ', end='', flush=True)
    faster_rcnn = hcat.rcnn(path='/media/DataStorage/Dropbox (Partners HealthCare)/HcUnet/fasterrcnn_Aug20_13:49.pth')
    faster_rcnn.to(device)
    faster_rcnn.eval()
    print('Done', flush=True)
    print(f'Starting Analysis...', flush=True)

    num_chunks = numchunks

    y_ind = np.linspace(0, image.shape[1], num_chunks).astype(np.int16)
    x_ind = np.linspace(0, image.shape[2], num_chunks).astype(np.int16)

    # base = './maskfiles/'
    # newfolder = time.strftime('%y%m%d%H%M')
    # os.mkdir(base + newfolder)

    all_cells = []

    for i, y in enumerate(y_ind):
        if i == 0: continue
        for j, x in enumerate(x_ind):
            if j == 0: continue

            # We take the chunk from the original image.
            image_slice = image[:, y_ind[i - 1]:y, x_ind[j - 1]:x, :]

            # Apply only necessary transforms needed to turn it into a suitable image for pytorch.
            for tr in transforms:
                image_slice = tr(image_slice)

            # Convert to a 3 channel image for faster rcnn.
            image_slice_frcnn = image_slice[:, [0, 2, 3], :, :, :]

            # We want this to generate a list of all the cells in the chunk.
            # These cells will have centers that can be filled in with watershed later.
            print(f'\tGenerating list of cell candidates for chunk [{i}, {j}] [{x_ind[j - 1]}:{x} , {y_ind[i - 1]}:{y}]: ', end='', flush=True)
            if os.path.exists(f'pccl{i}_{j}.pkl'):
                predicted_cell_candidate_list = torch.load(open(f'pccl{i}_{j}.pkl','rb'))
            else:
                predicted_cell_candidate_list = hcat.predict_cell_candidates(image_slice_frcnn.float().to(device),
                                                                             model=faster_rcnn,
                                                                             initial_coords=(x_ind[j - 1], y_ind[i - 1]))
            if not os.path.exists(f'pccl{i}_{j}.pkl'):
                torch.save(predicted_cell_candidate_list, open(f'pccl{i}_{j}.pkl','wb'))

            print(f'Done [Predicted {len(predicted_cell_candidate_list["scores"])} cells]'
                  f'[Max Probability: {predicted_cell_candidate_list["scores"].max()}]', flush=True)

            # We now want to predict the semantic segmentation mask for the chunk.
            print(f'\tPredicting segmentation mask for [{x_ind[j - 1]}:{x} , {y_ind[i - 1]}:{y}]:', end=' ', flush=True)
            if os.path.exists(f'psm{i}_{j}.pkl'):
                predicted_semantic_mask = torch.load(open(f'psm{i}_{j}.pkl', 'rb'))
            else:
                predicted_semantic_mask = hcat.predict_segmentation_mask(unet, image_slice, device, use_probability_map=True)

            if not os.path.exists(f'psm{i}_{j}.pkl'):
                torch.save(predicted_semantic_mask, open(f'psm{i}_{j}.pkl','wb'))

            print(f'Done ', flush=True)


            # Experimental
            predicted_semantic_mask = torch.tensor(skimage.filters.gaussian(predicted_semantic_mask.numpy(), sigma=3))
            predicted_semantic_mask[predicted_semantic_mask < 0.25] = 0
            predicted_semantic_mask = predicted_semantic_mask * 10


            # # Now take the segmentation mask, and list of cell candidates and uniquely segment the cells.
            print(f'\tAssigning cell labels for [{x_ind[j - 1]}:{x} , {y_ind[i - 1]}:{y}]:', end=' ', flush=True)

            unique_mask, seed = hcat.generate_unique_segmentation_mask_from_probability(predicted_semantic_mask.numpy(),
                                                                                        predicted_cell_candidate_list,
                                                                                        image_slice,
                                                                                        cell_prob_threshold=hcat.__cell_prob_threshold__,
                                                                                        mask_prob_threshold=hcat.__mask_prob_threshold__)
            print('Done', flush=True)
            torch.save(unique_mask, open(f'unique_mask{i}_{j}.pkl','wb'))

            print(f'\tAssigning cell objects:', end=' ', flush=True)
            cell_list = hcat.generate_cell_objects(image_slice, unique_mask, cell_candidates=predicted_cell_candidate_list,
                                                   y_ind_chunk=y_ind[i - 1],
                                                   x_ind_chunk=x_ind[j - 1])
            all_cells = all_cells + cell_list
            print('Done', flush=True)

            if show_plots or save_plots:
                if len(predicted_cell_candidate_list['scores']) > 0:
                    plt.figure(figsize=(20, 20))
                    utils.show_box_pred(predicted_semantic_mask[0, :, :, :, 7], [predicted_cell_candidate_list], .3)
                    if save_plots:
                        plt.savefig(f'chunk{i}_{j}.tif')
                    if show_plots:
                        plt.show()
                    plt.close()

                plt.figure(figsize=(20, 20))
                plt.imshow(unique_mask[0, 0, :, :, 8])
                if show_plots:
                    plt.show()
                plt.close()

                plt.figure(figsize=(20, 20))
                plt.imshow(predicted_semantic_mask.numpy()[0, 0, :, :, 8]**2 )
                if show_plots:
                    plt.show()
                plt.close()

            if save_plots:
                io.imsave(f'unique_mask_{i}_{j}.tif', unique_mask[0, 0, :, :, :].transpose((2, 1, 0)))
                io.imsave(f'predicted_prob_map_{i}_{j}.tif',
                          predicted_semantic_mask.numpy()[0, 0, :, :, :].transpose((2, 1, 0)))

            a = hcat.mask.Part(predicted_semantic_mask.numpy(), unique_mask, (x_ind[j - 1], y_ind[i - 1]))
            pickle.dump(a, open(
                path_chunk_storage + '/' + time.strftime("%y-%m-%d_%H-%M_") + str(time.monotonic_ns()) + '.maskpart', 'wb'))
            a = a.mask.astype(np.uint8)[0, 0, :, :, :].transpose(2, 1, 0)

            del unique_mask, seed, predicted_cell_candidate_list, image_slice_frcnn, image_slice

    del image

    print('Reconstructing Mask...', end='', flush=True)
    mask = utils.reconstruct_mask(path_chunk_storage)
    print('Done!', flush=True)

    print('Reconstructing Unique Mask...', end='', flush=True)
    unique_mask=utils.reconstruct_segmented(path_chunk_storage)
    print('Done!', flush=True)

    if save_plots:
        print('Saving Instance Mask...', end='', flush=True)
        io.imsave('test_mask.tif', mask[0, 0, :, :, :].transpose((2, 1, 0)))
        print('Done!', flush=True)

    if save_plots or True: # We always want to plot this guy
        print('Saving Unique Mask...', end='', flush=True)
        io.imsave('test_unqiue_mask.tif', unique_mask[0, 0, :, :, :].transpose((2, 1, 0)))
        print('Done!', flush=True)

    print('Saving Cells...',end='', flush=True)
    pickle.dump(all_cells, open('all_cells.pkl', 'wb'))
    print('Done!', flush=True)

    print(f'Caluclating spline fit of cochlea...',end=' ', flush=True)
    if mask.dtype == np.float:
        mask = torch.from_numpy(mask)
        mask.gt_(.5)
        mask = mask.numpy()

    cochlear_length, percent_base_to_apex, apex = utils.get_cochlear_length(mask[0,0,:,:,:].sum(-1), equal_spaced_distance=2)
    print('Done', flush=True)

    print(f'Assigning freq to cell...', end=' ', flush=True)
    for cell in all_cells:
        cell.set_frequency(cochlear_length, percent_base_to_apex)
    print('Done', flush=True)

    return mask, unique_mask, cell_list


if __name__ =='__main__':

    base = '/home/chris/Dropbox (Partners HealthCare)/HcUnet/maskfiles/'
    # base = '/home/chris/Dropbox (Partners HealthCare)/HcUnet/maskfiles/'
    # path = '/home/chris/Dropbox (Partners HealthCare)/HcUnet/Data/validate/Mar 6 AAV2-PHP.B-CMV11 m5.lif - m5.tif'
    path = None
    newfolder = time.strftime('%y%m%d%H%M')
    path_chunk = base+newfolder
    os.mkdir(base + newfolder)

    try:
        analyze(path=path, numchunks=3, save_plots=True, show_plots=True, path_chunk_storage=path_chunk)

    except:
        files = glob.glob(path_chunk+'/*')
        for file in files:
            os.remove(file)
        os.rmdir(path_chunk)
        raise RuntimeError('Error in analysis Script - Chunk Files have been removed.')

    files = glob.glob(path_chunk + '/*')
    for file in files:
        os.remove(file)
    os.rmdir(path_chunk)
