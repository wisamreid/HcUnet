import torch
import torchvision.ops
import numpy as np
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.feature
import skimage.segmentation
import skimage
import skimage.feature
import cv2
import utils
from haircell import HairCell
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# from multiprocessing import Pool
# import mask
# import scipy.ndimage
# import scipy.ndimage.morphology
# import transforms as t
# from scipy.interpolate import splprep, splev
# import pickle
# import glob
# import ray


def predict_segmentation_mask(unet, image, device, use_probability_map = False):
    """
    Uses pretrained unet model to predict semantic segmentation of hair cells.

    ALGORITHM:
    Remove inf and nan ->
    apply padding ->
    calculate indexes for unet based on image.shape ->
    apply Unet on slices of image based on indexes ->
    take only valid portion of the middle of each output mask ->
    construct full valid mask ->
    RETURN mask


    BLUR PROBABILITY MAP???

    :param model: Trained Unet Model from unet.py
    :param image: torch.Tensor image with transforms pre applied!!! [1, 4, X, Y, Z]
    :param device: 'cuda' or 'cpu'
    :return mask:

    """
    PAD_SIZE = (128, 128, 4)
    EVAL_IMAGE_SIZE = (300, 300, 15)

    mask = torch.zeros((1, 1, image.shape[2], image.shape[3], image.shape[4]), dtype=torch.float)
    im_shape = image.shape

    # inf and nan screw up model evaluation. Happens occasionally
    image[np.isnan(image)] = 0
    image[np.isinf(image)] = 1

    # Apply Padding
    image = utils.pad_image_with_reflections(torch.as_tensor(image), pad_size=PAD_SIZE)

    #  We now calculate the indicies for our image
    x_ind = utils.calculate_indexes(PAD_SIZE[0], EVAL_IMAGE_SIZE[0], im_shape[2], image.shape[2])
    y_ind = utils.calculate_indexes(PAD_SIZE[1], EVAL_IMAGE_SIZE[1], im_shape[3], image.shape[3])
    z_ind = utils.calculate_indexes(PAD_SIZE[2], EVAL_IMAGE_SIZE[2], im_shape[4], image.shape[4])

    iterations = 0
    max_iter = (len(x_ind) * len(y_ind) * len(z_ind))-1

    cell_candidates = None
    # Loop and apply unet
    for i, x in enumerate(x_ind):
        for j, y in enumerate(y_ind):
            for k, z in enumerate(z_ind):
                print(f'{" " * (len(str(max_iter)) - len(str(iterations)) + 1)}{iterations}/{max_iter}', end='')
                # t.to_tensor reshapes image to [B C X Y Z]!!!
                padded_image_slice = image[:, :, x[0]:x[1], y[0]:y[1]:, z[0]:z[1]].float().to(device)

                # Occasionally everything is just -1 in the whole mat. Skip for speed
                if (padded_image_slice.float() != -1).sum() == 0:
                    iterations += 1
                    for _ in f'{" " * (len(str(max_iter)) - len(str(iterations)) + 1)}{iterations}/{max_iter}':
                        print('\b \b', end='')
                    continue

                with torch.no_grad():
                    valid_out = unet(padded_image_slice)

                valid_out = valid_out[:,:,
                                     PAD_SIZE[0]:EVAL_IMAGE_SIZE[0]+PAD_SIZE[0],
                                     PAD_SIZE[1]:EVAL_IMAGE_SIZE[1]+PAD_SIZE[1],
                                     PAD_SIZE[2]:EVAL_IMAGE_SIZE[2]+PAD_SIZE[2]]

                # Perform an in place sigmoid function to save memory.
                # 1/ (1+exp(-x))
                valid_out.mul_(-1)
                valid_out.exp_()
                valid_out.add_(1)
                valid_out.pow_(-1)

                # Take pixels that are greater than 75% likely to be a cell.
                if not use_probability_map:
                    valid_out.gt_(.50)  # Greater Than
                    valid_out = valid_out.type(torch.uint8)
                    if mask.dtype != torch.uint8:
                        mask = mask.type(torch.uint8)

                # print(valid_out.dtype, mask.dtype, valid_out.max())

                try:
                    mask[:, :, x[0]:x[0]+EVAL_IMAGE_SIZE[0],
                               y[0]:y[0]+EVAL_IMAGE_SIZE[1],
                               z[0]:z[0]+EVAL_IMAGE_SIZE[2]] = valid_out
                except IndexError:
                    raise RuntimeError(f'Amount of padding is not sufficient.\nvalid_out.shape: {valid_out.shape}\neval_image_size: {EVAL_IMAGE_SIZE} ')

                iterations += 1
                for _ in f'{" " * (len(str(max_iter)) - len(str(iterations)) + 1)}{iterations}/{max_iter}':
                    print('\b \b', end='')


    return mask


def predict_cell_candidates(image, model, candidate_list = None, initial_coords=(0,0)):
    """
    Takes in an image in torch spec from the dataloader for unet and performs the 2D search for hair cells on each
    z plane, removes duplicates and spits back a list of hair cell locations, row identities, and probablilities

    Steps:
    process image ->
    load model ->
    loop through each slice ->
    compile list of dicts ->
    for each dict add a cell candidate to a master list ->
    if a close cell candidate has a higher probability, replace old candidate with new one ->
    return lists of master cell candidtates


    :param image:
    :param model:
    :return:
    """
    PAD_SIZE = (24, 24, 0)
    EVAL_IMAGE_SIZE = (500, 500, image.shape[-1])

    im_shape = image.shape[2::]
    x_ind = utils.calculate_indexes(PAD_SIZE[0], EVAL_IMAGE_SIZE[0], im_shape[0], image.shape[2])
    y_ind = utils.calculate_indexes(PAD_SIZE[1], EVAL_IMAGE_SIZE[1], im_shape[1], image.shape[3])

    with torch.no_grad():
        for i, x in enumerate(x_ind):
            for j, y in enumerate(y_ind):
                for z in range(image.shape[-1]): # ASSUME TORCH VALID IMAGES [B, C, X, Y, Z]

                    image_z_slice = image[:, :, x[0]:x[1], y[0]:y[1]:, z]
                    predicted_cell_locations = model(image_z_slice)
                    predicted_cell_locations = predicted_cell_locations[0]  # list of dicts (batch size) with each dict containing 'boxes' 'labels' 'scores'
                    z_level = torch.ones(len(predicted_cell_locations['scores'])) * z
                    predicted_cell_locations['z_level'] = z_level

                    for name in predicted_cell_locations:
                        predicted_cell_locations[name] = predicted_cell_locations[name].float()

                    if not candidate_list:
                        candidate_list = predicted_cell_locations
                    else:
                        candidate_list = utils.merge_cell_candidates(candidate_list, predicted_cell_locations, initial_coords=(x[0], y[0])) # Two dicts will get merged into one!

    # candidate_list['boxes'][:, [0, 2]] += initial_coords[1]
    # candidate_list['boxes'][:, [1, 3]] += initial_coords[0]

    return candidate_list


def generate_unique_segmentation_mask(predicted_semantic_mask, predicted_cell_candidate_list, image,  rejection_probability_threshold = .95):
    """
    Takes in a dict of predicted cell candiates, and a 5D mask image. Assigns a unique label to every cell in the chunk.

    S C R A T C H  P A D
    Algorithm:
    Save each hair cell as an object?
    Run watershed on each object?
    Save each object as an individual file maybe? <- slow af

    Maybe each HC object just contains the necessary metadata to get the og image info,
        only saves the mask
        calculates and saves some critical statistics of the gfp signal?
        calculates rough volume? (Helpful with trying to exclude outliers)

    A S S U M P T I O N S
    every cell is uniquely segmented and does not touch another cell. (bad assumption)

    :param predicted_semantic_mask:
        array bool with semantic segmentation results
    :param predicted_cell_candidate_list:
        dict of lists: {'boxes' 'labels' 'scores' 'centers'}
    :param image:
        base image that was segmented
    :return: list of cell objects
    """

    if len(predicted_cell_candidate_list['scores']) == 0:
        return None

    unique_cell_id = 0
    cells = []
    for i, (y1, x1, y2, x2) in enumerate(predicted_cell_candidate_list['boxes']):

        center = [int((x2-x1)/2), int((y2-y1)/2), int(predicted_cell_candidate_list['z_level'][i])]

        if x1 > image.shape[2]:
            continue
        elif y1 > image.shape[3]:
            continue
        if x2 > image.shape[2]:
            x2 = torch.tensor(image.shape[2] - 1).float()
        elif y2 > image.shape[3]:
            y2 = torch.tensor(image.shape[3] - 1).float()

        if predicted_cell_candidate_list['scores'][i] < rejection_probability_threshold:
            continue

        dx = [-10, 10]
        dy = [-10, 10]

        if (x1 + dx[0]) < 0:
            dx[0] = x1
        if (y1 + dy[0]) < 0:
            dy[0] = y1
        if (x2 + dx[1]) > image.shape[2]:
            dx[1] = image.shape[2] - x2
        if (y2 + dy[1]) > image.shape[3]:
            dy[1] = image.shape[3] - y2



        x1 = x1.clone()
        x2 = x2.clone()
        y1 = y1.clone()
        y2 = y2.clone()

        x1 += dx[0]
        x2 += dx[1]
        y1 += dy[0]
        y2 += dy[1]

        x1 = int(torch.round(x1.clone()))
        x2 = int(torch.round(x2.clone()))
        y1 = int(torch.round(y1.clone()))
        y2 = int(torch.round(y2.clone()))

        center[0] -= int(dx[0])
        center[1] -= int(dy[0])
        # print(center)

        image_slice = image[:, :, x1:x2, y1:y2, :]
        mask_slice = predicted_semantic_mask[:, :, x1:x2, y1:y2, :]

        cells.append(HairCell(image_coords=(x1,y1,x2,y2), center=center, image=image_slice, mask=mask_slice, id= unique_cell_id))

        unique_cell_id += 1

    return cells


def generate_unique_segmentation_mask_from_probability(predicted_semantic_mask, predicted_cell_candidate_list, image, rejection_probability_threshold=.95):
    """
    S C R A T C H  P A D

    Read in mask
    Read in list of cell candidates
    make seed matrix first!!!
    apply labels
    chunk up that shit
    Take only seeds from within the pad,
    run watershed on batch of cells
    take unique shindig and apply it to mask
    assign cell object to each unique mask
    ???
    Profit


    :param mask:
    :param predicted_cell_candidate_list:
    :param image:
    :param rejection_probability_threshold:
    :return:
    """

    # THESE DONT NECESSARILY HAVE TO BE THE SAME AS ABOVE.
    PAD_SIZE = (25, 25, 0)
    EVAL_IMAGE_SIZE = (200, 200, predicted_semantic_mask.shape[-1])
    num_dialate = 2

    im_shape = predicted_semantic_mask.shape[2::]
    x_ind = utils.calculate_indexes(PAD_SIZE[0], EVAL_IMAGE_SIZE[0], im_shape[0], predicted_semantic_mask.shape[2])
    y_ind = utils.calculate_indexes(PAD_SIZE[1], EVAL_IMAGE_SIZE[1], im_shape[1], predicted_semantic_mask.shape[3])

    iterations = 0
    max_iter = (len(x_ind) * len(y_ind))-1
    unique_mask = np.zeros(predicted_semantic_mask.shape, dtype=np.int32)
    seed = np.zeros(unique_mask.shape, dtype=np.int)

    if predicted_cell_candidate_list['boxes'] is None:
        Warning(f'No Predicted cells: {len(predicted_cell_candidate_list)}')
        return unique_mask, seed

    if len(predicted_cell_candidate_list['scores']) == 0:
        return unique_mask, seed

    unique_cell_id = 1
    cells = []
    for i, (y1, x1, y2, x2) in enumerate(predicted_cell_candidate_list['boxes']):

        center = [int((x2-x1)/2), int((y2-y1)/2), int(predicted_cell_candidate_list['z_level'][i])]

        if x1 > image.shape[2]:
            continue
        elif y1 > image.shape[3]:
            continue
        if x2 > image.shape[2]:
            x2 = torch.tensor(image.shape[2] - 1).float()
        elif y2 > image.shape[3]:
            y2 = torch.tensor(image.shape[3] - 1).float()

        if predicted_cell_candidate_list['scores'][i] < rejection_probability_threshold:
            continue

        dx = [-10, 10]
        dy = [-10, 10]

        if (x1 + dx[0]) < 0:
            dx[0] = x1
        if (y1 + dy[0]) < 0:
            dy[0] = y1
        if (x2 + dx[1]) > image.shape[2]:
            dx[1] = image.shape[2] - x2
        if (y2 + dy[1]) > image.shape[3]:
            dy[1] = image.shape[3] - y2

        x1 = x1.clone()
        x2 = x2.clone()
        y1 = y1.clone()
        y2 = y2.clone()

        x1 += dx[0]
        x2 += dx[1]
        y1 += dy[0]
        y2 += dy[1]

        x1 = int(torch.round(x1.clone()))
        x2 = int(torch.round(x2.clone()))
        y1 = int(torch.round(y1.clone()))
        y2 = int(torch.round(y2.clone()))

        # center[0] -= int(dx[0])
        # center[1] -= int(dy[0])
        # print(np.round(x1+(x2-x1)/2)), int(np.round(y1+(y2-y1)/2))

        image_slice = image[:, :, x1:x2, y1:y2, :]
        mask_slice = predicted_semantic_mask[:, :, x1:x2, y1:y2, :]
        seed[0, 0, int(np.round(x1+(x2-x1)/2)), int(np.round(y1+(y2-y1)/2)), center[2]] = int(unique_cell_id)

        # cells.append(HairCell(image_coords=(x1,y1,x2,y2), center=center, image=image_slice, mask=mask_slice, id= unique_cell_id))

        unique_cell_id += 1

    # print(unique_cell_id)
    # print(seed.max())

    for i, x in enumerate(x_ind):
        for j, y in enumerate(y_ind):
            print(f'{" " * (len(str(max_iter)) - len(str(iterations)) + 1)}{iterations}/{max_iter}', end='')

            mask_slice = predicted_semantic_mask[:, :, x[0]:x[1], y[0]:y[1]:, :]
            distance = np.zeros(mask_slice.shape)

            mask_slice_binary = mask_slice > 0.35

            for z in range(distance.shape[-1]):
                distance[0,0,:,:,i] = cv2.distanceTransform(mask_slice_binary[0, 0, :, :, i].astype(np.uint8), cv2.DIST_L2, 5)

            # for nul in range(num_dialate):
            #     mask_slice_binary = skimage.morphology.binary_dilation(mask_slice_binary)

            seed_slice = np.zeros(mask_slice.shape).astype(np.int)

            seed_slice[:, :, PAD_SIZE[0]:PAD_SIZE[0]+EVAL_IMAGE_SIZE[0], PAD_SIZE[0]:PAD_SIZE[0]+EVAL_IMAGE_SIZE[0], :] = seed[:, :, x[0]+PAD_SIZE[0]:x[1]-PAD_SIZE[0]+1, y[0]+PAD_SIZE[1]:y[1]-PAD_SIZE[1]+1, :]


            labels = skimage.segmentation.watershed(distance[0,0,:,:,:] * -1, seed_slice[0,0,:,:,:],
                                                    mask=mask_slice_binary[0,0,:,:,:],
                                                    watershed_line=True, compactness=2)

            # print(f' Watershed:{labels.max()}, SeedMax:{seed_slice.max()}, MaskSliceMax: {mask_slice.max()}, {seed[:, :, x[0]+PAD_SIZE[0]:x[1]-PAD_SIZE[0]+1, y[0]+PAD_SIZE[1]:y[1]-PAD_SIZE[1]+1, :].max()}', end='\n')
            unique_mask[0, 0, x[0]:x[1], y[0]:y[1]:, :][labels>0] = labels[labels>0]
            # poop?
            iterations += 1
            for _ in f'{" " * (len(str(max_iter)) - len(str(iterations)) + 1)}{iterations}/{max_iter}':
                print('\b \b', end='')


    return unique_mask, seed


def generate_cell_objects(image: torch.Tensor,  unique_mask):
    """
    Quick and dirty

    :param image: [B,C,X,Y,Z] torch tensor
    :param unique_mask: [X,Y,Z] numpy array
    :return:
    """

    cell_ids = np.unique(unique_mask)
    indicies = np.indices(unique_mask[0,0,:,:,:].shape) # Should be 3xMxN
    cell_list = []

    for id in cell_ids:
        if id == 0:
            continue

        mask = (unique_mask == id)[0,0,:,:,:]
        x_ind = indicies[0, :, :, :][mask]
        y_ind = indicies[1, :, :, :][mask]
        z_ind = indicies[2, :, :, :][mask]
        x = (x_ind.min(), x_ind.max())
        y = (y_ind.min(), y_ind.max())
        z = (z_ind.min(), z_ind.max())

        image_coords = [x[0], y[0], z[0], x[1], y[1], z[1]]
        center = [(x[1]-x[0])/2, (y[1]-y[0])/2, (z[1]-z[0])/2]
        image_slice = image[:, :, x[0]:x[1], y[0]:y[1], z[0]:z[1]]
        mask = mask[x[0]:x[1], y[0]:y[1], z[0]:z[1]]

        cell = HairCell(image_coords=image_coords, center=center, image=image_slice, mask=mask, id=id)
        cell_list.append(cell)

    return cell_list