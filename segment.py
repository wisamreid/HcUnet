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
import scipy.ndimage
import scipy.ndimage.morphology
import transforms as t
from scipy.interpolate import splprep, splev
import pickle
import glob
import ray
import cv2
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from multiprocessing import Pool
import mask
import utils


def predict_segmentation_mask(unet, image, device):
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

    :param model: Trained Unet Model from unet.py
    :param image: torch.Tensor image with transforms pre applied!!! [1, 4, X, Y, Z]
    :param device: 'cuda' or 'cpu'
    :return mask:

    """
    PAD_SIZE = (128, 128, 4)
    EVAL_IMAGE_SIZE = (300, 300, 20)

    mask = torch.zeros((1, 1, image.shape[2], image.shape[3], image.shape[4]), dtype=torch.bool)
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
                print(f'\r{iterations}/{max_iter} ', end=' ')
                # t.to_tensor reshapes image to [B C X Y Z]!!!
                padded_image_slice = image[:, :, x[0]:x[1], y[0]:y[1]:, z[0]:z[1]].float().to(device)

                # Occasionally everything is just -1 in the whole mat. Skip for speed
                if (padded_image_slice.float() != -1).sum() == 0:
                    iterations += 1
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
                valid_out.gt_(.75)  # Greater Than

                try:
                    mask[:, :, x[0]:x[0]+EVAL_IMAGE_SIZE[0],
                               y[0]:y[0]+EVAL_IMAGE_SIZE[1],
                               z[0]:z[0]+EVAL_IMAGE_SIZE[2]] = valid_out
                except IndexError:
                    raise RuntimeError(f'Amount of padding is not sufficient.\nvalid_out.shape: {valid_out.shape}\neval_image_size: {EVAL_IMAGE_SIZE} ')

                iterations += 1

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
                        predicted_cell_locations[name] = predicted_cell_locations[name].cpu()

                    if not candidate_list:
                        candidate_list = predicted_cell_locations
                    else:
                        candidate_list = utils.merge_cell_candidates(candidate_list, predicted_cell_locations, initial_coords=(x[0], y[0])) # Two dicts will get merged into one!

    # candidate_list['boxes'][:, [0, 2]] += initial_coords[1]
    # candidate_list['boxes'][:, [1, 3]] += initial_coords[0]

    return candidate_list


def generate_unique_segmentation_mask(predicted_semantic_mask, predicted_cell_candidate_list, image):
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

    for i, box_coords in enumerate(predicted_cell_candidate_list['boxes']):
        print(box_coords)

    return None


def generate_unique_segmentation_mask_distance(mask):
    """
    DEPRECIATED

    :param mask:
    :return:
    """
    raise DeprecationWarning

    # THESE DONT NECESSARILY HAVE TO BE THE SAME AS ABOVE.
    PAD_SIZE = (10, 10, 0)
    EVAL_IMAGE_SIZE = (500, 500, mask.shape[-1])

    im_shape = mask.shape[2::]
    x_ind = utils.calculate_indexes(PAD_SIZE[0], EVAL_IMAGE_SIZE[0], im_shape[0], mask.shape[2])
    y_ind = utils.calculate_indexes(PAD_SIZE[1], EVAL_IMAGE_SIZE[1], im_shape[1], mask.shape[3])

    iterations = 0
    max_iter = (len(x_ind) * len(y_ind))-1
    distance = np.zeros(mask.shape, dtype=np.float16)
    unique_mask = np.zeros(mask.shape, dtype=np.int32)

    @ray.remote
    def par_fun(x, y, PAD_SIZE, EVAL_IMAGE_SIZE, mask_part):
        if not mask_part.max():  # part should be a bool. Max would be True or False!
            out = np.zeros((1, 1, EVAL_IMAGE_SIZE[0], EVAL_IMAGE_SIZE[1], mask_part.shape[-1]), dtype=np.float16)
            return x, y, out, out.astype(np.int32)

        distance_part = np.zeros(mask_part.shape)
        local_maximum = np.zeros(mask_part.shape)
        for i in range(distance_part.shape[-1]):
            distance_part[0,0,:,:,i] = cv2.distanceTransform(mask_part[0,0,:,:,i].astype(np.uint8), cv2.DIST_L2, 5)

        distance_part = distance_part[:, :,
                        PAD_SIZE[0]:EVAL_IMAGE_SIZE[0] + PAD_SIZE[0],
                        PAD_SIZE[1]:EVAL_IMAGE_SIZE[1] + PAD_SIZE[1],
                        :]
        mask_part = mask_part[:, :,
                        PAD_SIZE[0]:EVAL_IMAGE_SIZE[0] + PAD_SIZE[0],
                        PAD_SIZE[1]:EVAL_IMAGE_SIZE[1] + PAD_SIZE[1],
                        :]

        distance_part = distance_part[0,0,:,:,:]
        local_maximum = skimage.feature.peak_local_max(distance_part, indices=False, min_distance=1)
        local_maximum = np.expand_dims(local_maximum, axis=(0,1))
        distance_part = np.expand_dims(distance_part, axis=(0,1))


        # print(local_maximum.shape, distance_part.max(), distance_part.min(), local_maximum.max())

        markers = scipy.ndimage.label(local_maximum)[0]

        labels = []
        labels = skimage.segmentation.watershed(-1*distance_part, markers, mask=mask_part, watershed_line=True)
        return x, y, distance_part.astype(np.float16), labels

    # Loop and apply unet
    distance_part_list=[]
    for i, x in enumerate(x_ind):
        for j, y in enumerate(y_ind):
            print(f'\r{iterations}/{max_iter} ', end=' ')
            mask_slice = mask[:, :, x[0]:x[1], y[0]:y[1]:, :] == 1

            distance_part_list.append(par_fun.remote(x, y, PAD_SIZE, EVAL_IMAGE_SIZE, mask_slice))
            iterations += 1

    distance_part_list = ray.get(distance_part_list)
    print('Finished Parallel Computation.')

    while distance_part_list:
        part = distance_part_list.pop(0)
        x = part[0]
        y = part[1]

        distance[:,
                 :,
                 x[0]:x[0]+EVAL_IMAGE_SIZE[0],
                 y[0]:y[0]+EVAL_IMAGE_SIZE[1],
                 :] = part[2]

        unique_mask[:,
                    :,
                    x[0]:x[0]+EVAL_IMAGE_SIZE[0],
                    y[0]:y[0]+EVAL_IMAGE_SIZE[1],
                    :] = part[3]

    return distance, unique_mask